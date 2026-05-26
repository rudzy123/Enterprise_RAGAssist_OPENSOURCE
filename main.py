import json
import logging
import os
import time
import uuid
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    DEBUG_MODE,
    FINAL_K,
    MIN_CHUNK_SIMILARITY,
    MIN_SIMILARITY_THRESHOLD,
    NOT_FOUND_ANSWER,
    RETRIEVE_K,
)
from answer_generation.generation import generate_answer_from_chunks
from retrieval.retrieve_chunks import retrieve_similar_chunks
from retrieval.similarity import chunk_similarity_score, max_similarity
from observability import TraceStore, log_event, setup_json_logger
from observability.request_log import RequestLogger

# -------------------------------------------------
# App
# -------------------------------------------------

app = FastAPI(title="Enterprise RAG Assistant", debug=DEBUG_MODE)

# -------------------------------------------------
# Models
# -------------------------------------------------

class Question(BaseModel):
    question: str


class RetrievedChunk(BaseModel):
    chunk_id: str
    rank: int
    source_file: str
    section_title: str
    document_source: str
    similarity_score: float
    distance: float
    text: str
    text_preview: str
    rerank_score: Optional[float] = None


class Answer(BaseModel):
    answer: str
    sources: List[str]
    confidence: float
    confidence_reason: Optional[str] = None
    top_k: int
    retrieved_chunks: Optional[List[RetrievedChunk]] = None
    trace_id: Optional[str] = None


# -------------------------------------------------
# Setup
# -------------------------------------------------

logger = setup_json_logger()
trace_store = TraceStore()
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="enterprise_docs"
)

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------

def _chunk_sources(chunks: List[dict]) -> List[str]:
    return [c.get("document_source") or c["source_file"] for c in chunks]


def _serialize_chunks(chunks: List[dict]) -> List[RetrievedChunk]:
    return [RetrievedChunk(**chunk) for chunk in chunks]


def embed(text: str):
    return embedding_model.encode(text).tolist()


def estimate_token_usage(question: str, answer: str) -> int:
    return max(1, len(question) // 4 + len(answer) // 4)


def determine_failure_type(
    retrieved_chunks: List[dict],
    confidence: float,
    top_similarity: Optional[float],
    answer_text: str,
    groundedness_score: Optional[float] = None,
) -> str:
    if not retrieved_chunks:
        return "weak_retrieval"

    if top_similarity is not None and top_similarity < MIN_SIMILARITY_THRESHOLD:
        return "weak_retrieval"

    low_confidence_text = answer_text.lower()
    if NOT_FOUND_ANSWER.lower() in low_confidence_text:
        return "partial_context"
    if "do not have enough information" in low_confidence_text:
        return "partial_context"

    if groundedness_score is not None and groundedness_score < 70.0:
        return "hallucination"

    return "success"


def log_step(trace_id: str, event: str, details: dict = None):
    """Legacy helper; prefer RequestLogger for new code."""
    log_event(logger, event, trace_id=trace_id, **(details or {}))


def save_trace(trace: dict):
    trace_store.save_trace(trace)
    logger.info(json.dumps({"event": "trace_saved", "trace_id": trace["trace_id"]}), extra={"extra": {"trace_id": trace["trace_id"], "event": "trace_saved"}})


def load_curated_markdown(directory: str):
    """
    Load curated markdown documents from disk and split by section headers.
    Each section becomes an individual retrieval unit.
    """
    documents = []

    for filename in os.listdir(directory):
        if not filename.endswith(".md"):
            continue

        path = os.path.join(directory, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        sections = content.split("\n## ")
        for section in sections:
            section_text = section.strip()
            if not section_text:
                continue

            documents.append(
                {
                    "text": section_text,
                    "metadata": {
                        "source_file": filename
                    }
                }
            )

    return documents


# -------------------------------------------------
# Routes
# -------------------------------------------------

@app.post("/ingest")
def ingest_docs():
    """
    Ingest curated markdown documents from data/docs/curated into the vector store.
    """
    docs_path = "data/docs/curated"
    
    if not os.path.exists(docs_path):
        return {
            "error": "Document directory not found",
            "details": f"Expected directory '{docs_path}' does not exist",
            "status": "failed"
        }

    try:
        md_files = [f for f in os.listdir(docs_path) if f.endswith('.md')]
        if not md_files:
            return {
                "error": "No markdown files found",
                "details": f"No .md files found in '{docs_path}'",
                "status": "failed"
            }
    except OSError as e:
        return {
            "error": "Directory access error",
            "details": f"Cannot access directory '{docs_path}': {str(e)}",
            "status": "failed"
        }

    documents = load_curated_markdown(docs_path)
    for idx, doc in enumerate(documents):
        collection.add(
            ids=[f"doc_{idx}"],
            embeddings=[embed(doc["text"])],
            documents=[doc["text"]],
            metadatas=[doc["metadata"]],
        )

    return {
        "status": "ingested",
        "documents_ingested": len(documents),
        "source_directory": docs_path
    }


@app.post("/ask", response_model=Answer)
def ask(question: Question, final_k: int = FINAL_K):
    """
    Answer questions using retrieved evidence only.
    Confidence is based on retrieval quality (similarity scores, document count, source consolidation).
    Refuse to answer when confidence is low.
    """
    trace_id = str(uuid.uuid4())
    started_at = datetime.utcnow()
    request_log = RequestLogger(trace_id=trace_id, query=question.question)
    request_log.bind_logger(logger)
    request_log.log_query(question.question)

    retrieval_started = time.perf_counter()
    request_log.add_step("retrieval_started", {"retrieve_k": RETRIEVE_K, "final_k": final_k})

    retrieval = retrieve_similar_chunks(
        question.question,
        retrieve_k=RETRIEVE_K,
        final_k=final_k,
        return_trace=True,
        trace_id=trace_id,
    )

    retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000.0
    filtered_chunks = retrieval["chunks"]
    raw_candidates = retrieval["raw_candidates"]

    filtered_similarity_scores = [chunk_similarity_score(c) for c in filtered_chunks]
    top_similarity = max_similarity(
        filtered_similarity_scores
        or [chunk_similarity_score(c) for c in raw_candidates]
    )

    request_log.log_retrieved_chunks(filtered_chunks)
    request_log.add_step(
        "retrieval_completed",
        {
            "raw_candidate_count": len(raw_candidates),
            "filtered_count": len(filtered_chunks),
            "rerank_enabled": retrieval.get("rerank_enabled", False),
            "top_similarity": top_similarity,
            "min_similarity_threshold": MIN_CHUNK_SIMILARITY,
            "max_chunks_per_file": retrieval["max_chunks_per_file"],
        },
    )
    request_log.add_step("relevance_scored", {"top_similarity": top_similarity})

    generation_started = time.perf_counter()
    request_log.add_step("generation_started", {"source_count": len(filtered_chunks)})

    answer_text, confidence, confidence_reason, generated_tokens, generation_obs = (
        generate_answer_from_chunks(
            question.question,
            filtered_chunks,
            use_llm=bool(os.getenv("OPENAI_API_KEY")),
            raw_candidate_count=len(raw_candidates),
        )
    )
    generation_latency_ms = (time.perf_counter() - generation_started) * 1000.0

    if generation_obs.get("llm_prompt"):
        request_log.log_llm_prompt(generation_obs["llm_prompt"])
    if generation_obs.get("model_response") is not None:
        request_log.log_model_response(
            generation_obs["model_response"],
            token_usage=generated_tokens,
        )

    generation_mode = generation_obs.get("generation_mode", "unknown")
    request_log.log_answer(answer_text, generation_mode=generation_mode)
    request_log.add_step(
        "confidence_computed",
        {
            "confidence": confidence,
            "confidence_reason": confidence_reason,
            "filtered_doc_count": len(filtered_chunks),
        },
    )

    total_latency_ms = (datetime.utcnow() - started_at).total_seconds() * 1000.0
    request_log.log_latency(
        total_ms=total_latency_ms,
        retrieval_ms=retrieval_latency_ms,
        generation_ms=generation_latency_ms,
    )

    token_usage = (
        generated_tokens
        if generated_tokens is not None
        else estimate_token_usage(question.question, answer_text)
    )
    groundedness_score = confidence * 100.0
    failure_type = determine_failure_type(
        filtered_chunks,
        confidence,
        top_similarity,
        answer_text,
        groundedness_score,
    )

    trace = request_log.to_trace_dict()
    trace.update({
        "answer": answer_text,
        "confidence": confidence,
        "confidence_reason": confidence_reason,
        "groundedness_score": groundedness_score,
        "failure_type": failure_type,
        "token_usage": token_usage,
        "evaluation": {
            "top_similarity": top_similarity,
            "confidence": confidence,
            "confidence_reason": confidence_reason,
            "groundedness_score": groundedness_score,
        },
    })
    save_trace(trace)
    request_log.save_json_file()

    log_event(logger, "request_completed", trace_id=trace_id, latency_ms=total_latency_ms)

    if DEBUG_MODE:
        print(f"TRACE SUMMARY: {trace_id}")
        print(f"  query={question.question}")
        print(f"  failure_type={failure_type}")
        print(f"  latency_ms={total_latency_ms:.1f}")
        print(f"  retrieved_chunks={len(filtered_chunks)}")

    return Answer(
        answer=answer_text,
        sources=_chunk_sources(filtered_chunks) if filtered_chunks else [],
        confidence=confidence,
        confidence_reason=confidence_reason,
        top_k=final_k,
        retrieved_chunks=_serialize_chunks(filtered_chunks) if filtered_chunks else [],
        trace_id=trace_id,
    )


@app.get("/traces/recent")
def get_recent_traces():
    return {"recent_traces": trace_store.get_recent_traces(limit=20)}


@app.get("/traces/{trace_id}")
def get_trace(trace_id: str):
    trace = trace_store.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace
