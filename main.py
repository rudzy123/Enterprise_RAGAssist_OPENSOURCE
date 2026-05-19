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
from answer_generation.confidence import compute_retrieval_confidence, is_low_confidence
from answer_generation.generation import (
    generate_retrieval_only_answer,
    generate_with_openai,
)
from retrieval.retrieve_chunks import retrieve_similar_chunks
from retrieval.similarity import chunk_similarity_score, max_similarity
from observability import TraceStore, build_step_log, log_event, setup_json_logger

# -------------------------------------------------
# App
# -------------------------------------------------

app = FastAPI(title="Enterprise RAG Assistant", debug=DEBUG_MODE)

# -------------------------------------------------
# Models
# -------------------------------------------------

class Question(BaseModel):
    question: str


class Answer(BaseModel):
    answer: str
    sources: List[str]
    confidence: float
    confidence_reason: Optional[str] = None
    retrieved_chunks: Optional[List[dict]] = None
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
    step = build_step_log(event, details)
    log_event(logger, event, trace_id=trace_id, details=details or {})
    return step


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
    step_logs = []
    answer_text = ""
    generated_tokens = None
    groundedness_score = None
    failure_type = "success"

    log_step(trace_id, "request_received", {"query": question.question})

    step_logs.append(build_step_log("query_embedding_created", {"query_length": len(question.question)}))
    log_step(trace_id, "retrieval_started", {"retrieve_k": RETRIEVE_K, "final_k": final_k})

    retrieval = retrieve_similar_chunks(
        question.question,
        retrieve_k=RETRIEVE_K,
        final_k=final_k,
        return_trace=True,
        trace_id=trace_id,
    )

    filtered_chunks = retrieval["chunks"]
    retrieved_chunks = retrieval["threshold_passed"]
    raw_candidates = retrieval["raw_candidates"]

    filtered_docs = [c["text"] for c in filtered_chunks]
    filtered_metas = [
        {"source_file": c["source_file"], "section_title": c["section_title"]}
        for c in filtered_chunks
    ]
    filtered_similarity_scores = [chunk_similarity_score(c) for c in filtered_chunks]

    step_logs.append(build_step_log("retrieval_completed", {
        "raw_candidate_count": len(raw_candidates),
        "threshold_passed_count": len(retrieved_chunks),
        "reranked_count": len(retrieval.get("reranked", [])),
        "filtered_count": len(filtered_chunks),
        "rerank_enabled": retrieval.get("rerank_enabled", False),
        "top_similarity": max_similarity(
            [chunk_similarity_score(c) for c in filtered_chunks]
            or [chunk_similarity_score(c) for c in raw_candidates]
        ),
        "min_similarity_threshold": MIN_CHUNK_SIMILARITY,
        "max_chunks_per_file": retrieval["max_chunks_per_file"],
    }))

    trace = {
        "trace_id": trace_id,
        "query": question.question,
        "retrieved_chunks": retrieved_chunks,
        "filtered_chunks": filtered_chunks,
        "created_at": started_at.isoformat() + "Z",
        "step_logs": step_logs,
    }

    if not raw_candidates:
        answer_text = NOT_FOUND_ANSWER
        failure_type = determine_failure_type(retrieved_chunks, 0.0, None, answer_text)
        trace.update({
            "answer": answer_text,
            "confidence": 0.0,
            "confidence_reason": "No documents matched the query",
            "groundedness_score": groundedness_score,
            "failure_type": failure_type,
            "latency_ms": 0.0,
            "token_usage": 0,
            "evaluation": {
                "retrieval_reason": "No docs retrieved"
            },
        })
        save_trace(trace)
        if DEBUG_MODE:
            print(json.dumps(trace, indent=2))
        return Answer(
            answer=answer_text,
            sources=[],
            confidence=0.0,
            confidence_reason="No documents matched the query",
            retrieved_chunks=retrieved_chunks,
            trace_id=trace_id,
        )

    if not filtered_chunks:
        answer_text = NOT_FOUND_ANSWER
        failure_type = determine_failure_type(retrieved_chunks, 0.0, None, answer_text)
        trace.update({
            "answer": answer_text,
            "confidence": 0.0,
            "confidence_reason": f"No retrieved chunks met similarity threshold ({MIN_CHUNK_SIMILARITY})",
            "groundedness_score": groundedness_score,
            "failure_type": failure_type,
            "latency_ms": 0.0,
            "token_usage": 0,
            "evaluation": {
                "filtered_count": len(filtered_chunks),
                "min_similarity_threshold": MIN_CHUNK_SIMILARITY,
            },
        })
        save_trace(trace)
        if DEBUG_MODE:
            print(json.dumps(trace, indent=2))
        return Answer(
            answer=answer_text,
            sources=[chunk["source_file"] for chunk in filtered_chunks],
            confidence=0.0,
            confidence_reason=f"No retrieved chunks met similarity threshold ({MIN_CHUNK_SIMILARITY})",
            retrieved_chunks=filtered_chunks,
            trace_id=trace_id,
        )

    top_similarity = max_similarity(filtered_similarity_scores)
    if filtered_chunks:
        log_step(trace_id, "relevance_scored", {"top_similarity": top_similarity})

        if top_similarity is not None and top_similarity < MIN_SIMILARITY_THRESHOLD:
            answer_text = NOT_FOUND_ANSWER
            failure_type = determine_failure_type(retrieved_chunks, 0.0, top_similarity, answer_text)
            trace.update({
                "answer": answer_text,
                "confidence": 0.0,
                "confidence_reason": (
                    f"Top similarity score ({top_similarity:.2f}) below relevance threshold "
                    f"({MIN_SIMILARITY_THRESHOLD})"
                ),
                "groundedness_score": groundedness_score,
                "failure_type": failure_type,
                "latency_ms": 0.0,
                "token_usage": 0,
                "evaluation": {
                    "top_similarity": top_similarity,
                    "relevance_threshold": MIN_SIMILARITY_THRESHOLD,
                },
            })
            save_trace(trace)
            if DEBUG_MODE:
                print(json.dumps(trace, indent=2))
            return Answer(
                answer=answer_text,
                sources=[m.get("source_file", "unknown") for m in filtered_metas],
                confidence=0.0,
                confidence_reason=(
                    f"Top similarity score ({top_similarity:.2f}) below relevance threshold "
                    f"({MIN_SIMILARITY_THRESHOLD})"
                ),
                retrieved_chunks=filtered_chunks,
                trace_id=trace_id,
            )
    else:
        log_step(trace_id, "relevance_scored", {"top_similarity": None})

    confidence, confidence_reason = compute_retrieval_confidence(
        num_docs=len(filtered_docs),
        similarity_scores=filtered_similarity_scores,
        metadatas=filtered_metas,
    )
    log_step(trace_id, "confidence_computed", {
        "confidence": confidence,
        "confidence_reason": confidence_reason,
        "filtered_doc_count": len(filtered_docs),
    })

    if is_low_confidence(confidence):
        answer_text = NOT_FOUND_ANSWER
        failure_type = determine_failure_type(retrieved_chunks, confidence, top_similarity, answer_text)
        trace.update({
            "answer": answer_text,
            "confidence": confidence,
            "confidence_reason": confidence_reason,
            "groundedness_score": groundedness_score,
            "failure_type": failure_type,
            "latency_ms": 0.0,
            "token_usage": 0,
            "evaluation": {
                "confidence_threshold": 0.3,
                "confidence_reason": confidence_reason,
            },
        })
        save_trace(trace)
        if DEBUG_MODE:
            print(json.dumps(trace, indent=2))
        return Answer(
            answer=answer_text,
            sources=[m.get("source_file", "unknown") for m in filtered_metas],
            confidence=confidence,
            confidence_reason=confidence_reason,
            retrieved_chunks=filtered_chunks,
            trace_id=trace_id,
        )

    log_step(trace_id, "generation_started", {"source_count": len(filtered_chunks)})
    if not os.getenv("OPENAI_API_KEY"):
        answer_text = generate_retrieval_only_answer(question.question, filtered_chunks)
        generated_tokens = estimate_token_usage(question.question, answer_text)
        log_step(trace_id, "generation_fallback", {"mode": "retrieval_only"})
    else:
        answer_text, generated_tokens = generate_with_openai(
            question.question,
            filtered_chunks,
        )
        log_step(trace_id, "generation_completed", {"generated_tokens": generated_tokens})

    token_usage = generated_tokens if generated_tokens is not None else estimate_token_usage(question.question, answer_text)
    groundedness_score = confidence * 100.0
    failure_type = determine_failure_type(retrieved_chunks, confidence, top_similarity, answer_text, groundedness_score)

    trace.update({
        "answer": answer_text,
        "confidence": confidence,
        "confidence_reason": confidence_reason,
        "groundedness_score": groundedness_score,
        "failure_type": failure_type,
        "latency_ms": (datetime.utcnow() - started_at).total_seconds() * 1000.0,
        "token_usage": token_usage,
        "evaluation": {
            "top_similarity": top_similarity,
            "confidence": confidence,
            "confidence_reason": confidence_reason,
            "groundedness_score": groundedness_score,
        },
    })
    save_trace(trace)

    if DEBUG_MODE:
        print(f"TRACE SUMMARY: {trace_id}")
        print(f"  query={question.question}")
        print(f"  failure_type={failure_type}")
        print(f"  groundedness_score={groundedness_score}")
        print(f"  retrieved_chunks={len(retrieved_chunks)}")
        for chunk in retrieved_chunks:
            print(f"    - {chunk['source_file']} similarity={chunk['similarity_score']:.4f}")

    return Answer(
        answer=answer_text,
        sources=[m.get("source_file", "unknown") for m in filtered_metas],
        confidence=confidence,
        confidence_reason=confidence_reason,
        retrieved_chunks=filtered_chunks,
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
