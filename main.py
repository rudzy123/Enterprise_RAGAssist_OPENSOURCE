import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from core.auth import APIKeyMiddleware, extract_api_key
from core.config import (
    API_KEY,
    DEBUG_MODE,
    FINAL_K,
    HYBRID_SEARCH,
    MAX_FINAL_K,
    MAX_QUESTION_LENGTH,
    MIN_CHUNK_SIMILARITY,
    MIN_SIMILARITY_THRESHOLD,
    NOT_FOUND_ANSWER,
    RATE_LIMIT,
    RETRIEVE_K,
)
from core.embeddings import get_embedding_model
from core.vector_store import collection_count
from answer_generation.generation import generate_answer_from_chunks
from ingestion.pipeline import ingest_corpus
from retrieval.retrieve_chunks import retrieve_similar_chunks
from retrieval.similarity import chunk_similarity_score, max_similarity
from observability import TraceStore, log_event, setup_json_logger
from observability.request_log import RequestLogger

# -------------------------------------------------
# App
# -------------------------------------------------

app = FastAPI(title="Enterprise RAG Assistant", debug=DEBUG_MODE)
app.add_middleware(APIKeyMiddleware)


def _rate_limit_key(request: Request) -> str:
    api_key = extract_api_key(request)
    if api_key:
        return api_key
    return get_remote_address(request)


limiter = Limiter(key_func=_rate_limit_key)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# -------------------------------------------------
# Models
# -------------------------------------------------


class Question(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH)


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


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    collection_count: int
    embedding_model_loaded: bool


# -------------------------------------------------
# Setup
# -------------------------------------------------

logger = setup_json_logger()
trace_store = TraceStore()

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------


def _chunk_sources(chunks: List[dict]) -> List[str]:
    return [c.get("document_source") or c["source_file"] for c in chunks]


def _serialize_chunks(chunks: List[dict]) -> List[RetrievedChunk]:
    return [RetrievedChunk(**chunk) for chunk in chunks]


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


def save_trace(trace: dict):
    trace_store.save_trace(trace)
    logger.info(
        json.dumps({"event": "trace_saved", "trace_id": trace["trace_id"]}),
        extra={"extra": {"trace_id": trace["trace_id"], "event": "trace_saved"}},
    )


# -------------------------------------------------
# Routes
# -------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness probe: process is running."""
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadyResponse)
def ready():
    """Readiness probe: dependencies initialized and corpus indexed."""
    count = collection_count()
    model_loaded = get_embedding_model() is not None
    if count <= 0:
        return ReadyResponse(
            status="not_ready",
            collection_count=count,
            embedding_model_loaded=model_loaded,
        )
    return ReadyResponse(
        status="ready",
        collection_count=count,
        embedding_model_loaded=model_loaded,
    )


@app.post("/ingest", deprecated=True)
@limiter.limit(RATE_LIMIT)
def ingest_docs(request: Request):
    """
    Deprecated: use `python ingestion/embed_and_store.py` or `ingestion.pipeline.ingest_corpus()`.

    Runs the unified ingestion pipeline (section-aware chunking, batch embed, cosine Chroma).
    """
    try:
        result = ingest_corpus(reset=True, verbose=False)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return {
        **result,
        "deprecated": True,
        "message": "Use `python ingestion/embed_and_store.py` for canonical ingestion.",
    }


@app.post("/ask", response_model=Answer)
@limiter.limit(RATE_LIMIT)
def ask(
    request: Request,
    question: Question,
    final_k: int = Query(default=FINAL_K, ge=1, le=MAX_FINAL_K),
    hybrid_search: Optional[bool] = Query(default=None, description="Enable dense+BM25 RRF fusion"),
    source_file: Optional[str] = Query(default=None, description="Filter by source_file metadata"),
    doc_type: Optional[str] = Query(default=None, description="Filter by doc_type metadata"),
):
    """
    Answer questions using retrieved evidence only.
    Confidence is based on retrieval quality (similarity scores, document count, source consolidation).
    Refuse to answer when confidence is low.
    """
    metadata_filters = {}
    if source_file:
        metadata_filters["source_file"] = source_file
    if doc_type:
        metadata_filters["doc_type"] = doc_type

    use_hybrid = HYBRID_SEARCH if hybrid_search is None else hybrid_search

    trace_id = str(uuid.uuid4())
    started_at = datetime.utcnow()
    request_log = RequestLogger(trace_id=trace_id, query=question.question)
    request_log.bind_logger(logger)
    request_log.log_query(question.question)

    retrieval_started = time.perf_counter()
    request_log.add_step(
        "retrieval_started",
        {
            "retrieve_k": RETRIEVE_K,
            "final_k": final_k,
            "hybrid_search": use_hybrid,
            "metadata_filters": metadata_filters or None,
        },
    )

    retrieval = retrieve_similar_chunks(
        question.question,
        retrieve_k=RETRIEVE_K,
        final_k=final_k,
        hybrid_search=use_hybrid,
        metadata_filters=metadata_filters or None,
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
            "hybrid_search": retrieval.get("hybrid_search", False),
            "top_similarity": top_similarity,
            "min_similarity_threshold": MIN_CHUNK_SIMILARITY,
            "max_chunks_per_file": retrieval["max_chunks_per_file"],
            "metadata_filters": retrieval.get("metadata_filters"),
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
    trace.update(
        {
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
        }
    )
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
@limiter.limit(RATE_LIMIT)
def get_recent_traces(request: Request):
    return {"recent_traces": trace_store.get_recent_traces(limit=20)}


@app.get("/traces/{trace_id}")
@limiter.limit(RATE_LIMIT)
def get_trace(request: Request, trace_id: str):
    trace = trace_store.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace


@app.on_event("startup")
def log_security_state():
    if not API_KEY:
        logger.warning(
            json.dumps(
                {
                    "event": "security_warning",
                    "message": "API_KEY is not set; API endpoints are unauthenticated.",
                }
            )
        )
