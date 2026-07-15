import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.exceptions import HTTPException as StarletteHTTPException

from answer_generation.generation import generate_answer_from_chunks
from core.auth import APIKeyMiddleware, extract_api_key
from core.cache import cache_ping
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
from ingestion.pipeline import ingest_corpus
from observability import TraceStore, configure_logging, log_event, setup_json_logger
from observability.request_log import RequestLogger
from retrieval.bm25_store import bm25_index_exists
from retrieval.metadata_filter import merge_metadata_filters
from retrieval.retrieve_chunks import retrieve_similar_chunks
from retrieval.similarity import chunk_similarity_score, max_similarity

# -------------------------------------------------
# App factory / lifespan
# -------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: logging, security checks. Shutdown: flush log handlers."""
    configure_logging()
    app.state.logger = setup_json_logger("enterprise_rag.api")

    if not API_KEY:
        app.state.logger.warning(
            json.dumps(
                {
                    "event": "security_warning",
                    "message": "API_KEY is not set; API endpoints are unauthenticated.",
                }
            )
        )
    else:
        app.state.logger.info(
            json.dumps({"event": "startup", "message": "API key authentication enabled."})
        )

    # Warm embedding model so /ready reflects true startup cost.
    try:
        get_embedding_model()
        app.state.logger.info(json.dumps({"event": "startup", "embedding_model": "loaded"}))
    except Exception as exc:
        app.state.logger.error(
            json.dumps({"event": "startup_error", "component": "embedding_model", "error": str(exc)})
        )

    yield

    logging.shutdown()


app = FastAPI(
    title="Enterprise RAG Assistant",
    debug=DEBUG_MODE,
    lifespan=lifespan,
)
app.add_middleware(APIKeyMiddleware)


def _rate_limit_key(request: Request) -> str:
    api_key = extract_api_key(request)
    if api_key:
        return api_key
    return get_remote_address(request)


limiter = Limiter(key_func=_rate_limit_key)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize logging early so exception handlers can emit structured logs.
configure_logging()
logger = setup_json_logger("enterprise_rag.api")
trace_store = TraceStore()

# -------------------------------------------------
# Models
# -------------------------------------------------


class Question(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH)
    hybrid_search: Optional[bool] = Field(
        default=None,
        description="Enable dense+BM25 hybrid retrieval. Defaults to HYBRID_SEARCH env (false).",
    )
    metadata_filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description='Optional metadata filters, e.g. {"doc_type": "policy", "source_file": "access_control_policy.md"}',
    )


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
    bm25_index_available: bool
    redis_available: Optional[bool] = None
    checks: Dict[str, bool]


# -------------------------------------------------
# Exception handlers (sanitized responses in production)
# -------------------------------------------------


def _error_detail(message: str, exc: Optional[Exception] = None) -> str:
    if DEBUG_MODE and exc is not None:
        return f"{message}: {exc}"
    return message


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(
        json.dumps({"event": "validation_error", "path": request.url.path, "errors": exc.errors()})
    )
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error(
        json.dumps({"event": "unhandled_exception", "path": request.url.path, "error": str(exc)}),
        exc_info=exc,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": _error_detail("Internal server error", exc)},
    )


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
    """Liveness probe: process is running (always HTTP 200)."""
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadyResponse)
def ready():
    """
    Readiness probe: corpus indexed and models available.

    Returns HTTP 503 when the service should not receive traffic.
    """
    count = collection_count()
    model_loaded = get_embedding_model() is not None
    bm25_available = bm25_index_exists()
    redis_available: Optional[bool] = cache_ping() if os.getenv("REDIS_URL") else None

    checks = {
        "collection_populated": count > 0,
        "embedding_model_loaded": model_loaded,
        "bm25_index_available": bm25_available,
    }
    if redis_available is not None:
        checks["redis_available"] = redis_available

    # BM25 is required only when hybrid search is enabled globally.
    required_checks = ["collection_populated", "embedding_model_loaded"]
    if HYBRID_SEARCH:
        required_checks.append("bm25_index_available")

    is_ready = all(checks.get(name, False) for name in required_checks)

    payload = ReadyResponse(
        status="ready" if is_ready else "not_ready",
        collection_count=count,
        embedding_model_loaded=model_loaded,
        bm25_index_available=bm25_available,
        redis_available=redis_available,
        checks=checks,
    )

    if not is_ready:
        return JSONResponse(status_code=503, content=payload.model_dump())

    return payload


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
    hybrid_search: Optional[bool] = Query(
        default=None,
        description="Query-param override for hybrid retrieval (body field takes precedence when both set)",
    ),
    source_file: Optional[str] = Query(
        default=None,
        description="Shortcut metadata filter for source_file",
    ),
    doc_type: Optional[str] = Query(
        default=None,
        description="Shortcut metadata filter for doc_type",
    ),
    section_title: Optional[str] = Query(
        default=None,
        description="Shortcut metadata filter for section_title",
    ),
):
    """
    Answer questions using retrieved evidence only.
    Confidence is based on retrieval quality (similarity scores, document count, source consolidation).
    Refuse to answer when confidence is low.

    Hybrid search is off by default. Enable per-request via `hybrid_search: true` in the
    JSON body or `?hybrid_search=true` query param (requires BM25 index from ingestion).

    Metadata filters may be supplied in the request body as `metadata_filters` and/or via
    query-param shortcuts (`source_file`, `doc_type`, `section_title`).
    """
    try:
        metadata_filters = merge_metadata_filters(
            question.metadata_filters,
            {
                k: v
                for k, v in {
                    "source_file": source_file,
                    "doc_type": doc_type,
                    "section_title": section_title,
                }.items()
                if v
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Body field wins over query param when both are provided.
    if question.hybrid_search is not None:
        use_hybrid = question.hybrid_search
    elif hybrid_search is not None:
        use_hybrid = hybrid_search
    else:
        use_hybrid = HYBRID_SEARCH

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
            "metadata_filters": metadata_filters,
        },
    )

    try:
        retrieval = retrieve_similar_chunks(
            question.question,
            retrieve_k=RETRIEVE_K,
            final_k=final_k,
            hybrid_search=use_hybrid,
            metadata_filters=metadata_filters,
            return_trace=True,
            trace_id=trace_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

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
            "fusion_mode": (retrieval.get("fusion_stats") or {}).get("fusion_mode"),
            "retrieval_warnings": retrieval.get("warnings", []),
            "top_similarity": top_similarity,
            "min_similarity_threshold": MIN_CHUNK_SIMILARITY,
            "max_chunks_per_file": retrieval["max_chunks_per_file"],
            "metadata_filters": retrieval.get("metadata_filters"),
        },
    )
    request_log.add_step("relevance_scored", {"top_similarity": top_similarity})

    generation_started = time.perf_counter()
    request_log.add_step("generation_started", {"source_count": len(filtered_chunks)})

    try:
        answer_text, confidence, confidence_reason, generated_tokens, generation_obs = (
            generate_answer_from_chunks(
                question.question,
                filtered_chunks,
                use_llm=bool(os.getenv("OPENAI_API_KEY")),
                raw_candidate_count=len(raw_candidates),
            )
        )
    except Exception as exc:
        logger.error(
            json.dumps({"event": "generation_failed", "trace_id": trace_id, "error": str(exc)}),
            exc_info=exc,
        )
        raise HTTPException(
            status_code=502,
            detail=_error_detail("Answer generation failed", exc),
        ) from exc

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


# Note: startup logic lives in the lifespan context manager above.
