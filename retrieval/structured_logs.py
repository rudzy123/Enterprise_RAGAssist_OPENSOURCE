"""
Structured JSON logs for the retrieval pipeline.

Emits one log event per stage: candidates, similarity filter, rerank, final selection.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from observability import log_event
from retrieval.result_format import (
    TEXT_PREVIEW_LEN,
    format_document_source,
    make_text_preview,
)


def chunk_log_entry(chunk: dict, rank: Optional[int] = None) -> dict:
    """Compact, JSON-serializable chunk summary for logs."""
    source_file = chunk.get("source_file", "Unknown")
    section_title = chunk.get("section_title", "Unknown")
    entry = {
        "chunk_id": chunk.get("chunk_id"),
        "source_file": source_file,
        "section_title": section_title,
        "document_source": chunk.get("document_source")
        or format_document_source(source_file, section_title),
        "similarity_score": _round_score(chunk.get("similarity_score")),
        "distance": _round_score(chunk.get("distance")),
        "rerank_score": _round_score(chunk.get("rerank_score")),
        "text_preview": chunk.get("text_preview")
        or make_text_preview(chunk.get("text") or ""),
    }
    if rank is not None:
        entry["rank"] = rank
    elif chunk.get("rank") is not None:
        entry["rank"] = chunk["rank"]
    return entry


def _round_score(value) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 4)


def log_retrieved_candidates(
    logger: logging.Logger,
    query: str,
    candidates: List[dict],
    *,
    trace_id: Optional[str] = None,
    retrieve_k: Optional[int] = None,
) -> None:
    log_event(
        logger,
        "retrieval_candidates",
        trace_id=trace_id,
        query=query,
        retrieve_k=retrieve_k,
        count=len(candidates),
        candidates=[chunk_log_entry(c, rank=i + 1) for i, c in enumerate(candidates)],
    )


def log_similarity_scores(
    logger: logging.Logger,
    query: str,
    candidates: List[dict],
    min_similarity: float,
    passed: List[dict],
    *,
    trace_id: Optional[str] = None,
) -> None:
    passed_ids = {c.get("chunk_id") for c in passed}
    scored = []
    for i, chunk in enumerate(candidates):
        entry = chunk_log_entry(chunk, rank=i + 1)
        entry["passed_threshold"] = chunk.get("chunk_id") in passed_ids
        scored.append(entry)

    log_event(
        logger,
        "retrieval_similarity_scores",
        trace_id=trace_id,
        query=query,
        min_similarity=min_similarity,
        total_candidates=len(candidates),
        passed_count=len(passed),
        candidates=scored,
    )


def log_rerank_scores(
    logger: logging.Logger,
    query: str,
    reranked: List[dict],
    *,
    trace_id: Optional[str] = None,
    rerank_enabled: bool = True,
    rerank_model: Optional[str] = None,
) -> None:
    log_event(
        logger,
        "retrieval_rerank_scores",
        trace_id=trace_id,
        query=query,
        rerank_enabled=rerank_enabled,
        rerank_model=rerank_model,
        count=len(reranked),
        candidates=[
            chunk_log_entry(c, rank=i + 1) for i, c in enumerate(reranked)
        ],
    )


def log_final_selection(
    logger: logging.Logger,
    query: str,
    final_chunks: List[dict],
    *,
    trace_id: Optional[str] = None,
    final_k: Optional[int] = None,
    max_chunks_per_file: Optional[int] = None,
) -> None:
    log_event(
        logger,
        "retrieval_final_selection",
        trace_id=trace_id,
        query=query,
        final_k=final_k,
        max_chunks_per_file=max_chunks_per_file,
        count=len(final_chunks),
        selected=[chunk_log_entry(c, rank=i + 1) for i, c in enumerate(final_chunks)],
    )


def log_retrieval_results(
    logger: logging.Logger,
    query: str,
    final_chunks: List[dict],
    *,
    trace_id: Optional[str] = None,
    top_k: Optional[int] = None,
) -> None:
    """Emit a consolidated structured log of the final retrieval output."""
    log_event(
        logger,
        "retrieval_results",
        trace_id=trace_id,
        query=query,
        top_k=top_k,
        count=len(final_chunks),
        results=[chunk_log_entry(c) for c in final_chunks],
    )


def log_retrieval_pipeline(
    logger: logging.Logger,
    query: str,
    *,
    trace_id: Optional[str] = None,
    retrieve_k: int,
    final_k: int,
    min_similarity: float,
    max_chunks_per_file: int,
    rerank_enabled: bool,
    rerank_model: Optional[str],
    raw_candidates: List[dict],
    threshold_passed: List[dict],
    reranked: List[dict],
    final_chunks: List[dict],
) -> None:
    """Emit structured logs for all retrieval stages."""
    log_retrieved_candidates(
        logger,
        query,
        raw_candidates,
        trace_id=trace_id,
        retrieve_k=retrieve_k,
    )
    log_similarity_scores(
        logger,
        query,
        raw_candidates,
        min_similarity,
        threshold_passed,
        trace_id=trace_id,
    )
    log_rerank_scores(
        logger,
        query,
        reranked,
        trace_id=trace_id,
        rerank_enabled=rerank_enabled,
        rerank_model=rerank_model,
    )
    log_final_selection(
        logger,
        query,
        final_chunks,
        trace_id=trace_id,
        final_k=final_k,
        max_chunks_per_file=max_chunks_per_file,
    )
    log_retrieval_results(
        logger,
        query,
        final_chunks,
        trace_id=trace_id,
        top_k=final_k,
    )
