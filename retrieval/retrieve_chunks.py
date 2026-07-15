"""
Retrieval from Chroma (dense), optional BM25 (sparse), and weighted RRF fusion.

Pipeline:
  dense (+ optional BM25 + RRF) -> min-similarity/BM25 filter ->
  cross-encoder rerank -> per-document cap -> final_k.

When HYBRID_SEARCH=false (default), behavior matches the original dense-only path.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from core.config import (
    BM25_MIN_SCORE,
    BM25_RETRIEVE_K,
    CHROMA_COLLECTION_NAME,
    FINAL_K,
    HYBRID_ALPHA,
    HYBRID_SEARCH,
    MAX_CHUNKS_PER_FILE,
    MIN_CHUNK_SIMILARITY,
    RERANK_ENABLED,
    RERANK_MODEL,
    RERANK_TOP_N,
    RETRIEVAL_STRUCTURED_LOGS,
    RETRIEVE_K,
    RRF_K,
)
from core.embeddings import encode_query
from core.vector_store import get_collection
from observability import log_event, setup_json_logger
from retrieval.bm25_store import bm25_search
from retrieval.hybrid import fuse_hybrid_candidates
from retrieval.metadata_filter import build_chroma_where, normalize_metadata_filters
from retrieval.rerank import rerank_chunks
from retrieval.result_format import enrich_retrieved_chunks
from retrieval.similarity import cosine_similarity_from_distance
from retrieval.structured_logs import log_retrieval_pipeline

COLLECTION_NAME = CHROMA_COLLECTION_NAME

_retrieval_logger: Optional[logging.Logger] = None


def _get_retrieval_logger() -> logging.Logger:
    global _retrieval_logger
    if _retrieval_logger is None:
        _retrieval_logger = setup_json_logger("enterprise_rag.retrieval")
    return _retrieval_logger


def _log_retrieval_event(event: str, *, trace_id: Optional[str] = None, **details: Any) -> None:
    """Emit a structured retrieval-stage log entry."""
    log_event(_get_retrieval_logger(), event, trace_id=trace_id, **details)


def _apply_per_document_cap(chunks: List[dict], max_per_file: int) -> List[dict]:
    """Keep top chunks per source_file (chunks must already be ranked)."""
    counts: dict[str, int] = {}
    capped: List[dict] = []
    for chunk in chunks:
        source = chunk["source_file"]
        if counts.get(source, 0) >= max_per_file:
            continue
        counts[source] = counts.get(source, 0) + 1
        capped.append(chunk)
    return capped


def _chunk_from_chroma_row(
    chunk_id: str,
    distance: float,
    doc: str,
    metadata: dict,
) -> dict:
    similarity = cosine_similarity_from_distance(distance)
    return {
        "text": doc,
        "source_file": metadata.get("source_file", "Unknown"),
        "section_title": metadata.get("section_title", "Unknown"),
        "doc_type": metadata.get("doc_type", "curated_md"),
        "similarity_score": similarity,
        "distance": distance,
        "chunk_id": chunk_id,
    }


def _dense_search(
    query: str,
    *,
    retrieve_k: int,
    collection_name: str,
    metadata_filters: Optional[Dict[str, Any]],
    trace_id: Optional[str] = None,
) -> tuple[List[dict], dict]:
    """
    Run dense vector retrieval against Chroma.

    Returns:
        (candidates, search_stats)
    """
    stats: dict = {
        "retrieve_k": retrieve_k,
        "metadata_filters": metadata_filters,
        "warnings": [],
    }

    try:
        collection = get_collection(collection_name, create_if_missing=False)
    except Exception as exc:
        stats["warnings"].append("chroma_collection_unavailable")
        stats["error"] = str(exc)
        _log_retrieval_event(
            "dense_search_failed",
            trace_id=trace_id,
            error=str(exc),
            collection_name=collection_name,
        )
        return [], stats

    query_embedding = encode_query(query)
    where = build_chroma_where(metadata_filters)
    collection_count = collection.count()
    n_results = min(retrieve_k, max(collection_count, 1))

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        query_kwargs["where"] = where

    try:
        results = collection.query(**query_kwargs)
    except Exception as exc:
        stats["warnings"].append("chroma_query_failed")
        stats["error"] = str(exc)
        _log_retrieval_event(
            "dense_search_failed",
            trace_id=trace_id,
            error=str(exc),
            where=where,
        )
        return [], stats

    raw_candidates: List[dict] = []
    for chunk_id, distance, doc, metadata in zip(
        results["ids"][0],
        results["distances"][0],
        results["documents"][0],
        results["metadatas"][0],
    ):
        raw_candidates.append(_chunk_from_chroma_row(chunk_id, distance, doc, metadata))

    raw_candidates.sort(key=lambda c: c["similarity_score"], reverse=True)
    stats["collection_count"] = collection_count
    stats["returned_count"] = len(raw_candidates)
    _log_retrieval_event(
        "dense_search_completed",
        trace_id=trace_id,
        returned_count=len(raw_candidates),
        collection_count=collection_count,
        metadata_filters=metadata_filters,
    )
    return raw_candidates, stats


def _apply_retrieval_threshold(
    candidates: List[dict],
    *,
    min_similarity: float,
    hybrid_search: bool,
) -> List[dict]:
    """
    Apply post-fusion relevance thresholds.

    Dense-only mode keeps the original cosine-similarity gate.
    Hybrid mode also admits BM25-strong chunks that may have low dense scores.
    """
    if hybrid_search:
        return [
            c
            for c in candidates
            if c.get("similarity_score", 0.0) >= min_similarity
            or c.get("bm25_score_normalized", 0.0) >= BM25_MIN_SCORE
        ]
    return [c for c in candidates if c.get("similarity_score", 0.0) >= min_similarity]


def retrieve_similar_chunks(
    query: str,
    retrieve_k: int = RETRIEVE_K,
    final_k: int = FINAL_K,
    top_k: Optional[int] = None,
    collection_name: str = COLLECTION_NAME,
    min_similarity: float = MIN_CHUNK_SIMILARITY,
    max_chunks_per_file: int = MAX_CHUNKS_PER_FILE,
    rerank_enabled: bool = RERANK_ENABLED,
    rerank_top_n: int = RERANK_TOP_N,
    hybrid_search: Optional[bool] = None,
    hybrid_alpha: Optional[float] = None,
    bm25_retrieve_k: Optional[int] = None,
    rrf_k: Optional[int] = None,
    metadata_filters: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
    return_trace: bool = False,
    trace_id: Optional[str] = None,
    structured_logs: bool = RETRIEVAL_STRUCTURED_LOGS,
) -> Union[List[dict], dict]:
    """
    Retrieve chunks using dense search, or hybrid dense+BM25 with weighted RRF.

    Hybrid search is **off by default** (`HYBRID_SEARCH=false`). When disabled,
    this function behaves identically to the original dense-only implementation.

    Args:
        query: Plain text search query
        retrieve_k: Number of dense vector candidates
        final_k: Maximum chunks returned after filtering and caps
        top_k: Deprecated alias for final_k
        collection_name: Chroma collection name
        min_similarity: Minimum cosine similarity for dense matches
        max_chunks_per_file: Max chunks retained per source_file
        rerank_enabled: Apply cross-encoder reranking after fusion/filter
        rerank_top_n: Max candidates passed to the reranker
        hybrid_search: Enable BM25 + RRF fusion (default: HYBRID_SEARCH config)
        hybrid_alpha: Dense weight in weighted RRF (default: HYBRID_ALPHA config)
        bm25_retrieve_k: BM25 candidate count (default: BM25_RETRIEVE_K config)
        rrf_k: RRF rank constant (default: RRF_K config)
        metadata_filters: Optional filters, e.g. {"source_file": "policy.md", "doc_type": "policy"}
        verbose: Enable debug prints
        return_trace: If True, return dict with chunks and intermediate pipeline stages
        trace_id: Optional trace ID attached to structured log events
        structured_logs: Emit JSON logs for each pipeline stage

    Returns:
        List of chunk dicts, or trace dict when return_trace=True
    """
    if top_k is not None:
        final_k = top_k

    # Default-off hybrid preserves backward compatibility for all existing callers.
    use_hybrid = HYBRID_SEARCH if hybrid_search is None else hybrid_search
    alpha = HYBRID_ALPHA if hybrid_alpha is None else hybrid_alpha
    sparse_k = BM25_RETRIEVE_K if bm25_retrieve_k is None else bm25_retrieve_k
    rrf_constant = RRF_K if rrf_k is None else rrf_k

    retrieval_warnings: List[str] = []
    dense_stats: dict = {}
    sparse_stats: dict = {}
    fusion_stats: dict = {}

    try:
        filters = normalize_metadata_filters(metadata_filters)
    except ValueError:
        # Propagate invalid filters to API callers; CLI callers get an empty result.
        if return_trace:
            raise
        _log_retrieval_event(
            "retrieval_invalid_metadata_filters",
            trace_id=trace_id,
            metadata_filters=metadata_filters,
        )
        return []

    dense_candidates: List[dict] = []
    sparse_candidates: List[dict] = []
    raw_candidates: List[dict] = []

    _log_retrieval_event(
        "retrieval_started",
        trace_id=trace_id,
        hybrid_search=use_hybrid,
        hybrid_alpha=alpha,
        retrieve_k=retrieve_k,
        bm25_retrieve_k=sparse_k,
        metadata_filters=filters,
    )

    # --- Dense leg (always executed) ---
    dense_candidates, dense_stats = _dense_search(
        query,
        retrieve_k=retrieve_k,
        collection_name=collection_name,
        metadata_filters=filters,
        trace_id=trace_id,
    )
    retrieval_warnings.extend(dense_stats.get("warnings", []))

    # --- Sparse + fusion leg (hybrid only) ---
    if use_hybrid:
        try:
            sparse_candidates, sparse_stats = bm25_search(
                query,
                top_k=sparse_k,
                collection_name=collection_name,
                metadata_filters=filters,
            )
            retrieval_warnings.extend(sparse_stats.get("warnings", []))

            if not sparse_stats.get("index_available", True):
                retrieval_warnings.append("hybrid_degraded_to_dense_only")

            raw_candidates, fusion_stats = fuse_hybrid_candidates(
                dense_candidates,
                sparse_candidates,
                alpha=alpha,
                rrf_k=rrf_constant,
            )
            retrieval_warnings.extend(fusion_stats.get("warnings", []))

            _log_retrieval_event(
                "hybrid_fusion_completed",
                trace_id=trace_id,
                fusion_mode=fusion_stats.get("fusion_mode"),
                dense_count=fusion_stats.get("dense_count"),
                sparse_count=fusion_stats.get("sparse_count"),
                fused_count=fusion_stats.get("fused_count"),
                warnings=fusion_stats.get("warnings"),
            )
        except ValueError as exc:
            # Invalid alpha or RRF configuration — fall back to dense ordering.
            retrieval_warnings.append("hybrid_fusion_failed")
            raw_candidates = list(dense_candidates)
            fusion_stats = {"fusion_mode": "dense_fallback", "error": str(exc)}
            _log_retrieval_event(
                "hybrid_fusion_failed",
                trace_id=trace_id,
                error=str(exc),
            )
        except Exception as exc:
            retrieval_warnings.append("hybrid_sparse_leg_failed")
            raw_candidates = list(dense_candidates)
            fusion_stats = {"fusion_mode": "dense_fallback", "error": str(exc)}
            _log_retrieval_event(
                "hybrid_sparse_leg_failed",
                trace_id=trace_id,
                error=str(exc),
            )
    else:
        raw_candidates = dense_candidates

    if not raw_candidates and dense_stats.get("error"):
        empty = [] if not return_trace else _empty_trace(
            retrieve_k=retrieve_k,
            final_k=final_k,
            min_similarity=min_similarity,
            max_chunks_per_file=max_chunks_per_file,
            rerank_enabled=rerank_enabled,
            hybrid_search=use_hybrid,
            hybrid_alpha=alpha,
            bm25_retrieve_k=sparse_k,
            rrf_k=rrf_constant,
            metadata_filters=filters,
            warnings=retrieval_warnings,
        )
        return empty

    threshold_passed = _apply_retrieval_threshold(
        raw_candidates,
        min_similarity=min_similarity,
        hybrid_search=use_hybrid,
    )

    if use_hybrid and not threshold_passed and raw_candidates:
        retrieval_warnings.append("hybrid_threshold_filtered_all_candidates")

    rerank_input = threshold_passed[:rerank_top_n]
    if rerank_enabled and rerank_input:
        try:
            reranked = rerank_chunks(query, rerank_input)
        except Exception as exc:
            retrieval_warnings.append("rerank_failed")
            reranked = list(rerank_input)
            _log_retrieval_event("rerank_failed", trace_id=trace_id, error=str(exc))
    else:
        reranked = list(rerank_input)

    capped = _apply_per_document_cap(reranked, max_chunks_per_file)
    final_chunks = enrich_retrieved_chunks(capped[:final_k])

    if structured_logs:
        log_retrieval_pipeline(
            _get_retrieval_logger(),
            query,
            trace_id=trace_id,
            retrieve_k=retrieve_k,
            final_k=final_k,
            min_similarity=min_similarity,
            max_chunks_per_file=max_chunks_per_file,
            rerank_enabled=rerank_enabled,
            rerank_model=RERANK_MODEL if rerank_enabled else None,
            raw_candidates=raw_candidates,
            threshold_passed=threshold_passed,
            reranked=reranked,
            final_chunks=final_chunks,
        )

    _log_retrieval_event(
        "retrieval_completed",
        trace_id=trace_id,
        hybrid_search=use_hybrid,
        raw_candidate_count=len(raw_candidates),
        threshold_passed_count=len(threshold_passed),
        final_count=len(final_chunks),
        warnings=retrieval_warnings,
    )

    if verbose:
        print(f"\nHybrid search: {use_hybrid}")
        if use_hybrid:
            print(f"Dense candidates: {len(dense_candidates)}")
            print(f"BM25 candidates: {len(sparse_candidates)}")
            print(f"Fusion mode: {fusion_stats.get('fusion_mode', 'n/a')}")
            print(f"RRF alpha (dense weight): {alpha}")
        print(f"Fused/raw candidates: {len(raw_candidates)}")
        if filters:
            print(f"Metadata filters: {filters}")
        if retrieval_warnings:
            print(f"Warnings: {retrieval_warnings}")
        print(f"Above threshold: {len(threshold_passed)}")
        print(f"Rerank enabled: {rerank_enabled} ({len(rerank_input)} scored)")
        print(f"After per-file cap ({max_chunks_per_file}): {len(capped)}")
        print(f"Final (k={final_k}): {len(final_chunks)}")
        for i, c in enumerate(final_chunks, 1):
            score_label = (
                f"rerank={c['rerank_score']:.3f}"
                if "rerank_score" in c
                else f"rrf={c.get('rrf_score', 0):.4f}"
                if use_hybrid
                else f"sim={c['similarity_score']:.3f}"
            )
            print(f"  {i}. {c['source_file']} ({score_label})")

    if return_trace:
        return {
            "chunks": final_chunks,
            "top_k": final_k,
            "raw_candidates": raw_candidates,
            "dense_candidates": dense_candidates,
            "sparse_candidates": sparse_candidates,
            "threshold_passed": threshold_passed,
            "reranked": reranked,
            "capped": enrich_retrieved_chunks(capped),
            "retrieve_k": retrieve_k,
            "final_k": final_k,
            "min_similarity": min_similarity,
            "max_chunks_per_file": max_chunks_per_file,
            "rerank_enabled": rerank_enabled,
            "rerank_top_n": rerank_top_n,
            "hybrid_search": use_hybrid,
            "hybrid_alpha": alpha,
            "bm25_retrieve_k": sparse_k,
            "rrf_k": rrf_constant,
            "metadata_filters": filters,
            "dense_stats": dense_stats,
            "sparse_stats": sparse_stats,
            "fusion_stats": fusion_stats,
            "warnings": retrieval_warnings,
        }

    return final_chunks


def _empty_trace(
    *,
    retrieve_k: int,
    final_k: int,
    min_similarity: float,
    max_chunks_per_file: int,
    rerank_enabled: bool,
    hybrid_search: bool = False,
    hybrid_alpha: float = HYBRID_ALPHA,
    bm25_retrieve_k: int = BM25_RETRIEVE_K,
    rrf_k: int = RRF_K,
    metadata_filters: Optional[dict] = None,
    warnings: Optional[List[str]] = None,
) -> dict:
    return {
        "chunks": [],
        "top_k": final_k,
        "raw_candidates": [],
        "dense_candidates": [],
        "sparse_candidates": [],
        "threshold_passed": [],
        "reranked": [],
        "capped": [],
        "retrieve_k": retrieve_k,
        "final_k": final_k,
        "min_similarity": min_similarity,
        "max_chunks_per_file": max_chunks_per_file,
        "rerank_enabled": rerank_enabled,
        "rerank_top_n": RERANK_TOP_N,
        "hybrid_search": hybrid_search,
        "hybrid_alpha": hybrid_alpha,
        "bm25_retrieve_k": bm25_retrieve_k,
        "rrf_k": rrf_k,
        "metadata_filters": metadata_filters,
        "dense_stats": {},
        "sparse_stats": {},
        "fusion_stats": {},
        "warnings": warnings or [],
    }


if __name__ == "__main__":
    test_query = "What is incident response?"
    chunks = retrieve_similar_chunks(test_query, verbose=True)
    print(f"\nReturned {len(chunks)} chunks")
