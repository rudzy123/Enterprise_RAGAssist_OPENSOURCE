"""
Reciprocal Rank Fusion (RRF) for combining dense and sparse retrieval rankings.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from observability import setup_json_logger

logger = setup_json_logger("enterprise_rag.hybrid")

# Human-readable rank fields attached to fused chunks.
DENSE_RANK_FIELD = "dense_rank"
BM25_RANK_FIELD = "bm25_rank"


def _validate_alpha(alpha: float) -> float:
    """Clamp-check dense weight used in weighted RRF."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"hybrid_alpha must be between 0.0 and 1.0, got {alpha}")
    return alpha


def reciprocal_rank_fusion(
    ranked_lists: List[List[dict]],
    *,
    weights: Optional[List[float]] = None,
    rrf_k: int = 60,
    id_key: str = "chunk_id",
) -> List[dict]:
    """
    Fuse multiple ranked candidate lists with weighted Reciprocal Rank Fusion.

    score(chunk) = sum_i weight_i / (rrf_k + rank_i)

    Returns chunks sorted by descending fusion score. Each chunk includes:
    - rrf_score
    - dense_rank / bm25_rank when present in the corresponding list
    """
    if not ranked_lists:
        logger.debug("RRF received no ranked lists; returning empty result.")
        return []

    if rrf_k < 1:
        raise ValueError(f"rrf_k must be >= 1, got {rrf_k}")

    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights length must match ranked_lists length")

    fused_scores: Dict[str, float] = {}
    chunk_lookup: Dict[str, dict] = {}
    rank_fields = [DENSE_RANK_FIELD, BM25_RANK_FIELD]

    for list_idx, ranked in enumerate(ranked_lists):
        weight = weights[list_idx]
        rank_field = (
            rank_fields[list_idx]
            if list_idx < len(rank_fields)
            else f"fusion_rank_{list_idx}"
        )

        for rank, chunk in enumerate(ranked, start=1):
            chunk_id = chunk.get(id_key)
            if not chunk_id:
                logger.warning(
                    "Skipping ranked chunk without chunk_id during RRF.",
                    extra={"extra": {"list_idx": list_idx, "rank": rank}},
                )
                continue

            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (
                weight / (rrf_k + rank)
            )

            if chunk_id not in chunk_lookup:
                chunk_lookup[chunk_id] = dict(chunk)
            else:
                # Preserve the richest metadata/text from either retrieval leg.
                chunk_lookup[chunk_id].update(chunk)

            chunk_lookup[chunk_id][rank_field] = rank

    fused = []
    for chunk_id, score in fused_scores.items():
        merged = chunk_lookup[chunk_id]
        merged["rrf_score"] = score
        merged["chunk_id"] = chunk_id
        fused.append(merged)

    fused.sort(
        key=lambda c: (
            c.get("rrf_score", 0.0),
            c.get("similarity_score", 0.0),
            c.get("bm25_score_normalized", 0.0),
        ),
        reverse=True,
    )
    return fused


def fuse_hybrid_candidates(
    dense_candidates: List[dict],
    sparse_candidates: List[dict],
    *,
    alpha: float,
    rrf_k: int = 60,
) -> Tuple[List[dict], dict]:
    """
    Fuse dense and BM25 candidate lists with production-safe fallbacks.

    Fallback rules (hybrid remains functional even when one leg is empty):
    - alpha == 1.0 or sparse empty -> dense ordering
    - alpha == 0.0 or dense empty  -> sparse ordering
    - both non-empty               -> weighted RRF

    Returns:
        (fused_candidates, fusion_stats)
    """
    alpha = _validate_alpha(alpha)
    sparse_weight = 1.0 - alpha

    stats = {
        "hybrid_alpha": alpha,
        "rrf_k": rrf_k,
        "dense_count": len(dense_candidates),
        "sparse_count": len(sparse_candidates),
        "fusion_mode": "rrf",
        "warnings": [],
    }

    if not dense_candidates and not sparse_candidates:
        stats["fusion_mode"] = "empty"
        stats["fused_count"] = 0
        logger.info("Hybrid fusion produced no candidates from either leg.")
        return [], stats

    if alpha >= 1.0 or not sparse_candidates:
        if not sparse_candidates and alpha < 1.0:
            stats["warnings"].append("bm25_returned_no_candidates")
        stats["fusion_mode"] = "dense_only"
        stats["fused_count"] = len(dense_candidates)
        logger.info(
            "Hybrid fusion using dense-only ordering.",
            extra={"extra": stats},
        )
        return list(dense_candidates), stats

    if alpha <= 0.0 or not dense_candidates:
        if not dense_candidates and alpha > 0.0:
            stats["warnings"].append("dense_returned_no_candidates")
        stats["fusion_mode"] = "sparse_only"
        stats["fused_count"] = len(sparse_candidates)
        logger.info(
            "Hybrid fusion using BM25-only ordering.",
            extra={"extra": stats},
        )
        return list(sparse_candidates), stats

    fused = reciprocal_rank_fusion(
        [dense_candidates, sparse_candidates],
        weights=[alpha, sparse_weight],
        rrf_k=rrf_k,
    )
    stats["fusion_mode"] = "rrf"
    stats["fused_count"] = len(fused)
    logger.info(
        "Hybrid RRF fusion completed.",
        extra={"extra": stats},
    )
    return fused, stats
