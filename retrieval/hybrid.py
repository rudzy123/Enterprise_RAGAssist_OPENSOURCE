"""
Reciprocal Rank Fusion (RRF) for combining dense and sparse retrieval rankings.
"""

from __future__ import annotations

from typing import Dict, List, Optional


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
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights length must match ranked_lists length")

    fused_scores: Dict[str, float] = {}
    chunk_lookup: Dict[str, dict] = {}
    rank_fields = ["dense_rank", "bm25_rank", "fusion_rank_0", "fusion_rank_1"]

    for list_idx, ranked in enumerate(ranked_lists):
        weight = weights[list_idx]
        rank_field = rank_fields[list_idx] if list_idx < len(rank_fields) else f"fusion_rank_{list_idx}"

        for rank, chunk in enumerate(ranked, start=1):
            chunk_id = chunk.get(id_key)
            if not chunk_id:
                continue

            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (
                weight / (rrf_k + rank)
            )

            if chunk_id not in chunk_lookup:
                chunk_lookup[chunk_id] = dict(chunk)
            else:
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
