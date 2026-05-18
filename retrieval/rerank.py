"""
Cross-encoder reranking for retrieved chunks.

Uses ms-marco MiniLM to rescore (query, passage) pairs after bi-encoder retrieval.
"""

from __future__ import annotations

from typing import List, Optional

from sentence_transformers import CrossEncoder

from config import RERANK_MODEL

_cross_encoder: Optional[CrossEncoder] = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(RERANK_MODEL)
    return _cross_encoder


def rerank_chunks(query: str, chunks: List[dict]) -> List[dict]:
    """
    Rerank chunks by cross-encoder relevance to the query.

    Adds ``rerank_score`` to each chunk and returns a new list sorted descending.
    Bi-encoder ``similarity_score`` is preserved for thresholding and tracing.
    """
    if not chunks:
        return []

    model = _get_cross_encoder()
    pairs = [[query, chunk["text"]] for chunk in chunks]
    scores = model.predict(pairs)

    reranked = []
    for chunk, score in zip(chunks, scores):
        updated = dict(chunk)
        updated["rerank_score"] = float(score)
        reranked.append(updated)

    reranked.sort(key=lambda c: c["rerank_score"], reverse=True)
    return reranked
