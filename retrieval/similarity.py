"""
Cosine similarity scoring for Chroma vector search results.

All retrieval and confidence code should use these helpers so scores are
comparable across the API, CLI, evals, and UI.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

Number = Union[int, float]


def cosine_similarity_from_distance(cosine_distance: Number) -> float:
    """
    Convert Chroma cosine distance to cosine similarity.

    Chroma collections with ``metadata={"hnsw:space": "cosine"}`` return
    cosine distance, where:

        cosine_distance = 1 - cosine_similarity

    Therefore:

        similarity = 1 - cosine_distance
    """
    return 1.0 - float(cosine_distance)


def similarities_from_distances(cosine_distances: Sequence[Number]) -> List[float]:
    """Map a sequence of cosine distances to cosine similarities."""
    return [cosine_similarity_from_distance(d) for d in cosine_distances]


def max_similarity(similarity_scores: Sequence[Number]) -> Optional[float]:
    """Return the highest similarity score, or None if the sequence is empty."""
    if not similarity_scores:
        return None
    return float(max(similarity_scores))


def chunk_similarity_score(chunk: dict) -> float:
    """
    Read ``similarity_score`` from a chunk dict, or derive it from ``distance``.

    Raises:
        ValueError: If neither field is present.
    """
    if chunk.get("similarity_score") is not None:
        return float(chunk["similarity_score"])
    if chunk.get("distance") is not None:
        return cosine_similarity_from_distance(chunk["distance"])
    raise ValueError("Chunk must include 'similarity_score' or 'distance'")
