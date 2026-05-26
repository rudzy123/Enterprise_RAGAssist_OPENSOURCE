"""Retrieval confidence scoring for answer gating."""

from typing import List, Tuple

from config import LOW_CONFIDENCE_THRESHOLD, MIN_CHUNK_SIMILARITY, MIN_SIMILARITY_THRESHOLD
from retrieval.similarity import chunk_similarity_score, max_similarity


def compute_retrieval_confidence(
    num_docs: int,
    similarity_scores: List[float],
    metadatas: List[dict],
) -> Tuple[float, str]:
    """
    Compute confidence from retrieval quality signals.

    ``similarity_scores`` must be cosine similarities (1 - cosine distance).
    """
    if num_docs == 0:
        return 0.0, "No relevant documents found"

    doc_count_score = min(num_docs / 3.0, 1.0)
    if similarity_scores:
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
    else:
        avg_similarity = 0.5

    sources = [m.get("source_file", "unknown") for m in metadatas if m]
    unique_sources = len(set(sources))
    source_consistency = 1.0 if unique_sources == 1 else 0.85

    confidence = (
        0.55 * avg_similarity
        + 0.30 * doc_count_score
        + 0.15 * source_consistency
    )
    final_confidence = min(1.0, max(0.0, confidence))

    reason_parts = []
    if num_docs == 1:
        reason_parts.append("Single section retrieved")
    elif num_docs >= 3:
        reason_parts.append("Multiple sections retrieved")
    else:
        reason_parts.append(f"{num_docs} sections retrieved")

    if avg_similarity >= 0.75:
        reason_parts.append("high similarity to query")
    elif avg_similarity >= 0.5:
        reason_parts.append("moderate similarity to query")
    else:
        reason_parts.append("low similarity to query")

    if unique_sources == 1:
        reason_parts.append("from same document")
    else:
        reason_parts.append(f"from {unique_sources} different documents")

    return final_confidence, " ".join(reason_parts)


def is_low_confidence(confidence: float) -> bool:
    return confidence < LOW_CONFIDENCE_THRESHOLD


def assess_retrieval_context(
    chunks: List[dict],
    *,
    raw_candidate_count: int = 0,
) -> Tuple[float, str, bool]:
    """
    Score retrieval quality and decide whether context is too weak to answer.

    Returns:
        (confidence, confidence_reason, is_weak_context)
    """
    if not chunks:
        if raw_candidate_count == 0:
            return 0.0, "No documents matched the query", True
        return (
            0.0,
            f"No retrieved chunks met similarity threshold ({MIN_CHUNK_SIMILARITY})",
            True,
        )

    metadatas = [
        {"source_file": c["source_file"], "section_title": c.get("section_title")}
        for c in chunks
    ]
    similarity_scores = [chunk_similarity_score(c) for c in chunks]
    confidence, reason = compute_retrieval_confidence(
        num_docs=len(chunks),
        similarity_scores=similarity_scores,
        metadatas=metadatas,
    )

    top_similarity = max_similarity(similarity_scores)
    if top_similarity is not None and top_similarity < MIN_SIMILARITY_THRESHOLD:
        weak_reason = (
            f"Top similarity score ({top_similarity:.2f}) below relevance threshold "
            f"({MIN_SIMILARITY_THRESHOLD})"
        )
        return 0.0, weak_reason, True

    if is_low_confidence(confidence):
        weak_reason = (
            f"{reason}; below confidence threshold ({LOW_CONFIDENCE_THRESHOLD})"
        )
        return confidence, weak_reason, True

    return confidence, reason, False
