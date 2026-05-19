"""Retrieval confidence scoring for answer gating."""

from typing import List, Tuple

from config import LOW_CONFIDENCE_THRESHOLD


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
