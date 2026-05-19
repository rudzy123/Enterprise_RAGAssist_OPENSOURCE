"""
Retrieval evaluation metrics: precision@k, MRR, recall, and abstention.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Set


def normalize_expected_sources(question_obj: dict) -> List[str]:
    expected = question_obj.get("source_doc_id") or []
    if isinstance(expected, str):
        return [expected]
    return list(expected)


def _chunk_source(chunk: dict) -> str:
    return chunk.get("source_file") or "Unknown"


def is_chunk_relevant(chunk: dict, expected_sources: Set[str]) -> bool:
    if not expected_sources:
        return True
    return _chunk_source(chunk) in expected_sources


def precision_at_k(ranked_chunks: Sequence[dict], expected_sources: Iterable[str], k: int) -> float:
    """
    Fraction of the top-k ranked slots occupied by relevant chunks.

    Denominator is always k (standard precision@k definition).
    """
    if k <= 0:
        return 0.0
    expected = set(expected_sources)
    if not expected:
        return 1.0

    top_k = list(ranked_chunks)[:k]
    relevant = sum(1 for chunk in top_k if is_chunk_relevant(chunk, expected))
    return relevant / k


def recall_at_k(ranked_chunks: Sequence[dict], expected_sources: Iterable[str], k: int) -> float:
    """Fraction of expected source documents found in the top-k chunk list."""
    expected = set(expected_sources)
    if not expected:
        return 1.0

    top_k = list(ranked_chunks)[:k]
    returned_sources = {_chunk_source(c) for c in top_k}
    found = expected & returned_sources
    return len(found) / len(expected)


def hit_at_k(ranked_chunks: Sequence[dict], expected_sources: Iterable[str], k: int) -> bool:
    expected = set(expected_sources)
    if not expected:
        return True
    top_k = list(ranked_chunks)[:k]
    return any(is_chunk_relevant(chunk, expected) for chunk in top_k)


def reciprocal_rank(ranked_chunks: Sequence[dict], expected_sources: Iterable[str]) -> float:
    """Reciprocal rank of the first relevant chunk (0.0 if none)."""
    expected = set(expected_sources)
    if not expected:
        return 1.0

    for rank, chunk in enumerate(ranked_chunks, start=1):
        if is_chunk_relevant(chunk, expected):
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(reciprocal_ranks: Sequence[float]) -> float:
    if not reciprocal_ranks:
        return 0.0
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def compute_retrieval_metrics(
    question_obj: dict,
    *,
    final_chunks: List[dict],
    ranked_chunks: List[dict],
    k: int,
    abstained: bool,
    latency_ms: float = 0.0,
) -> dict:
    """
    Compute retrieval metrics for one question.

    Args:
        question_obj: Eval question with ``source_doc_id``
        final_chunks: Chunks returned to the caller (after final_k)
        ranked_chunks: Full ranked list for MRR (typically post-rerank, pre final_k)
        k: Cutoff for precision@k / recall@k (usually ``final_k``)
        abstained: True when the pipeline returned no chunks
        latency_ms: Retrieval latency
    """
    expected_sources = normalize_expected_sources(question_obj)
    expected_set = set(expected_sources)

    returned_sources = list({_chunk_source(c) for c in final_chunks})
    relevant_in_returned = expected_set & set(returned_sources)

    retrieval_hit = bool(relevant_in_returned) if expected_set else True
    retrieval_precision = (
        len(relevant_in_returned) / len(returned_sources) if returned_sources else 1.0
    )
    retrieval_recall = (
        len(relevant_in_returned) / len(expected_set) if expected_set else 1.0
    )

    rr = reciprocal_rank(ranked_chunks, expected_sources)

    return {
        "retrieval_hit": retrieval_hit,
        "retrieval_precision": retrieval_precision,
        "retrieval_recall": retrieval_recall,
        "precision_at_k": precision_at_k(final_chunks, expected_sources, k),
        "recall_at_k": recall_at_k(final_chunks, expected_sources, k),
        "hit_at_k": hit_at_k(final_chunks, expected_sources, k),
        "reciprocal_rank": rr,
        "abstained": abstained,
        "missing_critical": list(expected_set - set(returned_sources)),
        "irrelevant_chunks": list(set(returned_sources) - expected_set),
        "returned_sources": returned_sources,
        "num_retrieved_chunks": len(final_chunks),
        "num_ranked_chunks": len(ranked_chunks),
        "latency_ms": latency_ms,
    }


def aggregate_metrics(per_question_metrics: Sequence[dict]) -> dict:
    """Aggregate per-question metrics into run-level summary."""
    total = len(per_question_metrics)
    if total == 0:
        return {}

    def avg(key: str) -> float:
        return sum(m[key] for m in per_question_metrics) / total

    return {
        "total_questions": total,
        "hit_rate": avg("retrieval_hit"),
        "avg_precision": avg("retrieval_precision"),
        "avg_recall": avg("retrieval_recall"),
        "precision_at_k": avg("precision_at_k"),
        "recall_at_k": avg("recall_at_k"),
        "hit_at_k": avg("hit_at_k"),
        "mrr": mean_reciprocal_rank([m["reciprocal_rank"] for m in per_question_metrics]),
        "abstention_rate": sum(1 for m in per_question_metrics if m["abstained"]) / total,
        "avg_latency_ms": avg("latency_ms"),
    }
