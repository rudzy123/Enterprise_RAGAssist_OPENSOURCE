"""Tests for retrieval evaluation metrics."""

from evals.metrics import (
    aggregate_metrics,
    compute_retrieval_metrics,
    mean_reciprocal_rank,
    precision_at_k,
    reciprocal_rank,
)


def _chunk(source: str, rank_hint: str = "") -> dict:
    return {"source_file": source, "section_title": rank_hint, "text": "body"}


def test_precision_at_k():
    chunks = [_chunk("a.md"), _chunk("b.md"), _chunk("a.md")]
    assert precision_at_k(chunks, ["a.md"], k=3) == 2 / 3
    assert precision_at_k(chunks, ["a.md"], k=1) == 1.0


def test_reciprocal_rank_first_position():
    chunks = [_chunk("wrong.md"), _chunk("target.md")]
    assert reciprocal_rank(chunks, ["target.md"]) == 0.5


def test_reciprocal_rank_missing():
    chunks = [_chunk("wrong.md")]
    assert reciprocal_rank(chunks, ["target.md"]) == 0.0


def test_mrr_aggregate():
    assert mean_reciprocal_rank([1.0, 0.5, 0.0]) == 0.5


def test_abstention_in_metrics():
    metrics = compute_retrieval_metrics(
        {"source_doc_id": "a.md"},
        final_chunks=[],
        ranked_chunks=[],
        k=3,
        abstained=True,
    )
    assert metrics["abstained"] is True
    assert metrics["precision_at_k"] == 0.0
    assert metrics["reciprocal_rank"] == 0.0


def test_aggregate_summary():
    per_q = [
        {
            "retrieval_hit": True,
            "retrieval_precision": 1.0,
            "retrieval_recall": 1.0,
            "precision_at_k": 0.5,
            "recall_at_k": 1.0,
            "hit_at_k": True,
            "reciprocal_rank": 1.0,
            "abstained": False,
            "latency_ms": 100.0,
        },
        {
            "retrieval_hit": False,
            "retrieval_precision": 0.0,
            "retrieval_recall": 0.0,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "hit_at_k": False,
            "reciprocal_rank": 0.0,
            "abstained": True,
            "latency_ms": 50.0,
        },
    ]
    summary = aggregate_metrics(per_q)
    assert summary["mrr"] == 0.5
    assert summary["abstention_rate"] == 0.5
    assert summary["precision_at_k"] == 0.25
