"""Tests for retrieval and answer evaluation metrics."""

from config import NOT_FOUND_ANSWER
from evals.metrics import (
    aggregate_metrics,
    answer_has_citations,
    compute_answer_metrics,
    compute_groundedness_heuristic,
    compute_retrieval_metrics,
    mean_reciprocal_rank,
    precision_at_k,
    reciprocal_rank,
    token_overlap_with_chunks,
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
            "has_citations": True,
            "grounded": True,
            "groundedness_score": 0.8,
            "confidence": 0.7,
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
            "has_citations": False,
            "grounded": True,
            "groundedness_score": 1.0,
            "confidence": 0.0,
        },
    ]
    summary = aggregate_metrics(per_q)
    assert summary["mrr"] == 0.5
    assert summary["abstention_rate"] == 0.5
    assert summary["precision_at_k"] == 0.25
    assert summary["pct_with_citations"] == 0.5
    assert summary["pct_grounded"] == 1.0


def test_token_overlap_with_chunks():
    answer = "[a.md - S] Access must follow least privilege requirements."
    chunks = [{"source_file": "a.md", "section_title": "S", "text": "Access must follow least privilege."}]
    overlap = token_overlap_with_chunks(answer, chunks)
    assert overlap >= 0.5


def test_groundedness_abstention_with_no_chunks():
    grounded, score, reason = compute_groundedness_heuristic(
        NOT_FOUND_ANSWER,
        [],
        expected_answer="Some expected answer",
        abstained=True,
    )
    assert grounded is True
    assert score == 1.0


def test_groundedness_abstention_with_answerable_context():
    chunks = [
        {
            "source_file": "a.md",
            "section_title": "Purpose",
            "text": "Access must follow the principle of least privilege.",
        }
    ]
    grounded, score, _ = compute_groundedness_heuristic(
        NOT_FOUND_ANSWER,
        chunks,
        expected_answer="Access must follow the principle of least privilege",
        abstained=False,
    )
    assert grounded is False
    assert score == 0.0


def test_answer_has_citations():
    chunks = [{"source_file": "a.md", "section_title": "S", "text": "body"}]
    assert answer_has_citations("Fact [a.md - S].", chunks)
    assert not answer_has_citations("Fact without cite.", chunks)


def test_compute_answer_metrics():
    question = {
        "expected_answer": "Access must follow the principle of least privilege",
    }
    chunks = [
        {
            "source_file": "a.md",
            "section_title": "Policy",
            "text": "Access must follow the principle of least privilege.",
        }
    ]
    answer = "[a.md - Policy] Access must follow the principle of least privilege."
    metrics = compute_answer_metrics(
        question,
        answer=answer,
        chunks=chunks,
        abstained=False,
        confidence=0.75,
        confidence_reason="high similarity",
    )
    assert metrics["has_citations"] is True
    assert metrics["grounded"] is True
    assert metrics["answer"] == answer
