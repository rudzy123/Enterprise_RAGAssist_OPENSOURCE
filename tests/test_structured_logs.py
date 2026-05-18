"""Tests for retrieval structured logging."""

import json
import logging
from unittest.mock import patch

from retrieval.structured_logs import (
    chunk_log_entry,
    log_retrieval_pipeline,
)


def test_chunk_log_entry_truncates_text():
    chunk = {
        "chunk_id": "doc_1",
        "source_file": "a.md",
        "section_title": "Purpose",
        "similarity_score": 0.8123,
        "rerank_score": 4.5678,
        "distance": 0.1877,
        "text": "x" * 200,
    }
    entry = chunk_log_entry(chunk, rank=1)
    assert entry["rank"] == 1
    assert entry["similarity_score"] == 0.8123
    assert entry["rerank_score"] == 4.5678
    assert len(entry["text_preview"]) == 123  # 120 + "..."


def test_log_retrieval_pipeline_emits_four_events():
    logger = logging.getLogger("test.retrieval.pipeline")
    logger.handlers.clear()
    records = []

    class ListHandler(logging.Handler):
        def emit(self, record):
            records.append(json.loads(record.getMessage()))

    logger.addHandler(ListHandler())
    logger.setLevel(logging.INFO)

    raw = [
        {
            "chunk_id": "c1",
            "source_file": "a.md",
            "section_title": "S1",
            "similarity_score": 0.9,
            "distance": 0.1,
            "text": "alpha",
        },
        {
            "chunk_id": "c2",
            "source_file": "b.md",
            "section_title": "S2",
            "similarity_score": 0.3,
            "distance": 0.7,
            "text": "beta",
        },
    ]
    passed = [raw[0]]
    reranked = [{**raw[0], "rerank_score": 1.5}]
    final = reranked

    with patch("retrieval.structured_logs.log_event") as mock_log:
        log_retrieval_pipeline(
            logger,
            "test query",
            trace_id="trace-123",
            retrieve_k=15,
            final_k=3,
            min_similarity=0.4,
            max_chunks_per_file=2,
            rerank_enabled=True,
            rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            raw_candidates=raw,
            threshold_passed=passed,
            reranked=reranked,
            final_chunks=final,
        )

    events = [call.args[1] for call in mock_log.call_args_list]
    assert events == [
        "retrieval_candidates",
        "retrieval_similarity_scores",
        "retrieval_rerank_scores",
        "retrieval_final_selection",
    ]
    assert mock_log.call_args_list[0].kwargs["trace_id"] == "trace-123"
    assert mock_log.call_args_list[1].kwargs["passed_count"] == 1
    assert mock_log.call_args_list[2].kwargs["candidates"][0]["rerank_score"] == 1.5
