"""Tests for per-request observability logging."""

import json
import tempfile
from pathlib import Path

from observability.request_log import RequestLogger, summarize_chunk


def test_summarize_chunk_includes_preview():
    chunk = {
        "chunk_id": "c1",
        "source_file": "a.md",
        "section_title": "Purpose",
        "similarity_score": 0.9,
        "text": "hello world",
    }
    summary = summarize_chunk(chunk)
    assert summary["document_source"] == "a.md - Purpose"
    assert summary["text_preview"] == "hello world"


def test_request_logger_records_full_request_lifecycle():
    request_log = RequestLogger(trace_id="trace-1", query="initial")
    request_log.log_query("What is incident response?")
    request_log.log_retrieved_chunks(
        [
            {
                "chunk_id": "c1",
                "source_file": "runbook.md",
                "section_title": "Steps",
                "similarity_score": 0.88,
                "text": "Identify and classify the incident.",
            }
        ]
    )
    request_log.log_llm_prompt(
        {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "prompt body"}],
        }
    )
    request_log.log_model_response("Answer text [runbook.md - Steps].", token_usage=42)
    request_log.log_answer("Answer text [runbook.md - Steps].", generation_mode="llm")
    request_log.log_latency(total_ms=120.5, retrieval_ms=40.2, generation_ms=70.1)

    trace = request_log.to_trace_dict()
    assert trace["query"] == "What is incident response?"
    assert len(trace["retrieved_chunks"]) == 1
    assert trace["llm_prompt"]["model"] == "gpt-3.5-turbo"
    assert trace["model_response"].startswith("Answer text")
    assert trace["generation_mode"] == "llm"
    assert trace["latency_ms"] == 120.5
    assert trace["retrieval_latency_ms"] == 40.2
    assert trace["generation_latency_ms"] == 70.1

    events = [step["event"] for step in trace["step_logs"]]
    assert events == [
        "query_received",
        "retrieved_chunks",
        "llm_prompt",
        "model_response",
        "answer_finalized",
        "latency",
    ]


def test_request_logger_writes_json_file():
    request_log = RequestLogger(trace_id="trace-file", query="q")
    request_log.log_query("q")

    with tempfile.TemporaryDirectory() as tmp:
        path = request_log.save_json_file(Path(tmp))
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["trace_id"] == "trace-file"
        assert payload["query"] == "q"
