"""Tests for enriched retrieval result formatting."""

from retrieval.result_format import (
    enrich_retrieved_chunk,
    enrich_retrieved_chunks,
    format_document_source,
    make_text_preview,
)
from retrieval.structured_logs import chunk_log_entry, log_retrieval_pipeline


def test_format_document_source():
    assert format_document_source("policy.md", "Purpose") == "policy.md - Purpose"


def test_make_text_preview_truncates():
    preview = make_text_preview("x" * 200)
    assert len(preview) == 123
    assert preview.endswith("...")


def test_enrich_retrieved_chunk_adds_fields():
    chunk = {
        "chunk_id": "c1",
        "source_file": "a.md",
        "section_title": "Intro",
        "similarity_score": 0.85,
        "distance": 0.15,
        "text": "Sample chunk text for preview.",
    }
    enriched = enrich_retrieved_chunk(chunk, rank=2)

    assert enriched["rank"] == 2
    assert enriched["document_source"] == "a.md - Intro"
    assert enriched["text_preview"] == "Sample chunk text for preview."
    assert enriched["text"] == chunk["text"]


def test_enrich_retrieved_chunks_assigns_ranks():
    chunks = [
        {"source_file": "a.md", "section_title": "S1", "text": "one"},
        {"source_file": "b.md", "section_title": "S2", "text": "two"},
    ]
    enriched = enrich_retrieved_chunks(chunks)

    assert [c["rank"] for c in enriched] == [1, 2]
    assert enriched[0]["document_source"] == "a.md - S1"


def test_chunk_log_entry_includes_document_source():
    chunk = enrich_retrieved_chunk(
        {
            "chunk_id": "c1",
            "source_file": "a.md",
            "section_title": "Purpose",
            "similarity_score": 0.9,
            "distance": 0.1,
            "text": "alpha",
        },
        rank=1,
    )
    entry = chunk_log_entry(chunk)

    assert entry["document_source"] == "a.md - Purpose"
    assert entry["rank"] == 1
    assert entry["text_preview"] == "alpha"


def test_log_retrieval_pipeline_emits_results_event():
    import logging
    from unittest.mock import patch

    logger = logging.getLogger("test.retrieval.results")
    logger.handlers.clear()

    raw = [
        {
            "chunk_id": "c1",
            "source_file": "a.md",
            "section_title": "S1",
            "similarity_score": 0.9,
            "distance": 0.1,
            "text": "alpha",
        }
    ]
    final = enrich_retrieved_chunks(raw)

    with patch("retrieval.structured_logs._get_log_event") as mock_get:
        mock_log = mock_get.return_value
        log_retrieval_pipeline(
            logger,
            "test query",
            trace_id="trace-456",
            retrieve_k=15,
            final_k=3,
            min_similarity=0.4,
            max_chunks_per_file=2,
            rerank_enabled=False,
            rerank_model=None,
            raw_candidates=raw,
            threshold_passed=raw,
            reranked=raw,
            final_chunks=final,
        )

    events = [call.args[1] for call in mock_log.call_args_list]
    assert events[-1] == "retrieval_results"
    results_call = mock_log.call_args_list[-1]
    assert results_call.kwargs["top_k"] == 3
    assert results_call.kwargs["results"][0]["document_source"] == "a.md - S1"
