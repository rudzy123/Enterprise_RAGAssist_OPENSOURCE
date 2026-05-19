"""Tests for answer generation and citation rules."""

from unittest.mock import MagicMock, patch

from answer_generation.confidence import is_low_confidence
from answer_generation.generation import (
    answer_has_chunk_citations,
    build_generation_prompt,
    chunk_citation,
    format_chunks_for_prompt,
    generate_answer_from_chunks,
    generate_retrieval_only_answer,
    generate_with_openai,
)
from config import NOT_FOUND_ANSWER


def test_chunk_citation_format():
    chunk = {"source_file": "policy.md", "section_title": "Purpose"}
    assert chunk_citation(chunk) == "[policy.md - Purpose]"


def test_retrieval_only_answer_includes_citations():
    chunks = [
        {
            "source_file": "a.md",
            "section_title": "Intro",
            "text": "Hello world",
            "similarity_score": 0.8,
        }
    ]
    answer = generate_retrieval_only_answer("q", chunks)
    assert "[a.md - Intro]" in answer
    assert "Hello world" in answer


def test_low_confidence_returns_not_found():
    chunks = [
        {
            "source_file": "a.md",
            "section_title": "Intro",
            "text": "Hello",
            "similarity_score": 0.1,
            "distance": 0.9,
        }
    ]
    answer, confidence, _ = generate_answer_from_chunks(
        "q",
        chunks,
        confidence=0.1,
        use_llm=False,
    )
    assert answer == NOT_FOUND_ANSWER
    assert is_low_confidence(confidence)


def test_build_generation_prompt_requires_citations():
    chunks = [{"source_file": "a.md", "section_title": "S", "text": "body"}]
    context, labels = format_chunks_for_prompt(chunks)
    prompt = build_generation_prompt("What?", context, labels)
    assert "[a.md - S]" in prompt
    assert NOT_FOUND_ANSWER in prompt
    assert "inline citation" in prompt.lower()


def test_answer_has_chunk_citations():
    chunks = [{"source_file": "a.md", "section_title": "S", "text": "x"}]
    assert answer_has_chunk_citations("Fact [a.md - S].", chunks)
    assert not answer_has_chunk_citations("Fact without cite.", chunks)


def test_generate_with_openai_fallback_when_no_citations():
    chunks = [
        {
            "source_file": "a.md",
            "section_title": "S",
            "text": "Important text",
            "similarity_score": 0.9,
        }
    ]
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Uncited answer."))]
    mock_response.usage = None

    with patch("answer_generation.generation.openai.OpenAI") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = mock_response
        answer, _ = generate_with_openai("q", chunks, api_key="test-key")

    assert "[a.md - S]" in answer
    assert "Important text" in answer
