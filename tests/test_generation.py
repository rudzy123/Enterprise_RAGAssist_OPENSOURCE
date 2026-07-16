"""Tests for answer generation and citation rules."""

from unittest.mock import MagicMock, patch

from answer_generation.confidence import assess_retrieval_context, is_low_confidence
from answer_generation.generation import (
    answer_has_chunk_citations,
    build_generation_prompt,
    chunk_citation,
    ensure_cited_answer,
    format_chunks_for_prompt,
    generate_answer_from_chunks,
    generate_retrieval_only_answer,
    generate_with_ollama,
    generate_with_openai,
)
from config import MIN_SIMILARITY_THRESHOLD, NOT_FOUND_ANSWER


def test_chunk_citation_format():
    chunk = {"source_file": "policy.md", "section_title": "Purpose"}
    assert chunk_citation(chunk) == "[policy.md - Purpose]"


def test_chunk_citation_uses_document_source():
    chunk = {
        "source_file": "policy.md",
        "section_title": "Purpose",
        "document_source": "policy.md - Purpose",
    }
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
    answer, confidence, _, _, _ = generate_answer_from_chunks(
        "q",
        chunks,
        use_llm=False,
    )
    assert answer == NOT_FOUND_ANSWER
    assert is_low_confidence(confidence)


def test_weak_top_similarity_returns_not_found():
    chunks = [
        {
            "source_file": "a.md",
            "section_title": "Intro",
            "text": "Hello",
            "similarity_score": MIN_SIMILARITY_THRESHOLD - 0.05,
            "distance": 1.05 - MIN_SIMILARITY_THRESHOLD,
        }
    ]
    answer, confidence, reason, _, _ = generate_answer_from_chunks(
        "q",
        chunks,
        use_llm=False,
    )
    assert answer == NOT_FOUND_ANSWER
    assert confidence == 0.0
    assert "below relevance threshold" in reason


def test_assess_retrieval_context_no_raw_candidates():
    confidence, reason, is_weak = assess_retrieval_context([], raw_candidate_count=0)
    assert is_weak
    assert confidence == 0.0
    assert reason == "No documents matched the query"


def test_build_generation_prompt_requires_citations():
    chunks = [{"source_file": "a.md", "section_title": "S", "text": "body"}]
    context, labels = format_chunks_for_prompt(chunks)
    prompt = build_generation_prompt("What?", context, labels)
    assert "[a.md - S]" in prompt
    assert NOT_FOUND_ANSWER in prompt
    assert "ONLY" in prompt
    assert "inline citation" in prompt.lower()


def test_answer_has_chunk_citations():
    chunks = [{"source_file": "a.md", "section_title": "S", "text": "x"}]
    assert answer_has_chunk_citations("Fact [a.md - S].", chunks)
    assert not answer_has_chunk_citations("Fact without cite.", chunks)


def test_ensure_cited_answer_falls_back_to_snippets():
    chunks = [
        {
            "source_file": "a.md",
            "section_title": "S",
            "text": "Important text",
            "similarity_score": 0.9,
        }
    ]
    answer = ensure_cited_answer("Uncited answer.", chunks)
    assert "[a.md - S]" in answer
    assert "Important text" in answer


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
        answer, _, _ = generate_with_openai("q", chunks, api_key="test-key")

    assert "[a.md - S]" in answer
    assert "Important text" in answer


def test_generate_with_ollama_uses_local_endpoint():
    chunks = [
        {
            "source_file": "a.md",
            "section_title": "S",
            "text": "Important text",
            "similarity_score": 0.9,
        }
    ]
    mock_response = MagicMock()
    mock_response.message = MagicMock(content="Important text [a.md - S].")
    mock_response.prompt_eval_count = 10
    mock_response.eval_count = 5

    with patch("answer_generation.generation.ollama.Client") as mock_client_cls:
        mock_client_cls.return_value.chat.return_value = mock_response
        answer, tokens, obs = generate_with_ollama(
            "q",
            chunks,
            model="llama3.2",
            host="http://localhost:11434",
        )

    mock_client_cls.assert_called_once_with(host="http://localhost:11434")
    assert mock_client_cls.return_value.chat.call_args.kwargs["options"]["temperature"] == 0.0
    assert "[a.md - S]" in answer
    assert tokens == 15
    assert obs["llm_prompt"]["provider"] == "ollama"
    assert obs["llm_prompt"]["temperature"] == 0.0
    assert obs["generation_mode"] == "llm"


def test_generate_with_ollama_falls_back_when_unavailable():
    chunks = [
        {
            "source_file": "a.md",
            "section_title": "S",
            "text": "Important text",
            "similarity_score": 0.9,
        }
    ]
    with patch("answer_generation.generation.ollama.Client") as mock_client_cls:
        mock_client_cls.return_value.chat.side_effect = ConnectionError("refused")
        answer, tokens, obs = generate_with_ollama("q", chunks, model="llama3.2")

    assert tokens is None
    assert obs["generation_mode"] == "retrieval_only"
    assert "[a.md - S]" in answer
    assert "Important text" in answer
    assert "ollama_error" in obs


def test_generate_answer_routes_to_ollama_when_provider_ollama():
    chunks = [
        {
            "source_file": "a.md",
            "section_title": "Intro",
            "text": "Incident response starts with detection.",
            "similarity_score": 0.85,
        }
    ]
    with (
        patch("answer_generation.generation.LLM_PROVIDER", "ollama"),
        patch(
            "answer_generation.generation.generate_with_ollama",
            return_value=(
                "Incident response starts with detection. [a.md - Intro]",
                None,
                {
                    "llm_prompt": {"provider": "ollama"},
                    "model_response": "ok",
                    "generation_mode": "llm",
                },
            ),
        ) as mock_ollama,
    ):
        answer, _, _, _, obs = generate_answer_from_chunks("q", chunks, use_llm=True)

    mock_ollama.assert_called_once()
    assert obs["llm_provider"] == "ollama"
    assert "[a.md - Intro]" in answer


def test_generate_answer_falls_back_to_retrieval_when_ollama_unavailable():
    chunks = [
        {
            "source_file": "a.md",
            "section_title": "Intro",
            "text": "Incident response starts with detection.",
            "similarity_score": 0.85,
        }
    ]
    with (
        patch("answer_generation.generation.LLM_PROVIDER", "ollama"),
        patch(
            "answer_generation.generation.generate_with_ollama",
            return_value=(
                "[a.md - Intro] Incident response starts with detection.",
                None,
                {
                    "llm_prompt": {"provider": "ollama"},
                    "model_response": "fallback",
                    "generation_mode": "retrieval_only",
                    "ollama_error": "refused",
                },
            ),
        ),
        patch(
            "answer_generation.generation.generate_with_openai",
        ) as mock_openai,
    ):
        answer, _, _, tokens, obs = generate_answer_from_chunks("q", chunks, use_llm=True)

    mock_openai.assert_not_called()
    assert tokens is None
    assert obs["llm_provider"] == "retrieval_only"
    assert obs["generation_mode"] == "retrieval_only"
    assert "[a.md - Intro]" in answer


def test_generate_answer_uses_openai_when_key_and_not_ollama():
    chunks = [
        {
            "source_file": "a.md",
            "section_title": "Intro",
            "text": "Incident response starts with detection.",
            "similarity_score": 0.85,
        }
    ]
    with (
        patch("answer_generation.generation.LLM_PROVIDER", ""),
        patch("answer_generation.generation.OPENAI_API_KEY", "sk-test"),
        patch(
            "answer_generation.generation.generate_with_openai",
            return_value=(
                "Incident response starts with detection. [a.md - Intro]",
                12,
                {
                    "llm_prompt": {"provider": "openai"},
                    "model_response": "ok",
                    "generation_mode": "llm",
                },
            ),
        ) as mock_openai,
    ):
        answer, _, _, tokens, obs = generate_answer_from_chunks("q", chunks, use_llm=True)

    mock_openai.assert_called_once()
    assert tokens == 12
    assert obs["llm_provider"] == "openai"
    assert "[a.md - Intro]" in answer


def test_strong_context_returns_cited_answer():
    chunks = [
        {
            "source_file": "a.md",
            "section_title": "Intro",
            "text": "Incident response starts with detection.",
            "similarity_score": 0.85,
        }
    ]
    answer, confidence, _, _, obs = generate_answer_from_chunks(
        "How does incident response start?",
        chunks,
        use_llm=False,
    )
    assert answer != NOT_FOUND_ANSWER
    assert "[a.md - Intro]" in answer
    assert confidence > 0.3
    assert obs["generation_mode"] == "retrieval_only"
