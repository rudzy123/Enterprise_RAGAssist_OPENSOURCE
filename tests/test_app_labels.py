"""Unit tests for Streamlit powered-by labels."""

from app.app import format_powered_by_label


def test_powered_by_llama():
    assert (
        format_powered_by_label(
            {"resolved_provider": "ollama", "ollama_model": "llama3.2"}
        )
        == "🦙 Powered by Llama 3.2"
    )


def test_powered_by_openai():
    assert (
        format_powered_by_label(
            {"resolved_provider": "openai", "openai_model": "gpt-3.5-turbo"}
        )
        == "☁️ Powered by gpt-3.5-turbo"
    )


def test_powered_by_retrieval_only():
    assert (
        format_powered_by_label({"resolved_provider": "retrieval_only"})
        == "📄 Retrieval-only Mode"
    )
