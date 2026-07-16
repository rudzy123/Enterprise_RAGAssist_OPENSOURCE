"""Tests for LLM provider resolution."""

from unittest.mock import patch

from core.config import llm_generation_enabled, resolve_llm_provider


def test_ollama_when_provider_ollama():
    with (
        patch("core.config.LLM_PROVIDER", "ollama"),
        patch("core.config.OPENAI_API_KEY", "sk-test"),
    ):
        assert resolve_llm_provider() == "ollama"
        assert llm_generation_enabled() is True


def test_openai_when_provider_not_ollama_and_key_present():
    with (
        patch("core.config.LLM_PROVIDER", "openai"),
        patch("core.config.OPENAI_API_KEY", "sk-test"),
    ):
        assert resolve_llm_provider() == "openai"


def test_openai_when_empty_provider_and_key_present():
    with (
        patch("core.config.LLM_PROVIDER", ""),
        patch("core.config.OPENAI_API_KEY", "sk-test"),
    ):
        assert resolve_llm_provider() == "openai"


def test_retrieval_only_without_key_when_not_ollama():
    with (
        patch("core.config.LLM_PROVIDER", ""),
        patch("core.config.OPENAI_API_KEY", ""),
    ):
        assert resolve_llm_provider() == "retrieval_only"
        assert llm_generation_enabled() is False


def test_explicit_retrieval_only():
    with (
        patch("core.config.LLM_PROVIDER", "retrieval_only"),
        patch("core.config.OPENAI_API_KEY", "sk-test"),
    ):
        assert resolve_llm_provider() == "retrieval_only"


def test_use_llm_false_forces_retrieval_only():
    with patch("core.config.LLM_PROVIDER", "ollama"):
        assert resolve_llm_provider(use_llm=False) == "retrieval_only"
