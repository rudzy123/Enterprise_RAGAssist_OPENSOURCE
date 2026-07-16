"""Tests for public /status LLM provider reporting."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app


def test_status_endpoint_reports_llm_provider():
    with (
        patch("main.LLM_PROVIDER", "ollama"),
        patch("main.OLLAMA_MODEL", "llama3.2"),
        patch("main.OLLAMA_HOST", "http://localhost:11434"),
        patch("main.resolve_llm_provider", return_value="ollama"),
        patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False),
    ):
        client = TestClient(app)
        response = client.get("/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["llm_provider"] == "ollama"
    assert payload["resolved_provider"] == "ollama"
    assert payload["ollama_model"] == "llama3.2"
    assert payload["openai_configured"] is False
    assert "openai_model" in payload


def test_status_endpoint_is_public():
    with patch("core.auth.API_KEY", "secret-key"), patch("main.API_KEY", "secret-key"):
        client = TestClient(app)
        response = client.get("/status")
    assert response.status_code == 200
