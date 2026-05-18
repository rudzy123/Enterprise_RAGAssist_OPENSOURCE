"""Tests for cross-encoder reranking."""

from unittest.mock import MagicMock, patch

from retrieval.rerank import rerank_chunks


def test_rerank_chunks_orders_by_score():
    chunks = [
        {"text": "low relevance", "source_file": "a.md", "similarity_score": 0.9},
        {"text": "high relevance", "source_file": "b.md", "similarity_score": 0.5},
    ]

    mock_model = MagicMock()
    mock_model.predict.return_value = [0.1, 0.9]

    with patch("retrieval.rerank._get_cross_encoder", return_value=mock_model):
        result = rerank_chunks("test query", chunks)

    assert result[0]["text"] == "high relevance"
    assert result[0]["rerank_score"] == 0.9
    assert result[1]["rerank_score"] == 0.1
    mock_model.predict.assert_called_once_with(
        [["test query", "low relevance"], ["test query", "high relevance"]]
    )


def test_rerank_chunks_empty():
    assert rerank_chunks("query", []) == []
