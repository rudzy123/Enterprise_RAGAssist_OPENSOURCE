"""Tests for cosine similarity scoring."""

from retrieval.similarity import (
    chunk_similarity_score,
    cosine_similarity_from_distance,
    max_similarity,
    similarities_from_distances,
)


def test_cosine_similarity_from_distance():
    assert cosine_similarity_from_distance(0.0) == 1.0
    assert cosine_similarity_from_distance(1.0) == 0.0
    assert cosine_similarity_from_distance(0.35) == 0.65


def test_similarities_from_distances():
    assert similarities_from_distances([0.0, 0.5]) == [1.0, 0.5]


def test_max_similarity():
    assert max_similarity([0.4, 0.9, 0.2]) == 0.9
    assert max_similarity([]) is None


def test_chunk_similarity_score_from_distance():
    chunk = {"distance": 0.2, "source_file": "a.md"}
    assert chunk_similarity_score(chunk) == 0.8


def test_chunk_similarity_score_prefers_existing_score():
    chunk = {"distance": 0.2, "similarity_score": 0.75}
    assert chunk_similarity_score(chunk) == 0.75
