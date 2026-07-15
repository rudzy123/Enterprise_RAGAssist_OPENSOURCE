"""
Singleton embedding model for bi-encoder retrieval and ingestion.

Optional Redis caching accelerates repeated query embeddings when CACHE_ENABLED=true.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from core.cache import cache_get_json, cache_set_json
from core.config import CACHE_ENABLED, EMBEDDING_MODEL_NAME

_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Return the process-wide SentenceTransformer singleton."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def embed_text(text: str) -> List[float]:
    """Encode a single text string into an embedding vector."""
    model = get_embedding_model()
    vector = model.encode(text)
    if isinstance(vector, np.ndarray):
        return vector.tolist()
    return list(vector)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Encode multiple texts in one batch."""
    model = get_embedding_model()
    vectors = model.encode(texts)
    if isinstance(vectors, np.ndarray):
        return vectors.tolist()
    return [list(vector) for vector in vectors]


def encode_query(query: str) -> Union[np.ndarray, List[float]]:
    """Encode a query for vector search, with optional Redis cache."""
    if CACHE_ENABLED:
        cache_key = f"{EMBEDDING_MODEL_NAME}:{query}"
        cached = cache_get_json("query_embedding", cache_key)
        if cached is not None:
            return np.array(cached, dtype=np.float32)

    vector = get_embedding_model().encode(query)

    if CACHE_ENABLED:
        as_list = vector.tolist() if isinstance(vector, np.ndarray) else list(vector)
        cache_set_json("query_embedding", cache_key, as_list)

    return vector
