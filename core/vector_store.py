"""
ChromaDB client and collection management.
"""

from __future__ import annotations

from typing import List, Optional

import chromadb
from chromadb.api.models.Collection import Collection

from core.config import CHROMA_COLLECTION_NAME, CHROMA_DB_PATH

_chroma_client: Optional[chromadb.PersistentClient] = None


def get_chroma_client() -> chromadb.PersistentClient:
    """Return the process-wide Chroma persistent client singleton."""
    global _chroma_client
    if _chroma_client is None:
        CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    return _chroma_client


def get_collection(
    name: str = CHROMA_COLLECTION_NAME,
    *,
    create_if_missing: bool = True,
) -> Collection:
    """Get or create a Chroma collection configured for cosine similarity."""
    client = get_chroma_client()
    if create_if_missing:
        return client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
    return client.get_collection(name=name)


def collection_count(name: str = CHROMA_COLLECTION_NAME) -> int:
    """Return the number of documents in a collection, or 0 if unavailable."""
    try:
        return get_collection(name).count()
    except Exception:
        return 0


def reset_collection(name: str = CHROMA_COLLECTION_NAME) -> Collection:
    """Delete and recreate a collection (idempotent ingestion)."""
    client = get_chroma_client()
    try:
        client.delete_collection(name=name)
    except Exception:
        pass
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(
    *,
    ids: List[str],
    embeddings: List[List[float]],
    documents: List[str],
    metadatas: List[dict],
    collection_name: str = CHROMA_COLLECTION_NAME,
    reset: bool = False,
) -> int:
    """
    Store chunk embeddings in Chroma.

    When reset=True, the collection is recreated before insert.
    Returns the total document count after insert.
    """
    collection = reset_collection(collection_name) if reset else get_collection(collection_name)
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    return collection.count()
