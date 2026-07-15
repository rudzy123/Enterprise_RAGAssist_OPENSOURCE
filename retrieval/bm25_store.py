"""
BM25 sparse retrieval index backed by rank_bm25.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi

from core.config import BM25_INDEX_DIR, CHROMA_COLLECTION_NAME
from retrieval.metadata_filter import filter_chunks

_INDEX_CACHE: Dict[str, "BM25Index"] = {}


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """In-memory BM25 index for a single collection."""

    def __init__(
        self,
        *,
        collection_name: str,
        chunk_ids: List[str],
        documents: List[str],
        metadatas: List[dict],
        bm25: BM25Okapi,
    ):
        self.collection_name = collection_name
        self.chunk_ids = chunk_ids
        self.documents = documents
        self.metadatas = metadatas
        self.bm25 = bm25

    def search(
        self,
        query: str,
        top_k: int,
        metadata_filters: Optional[dict] = None,
    ) -> List[dict]:
        if not self.chunk_ids:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        max_score = max(scores) if len(scores) else 0.0

        candidates = []
        for idx, score in enumerate(scores):
            if score <= 0:
                continue

            metadata = self.metadatas[idx]
            normalized = (float(score) / max_score) if max_score > 0 else 0.0
            candidates.append(
                {
                    "chunk_id": self.chunk_ids[idx],
                    "text": self.documents[idx],
                    "source_file": metadata.get("source_file", "Unknown"),
                    "section_title": metadata.get("section_title", "Unknown"),
                    "doc_type": metadata.get("doc_type", "curated_md"),
                    "bm25_score": float(score),
                    "bm25_score_normalized": normalized,
                    "similarity_score": 0.0,
                    "distance": 1.0,
                }
            )

        candidates.sort(key=lambda c: c["bm25_score"], reverse=True)
        candidates = filter_chunks(candidates, metadata_filters)
        return candidates[:top_k]


def _index_path(collection_name: str) -> Path:
    BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return BM25_INDEX_DIR / f"{collection_name}.json"


def build_bm25_index(
    *,
    chunk_ids: List[str],
    documents: List[str],
    metadatas: List[dict],
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> BM25Index:
    """Build and persist a BM25 index for a collection."""
    tokenized_corpus = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)

    index = BM25Index(
        collection_name=collection_name,
        chunk_ids=chunk_ids,
        documents=documents,
        metadatas=metadatas,
        bm25=bm25,
    )

    payload = {
        "collection_name": collection_name,
        "chunk_ids": chunk_ids,
        "documents": documents,
        "metadatas": metadatas,
        "tokenized_corpus": tokenized_corpus,
    }
    _index_path(collection_name).write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    _INDEX_CACHE[collection_name] = index
    return index


def load_bm25_index(collection_name: str = CHROMA_COLLECTION_NAME) -> Optional[BM25Index]:
    """Load a persisted BM25 index into memory."""
    if collection_name in _INDEX_CACHE:
        return _INDEX_CACHE[collection_name]

    path = _index_path(collection_name)
    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    bm25 = BM25Okapi(payload["tokenized_corpus"])
    index = BM25Index(
        collection_name=payload["collection_name"],
        chunk_ids=payload["chunk_ids"],
        documents=payload["documents"],
        metadatas=payload["metadatas"],
        bm25=bm25,
    )
    _INDEX_CACHE[collection_name] = index
    return index


def get_bm25_index(collection_name: str = CHROMA_COLLECTION_NAME) -> Optional[BM25Index]:
    """Return a loaded BM25 index, or None if unavailable."""
    return load_bm25_index(collection_name)


def bm25_search(
    query: str,
    *,
    top_k: int,
    collection_name: str = CHROMA_COLLECTION_NAME,
    metadata_filters: Optional[dict] = None,
) -> List[dict]:
    """Run BM25 search against the persisted index."""
    index = get_bm25_index(collection_name)
    if index is None:
        return []
    return index.search(query, top_k=top_k, metadata_filters=metadata_filters)
