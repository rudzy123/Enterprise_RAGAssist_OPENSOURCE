"""
BM25 sparse retrieval index backed by rank_bm25.

Indexes are persisted as JSON under BM25_INDEX_DIR and loaded on demand.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from core.config import BM25_INDEX_DIR, CHROMA_COLLECTION_NAME
from observability import setup_json_logger
from retrieval.metadata_filter import chunk_matches_filters, normalize_metadata_filters

logger = setup_json_logger("enterprise_rag.bm25")

_INDEX_CACHE: Dict[str, "BM25Index"] = {}


def tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer suitable for policy / runbook prose."""
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

    @property
    def size(self) -> int:
        return len(self.chunk_ids)

    def _eligible_indices(
        self,
        metadata_filters: Optional[dict],
    ) -> List[int]:
        """Return corpus indices that satisfy metadata filters before scoring."""
        filters = normalize_metadata_filters(metadata_filters)
        if not filters:
            return list(range(len(self.chunk_ids)))

        eligible = []
        for idx, metadata in enumerate(self.metadatas):
            chunk = {
                "source_file": metadata.get("source_file", "Unknown"),
                "section_title": metadata.get("section_title", "Unknown"),
                "doc_type": metadata.get("doc_type", "curated_md"),
            }
            if chunk_matches_filters(chunk, filters):
                eligible.append(idx)
        return eligible

    def search(
        self,
        query: str,
        top_k: int,
        metadata_filters: Optional[dict] = None,
    ) -> Tuple[List[dict], dict]:
        """
        Score and rank BM25 candidates.

        Returns:
            (candidates, search_stats)
        """
        stats = {
            "index_size": self.size,
            "top_k_requested": top_k,
            "metadata_filters": normalize_metadata_filters(metadata_filters),
            "warnings": [],
        }

        if not self.chunk_ids:
            stats["warnings"].append("bm25_index_empty")
            return [], stats

        query_tokens = tokenize(query)
        if not query_tokens:
            stats["warnings"].append("query_tokenization_empty")
            return [], stats

        eligible_indices = self._eligible_indices(metadata_filters)
        stats["eligible_count"] = len(eligible_indices)
        if not eligible_indices:
            stats["warnings"].append("no_chunks_match_metadata_filters")
            return [], stats

        scores = self.bm25.get_scores(query_tokens)
        max_score = max(float(scores[i]) for i in eligible_indices)

        candidates = []
        for idx in eligible_indices:
            score = float(scores[idx])
            if score <= 0:
                continue

            metadata = self.metadatas[idx]
            normalized = (score / max_score) if max_score > 0 else 0.0
            candidates.append(
                {
                    "chunk_id": self.chunk_ids[idx],
                    "text": self.documents[idx],
                    "source_file": metadata.get("source_file", "Unknown"),
                    "section_title": metadata.get("section_title", "Unknown"),
                    "doc_type": metadata.get("doc_type", "curated_md"),
                    "bm25_score": score,
                    "bm25_score_normalized": normalized,
                    "similarity_score": 0.0,
                    "distance": 1.0,
                }
            )

        candidates.sort(key=lambda c: c["bm25_score"], reverse=True)
        results = candidates[:top_k]
        stats["returned_count"] = len(results)
        return results, stats


def _index_path(collection_name: str) -> Path:
    BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return BM25_INDEX_DIR / f"{collection_name}.json"


def bm25_index_exists(collection_name: str = CHROMA_COLLECTION_NAME) -> bool:
    """Return True when a persisted BM25 index file exists."""
    return _index_path(collection_name).exists()


def build_bm25_index(
    *,
    chunk_ids: List[str],
    documents: List[str],
    metadatas: List[dict],
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> BM25Index:
    """Build and persist a BM25 index for a collection."""
    if not chunk_ids:
        raise ValueError("Cannot build BM25 index from zero chunks.")

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
    path = _index_path(collection_name)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    _INDEX_CACHE[collection_name] = index
    logger.info(
        "BM25 index built.",
        extra={
            "extra": {
                "collection_name": collection_name,
                "chunk_count": len(chunk_ids),
                "index_path": str(path),
            }
        },
    )
    return index


def load_bm25_index(collection_name: str = CHROMA_COLLECTION_NAME) -> Optional[BM25Index]:
    """Load a persisted BM25 index into memory."""
    if collection_name in _INDEX_CACHE:
        return _INDEX_CACHE[collection_name]

    path = _index_path(collection_name)
    if not path.exists():
        logger.warning(
            "BM25 index file not found.",
            extra={"extra": {"collection_name": collection_name, "index_path": str(path)}},
        )
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        bm25 = BM25Okapi(payload["tokenized_corpus"])
        index = BM25Index(
            collection_name=payload["collection_name"],
            chunk_ids=payload["chunk_ids"],
            documents=payload["documents"],
            metadatas=payload["metadatas"],
            bm25=bm25,
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.error(
            "Failed to load BM25 index.",
            extra={"extra": {"collection_name": collection_name, "error": str(exc)}},
        )
        return None

    _INDEX_CACHE[collection_name] = index
    logger.info(
        "BM25 index loaded.",
        extra={
            "extra": {
                "collection_name": collection_name,
                "chunk_count": index.size,
            }
        },
    )
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
) -> Tuple[List[dict], dict]:
    """
    Run BM25 search against the persisted index.

    Returns:
        (candidates, search_stats) — empty candidates when index is missing.
    """
    index = get_bm25_index(collection_name)
    if index is None:
        return [], {
            "index_available": False,
            "warnings": ["bm25_index_missing"],
            "top_k_requested": top_k,
            "metadata_filters": normalize_metadata_filters(metadata_filters),
        }

    candidates, stats = index.search(query, top_k=top_k, metadata_filters=metadata_filters)
    stats["index_available"] = True
    return candidates, stats
