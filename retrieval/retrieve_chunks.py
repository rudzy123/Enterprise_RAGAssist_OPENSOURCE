"""
Semantic retrieval from Chroma collection (pure vector search).

Pipeline: vector retrieve (retrieve_k) -> min-similarity filter ->
cross-encoder rerank -> per-document cap -> final_k.
No fallback below min similarity.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    FINAL_K,
    MAX_CHUNKS_PER_FILE,
    MIN_CHUNK_SIMILARITY,
    RERANK_ENABLED,
    RERANK_MODEL,
    RERANK_TOP_N,
    RETRIEVE_K,
    RETRIEVAL_STRUCTURED_LOGS,
)
from observability import setup_json_logger
from retrieval.rerank import rerank_chunks
from retrieval.similarity import cosine_similarity_from_distance
from retrieval.result_format import enrich_retrieved_chunks
from retrieval.structured_logs import log_retrieval_pipeline

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "enterprise_docs"
CHROMA_PATH = "./chroma_db"

_embedding_model: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.PersistentClient] = None
_retrieval_logger: Optional[logging.Logger] = None


def _get_retrieval_logger() -> logging.Logger:
    global _retrieval_logger
    if _retrieval_logger is None:
        _retrieval_logger = setup_json_logger("enterprise_rag.retrieval")
    return _retrieval_logger


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def _get_chroma_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _chroma_client


def _apply_per_document_cap(chunks: List[dict], max_per_file: int) -> List[dict]:
    """Keep top chunks per source_file (chunks must already be ranked)."""
    counts: dict[str, int] = {}
    capped: List[dict] = []
    for chunk in chunks:
        source = chunk["source_file"]
        if counts.get(source, 0) >= max_per_file:
            continue
        counts[source] = counts.get(source, 0) + 1
        capped.append(chunk)
    return capped


def retrieve_similar_chunks(
    query: str,
    retrieve_k: int = RETRIEVE_K,
    final_k: int = FINAL_K,
    top_k: Optional[int] = None,
    collection_name: str = COLLECTION_NAME,
    min_similarity: float = MIN_CHUNK_SIMILARITY,
    max_chunks_per_file: int = MAX_CHUNKS_PER_FILE,
    rerank_enabled: bool = RERANK_ENABLED,
    rerank_top_n: int = RERANK_TOP_N,
    verbose: bool = False,
    return_trace: bool = False,
    trace_id: Optional[str] = None,
    structured_logs: bool = RETRIEVAL_STRUCTURED_LOGS,
) -> Union[List[dict], dict]:
    """
    Perform semantic search against the Chroma collection.

    Args:
        query: Plain text search query
        retrieve_k: Number of vector candidates to fetch from Chroma
        final_k: Maximum chunks returned after filtering and caps
        top_k: Deprecated alias for final_k
        collection_name: Chroma collection name
        min_similarity: Minimum cosine similarity (1 - distance); no fallback below this
        max_chunks_per_file: Max chunks retained per source_file
        rerank_enabled: Apply cross-encoder reranking after similarity filter
        rerank_top_n: Max candidates passed to the reranker
        verbose: Enable debug prints
        return_trace: If True, return dict with chunks and intermediate pipeline stages
        trace_id: Optional trace ID attached to structured log events
        structured_logs: Emit JSON logs for each pipeline stage

    Returns:
        List of chunk dicts, or trace dict when return_trace=True
    """
    if top_k is not None:
        final_k = top_k

    try:
        client = _get_chroma_client()
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        if verbose:
            print(f"Error connecting to Chroma: {e}")
        empty = [] if not return_trace else _empty_trace(
            retrieve_k, final_k, min_similarity, max_chunks_per_file, rerank_enabled
        )
        return empty

    model = _get_embedding_model()
    query_embedding = model.encode(query)

    n_results = min(retrieve_k, max(collection.count(), 1))

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    raw_candidates: List[dict] = []
    for chunk_id, distance, doc, metadata in zip(
        results["ids"][0],
        results["distances"][0],
        results["documents"][0],
        results["metadatas"][0],
    ):
        similarity = cosine_similarity_from_distance(distance)
        raw_candidates.append(
            {
                "text": doc,
                "source_file": metadata.get("source_file", "Unknown"),
                "section_title": metadata.get("section_title", "Unknown"),
                "similarity_score": similarity,
                "distance": distance,
                "chunk_id": chunk_id,
            }
        )

    raw_candidates.sort(key=lambda c: c["similarity_score"], reverse=True)

    threshold_passed = [
        c for c in raw_candidates if c["similarity_score"] >= min_similarity
    ]

    rerank_input = threshold_passed[:rerank_top_n]
    if rerank_enabled and rerank_input:
        reranked = rerank_chunks(query, rerank_input)
    else:
        reranked = list(rerank_input)

    capped = _apply_per_document_cap(reranked, max_chunks_per_file)
    final_chunks = enrich_retrieved_chunks(capped[:final_k])

    if structured_logs:
        log_retrieval_pipeline(
            _get_retrieval_logger(),
            query,
            trace_id=trace_id,
            retrieve_k=retrieve_k,
            final_k=final_k,
            min_similarity=min_similarity,
            max_chunks_per_file=max_chunks_per_file,
            rerank_enabled=rerank_enabled,
            rerank_model=RERANK_MODEL if rerank_enabled else None,
            raw_candidates=raw_candidates,
            threshold_passed=threshold_passed,
            reranked=reranked,
            final_chunks=final_chunks,
        )

    if verbose:
        print(f"\nRaw candidates: {len(raw_candidates)}")
        print(f"Above threshold ({min_similarity}): {len(threshold_passed)}")
        print(f"Rerank enabled: {rerank_enabled} ({len(rerank_input)} scored)")
        print(f"After per-file cap ({max_chunks_per_file}): {len(capped)}")
        print(f"Final (k={final_k}): {len(final_chunks)}")
        for i, c in enumerate(final_chunks, 1):
            score_label = (
                f"rerank={c['rerank_score']:.3f}"
                if "rerank_score" in c
                else f"sim={c['similarity_score']:.3f}"
            )
            print(f"  {i}. {c['source_file']} ({score_label})")

    if return_trace:
        return {
            "chunks": final_chunks,
            "top_k": final_k,
            "raw_candidates": raw_candidates,
            "threshold_passed": threshold_passed,
            "reranked": reranked,
            "capped": enrich_retrieved_chunks(capped),
            "retrieve_k": retrieve_k,
            "final_k": final_k,
            "min_similarity": min_similarity,
            "max_chunks_per_file": max_chunks_per_file,
            "rerank_enabled": rerank_enabled,
            "rerank_top_n": rerank_top_n,
        }

    return final_chunks


def _empty_trace(
    retrieve_k: int,
    final_k: int,
    min_similarity: float,
    max_chunks_per_file: int,
    rerank_enabled: bool,
) -> dict:
    return {
        "chunks": [],
        "top_k": final_k,
        "raw_candidates": [],
        "threshold_passed": [],
        "reranked": [],
        "capped": [],
        "retrieve_k": retrieve_k,
        "final_k": final_k,
        "min_similarity": min_similarity,
        "max_chunks_per_file": max_chunks_per_file,
        "rerank_enabled": rerank_enabled,
        "rerank_top_n": RERANK_TOP_N,
    }


if __name__ == "__main__":
    test_query = "What is incident response?"
    chunks = retrieve_similar_chunks(test_query, verbose=True)
    print(f"\nReturned {len(chunks)} chunks")
