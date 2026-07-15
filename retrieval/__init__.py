from retrieval.result_format import enrich_retrieved_chunks, format_document_source
from retrieval.similarity import (
    chunk_similarity_score,
    cosine_similarity_from_distance,
    max_similarity,
    similarities_from_distances,
)
from retrieval.structured_logs import (
    log_final_selection,
    log_rerank_scores,
    log_retrieval_pipeline,
    log_retrieval_results,
    log_retrieved_candidates,
    log_similarity_scores,
)


def __getattr__(name: str):
    if name == "rerank_chunks":
        from retrieval.rerank import rerank_chunks

        return rerank_chunks
    if name == "retrieve_similar_chunks":
        from retrieval.retrieve_chunks import retrieve_similar_chunks

        return retrieve_similar_chunks
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "chunk_similarity_score",
    "cosine_similarity_from_distance",
    "enrich_retrieved_chunks",
    "format_document_source",
    "log_final_selection",
    "log_rerank_scores",
    "log_retrieval_pipeline",
    "log_retrieval_results",
    "log_retrieved_candidates",
    "log_similarity_scores",
    "max_similarity",
    "rerank_chunks",
    "retrieve_similar_chunks",
    "similarities_from_distances",
]
