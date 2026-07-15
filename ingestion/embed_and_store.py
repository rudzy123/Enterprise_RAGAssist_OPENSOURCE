"""
Embed chunks and store them in Chroma via the unified ingestion pipeline.

Takes the output from ingest_curated_md.py, generates embeddings,
and persists to a Chroma collection.
"""

from ingestion.pipeline import ingest_corpus


def embed_and_store_chunks(collection_name: str = "enterprise_docs"):
    """
    Run the canonical ingestion pipeline (chunk -> embed -> store).

    Args:
        collection_name: Name of the Chroma collection to store in

    Returns:
        dict with ingestion summary
    """
    print("\n" + "=" * 80)
    print("UNIFIED INGESTION PIPELINE")
    print("=" * 80)

    result = ingest_corpus(
        collection_name=collection_name,
        reset=True,
        verbose=True,
    )

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"  Collection: {result['collection_name']}")
    print(f"  Chunks: {result['chunks_ingested']}")
    print(f"  Embeddings stored: {result['embeddings_stored']}")

    return {
        "collection_name": result["collection_name"],
        "num_chunks": result["chunks_ingested"],
        "num_embeddings_stored": result["embeddings_stored"],
        "embedding_model": "all-MiniLM-L6-v2",
    }


if __name__ == "__main__":
    embed_and_store_chunks()
