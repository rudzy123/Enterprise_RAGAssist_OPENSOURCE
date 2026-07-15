"""
Unified ingestion pipeline: curated Markdown -> chunks -> embeddings -> Chroma + BM25.

This is the canonical ingestion path for both CLI and the deprecated API endpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from core.config import CHROMA_COLLECTION_NAME, CURATED_DOCS_DIR
from core.embeddings import embed_texts
from core.vector_store import add_chunks
from ingestion.ingest_curated_md import ingest_curated_markdown
from retrieval.bm25_store import build_bm25_index


def ingest_corpus(
    *,
    docs_dir: Optional[Path] = None,
    collection_name: str = CHROMA_COLLECTION_NAME,
    reset: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Ingest curated Markdown documents into the vector store and BM25 index.

    Args:
        docs_dir: Directory containing curated markdown files.
        collection_name: Target Chroma collection name.
        reset: When True, recreate the collection before insert (idempotent).
        verbose: Print progress messages.

    Returns:
        Summary dict with chunk and storage counts.
    """
    target_dir = docs_dir or CURATED_DOCS_DIR
    if not target_dir.exists():
        raise FileNotFoundError(f"Document directory not found: {target_dir}")

    md_files = sorted(target_dir.glob("*.md"))
    if not md_files:
        raise ValueError(f"No markdown files found in {target_dir}")

    chunks = ingest_curated_markdown(docs_dir=target_dir)
    if not chunks:
        raise ValueError(f"No chunks produced from documents in {target_dir}")

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)

    ids = [chunk["chunk_id"] for chunk in chunks]
    documents = texts
    metadatas = [
        {
            "source_file": chunk["source_file"],
            "section_title": chunk["section_title"],
            "doc_type": chunk.get("doc_type", "curated_md"),
        }
        for chunk in chunks
    ]

    stored_count = add_chunks(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        collection_name=collection_name,
        reset=reset,
    )

    build_bm25_index(
        chunk_ids=ids,
        documents=documents,
        metadatas=metadatas,
        collection_name=collection_name,
    )

    if verbose:
        print(f"Ingested {len(chunks)} chunks from {len(md_files)} files")
        print(f"Collection '{collection_name}' now has {stored_count} documents")
        print(f"BM25 index built for collection '{collection_name}'")

    return {
        "status": "ingested",
        "collection_name": collection_name,
        "source_directory": str(target_dir),
        "documents_found": len(md_files),
        "chunks_ingested": len(chunks),
        "embeddings_stored": stored_count,
        "bm25_index_built": True,
        "reset": reset,
    }
