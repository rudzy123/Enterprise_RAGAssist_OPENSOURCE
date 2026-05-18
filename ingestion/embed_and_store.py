"""
Embed chunks and store them in Chroma.

Takes the output from ingest_curated_md.py, generates embeddings,
and persists to a Chroma collection.
"""

import chromadb
from sentence_transformers import SentenceTransformer
from ingest_curated_md import ingest_curated_markdown


def embed_and_store_chunks(collection_name: str = "enterprise_docs"):
    """
    1. Get chunks from markdown ingestion
    2. Generate embeddings for each chunk's text
    3. Store in Chroma with metadata
    4. Print storage summary
    
    Args:
        collection_name: Name of the Chroma collection to store in
    
    Returns:
        dict with {collection_name, num_chunks, model_name}
    """
    
    # Step 1: Get chunks from ingestion pipeline
    print("\n" + "=" * 80)
    print("STEP 1: INGEST MARKDOWN CHUNKS")
    print("=" * 80)
    
    chunks = ingest_curated_markdown()
    print(f"\n✓ Ingested {len(chunks)} chunks")
    
    # Step 2: Initialize embedding model and Chroma client
    print("\n" + "=" * 80)
    print("STEP 2: INITIALIZE EMBEDDING MODEL & CHROMA")
    print("=" * 80)
    
    model_name = "all-MiniLM-L6-v2"
    embedding_model = SentenceTransformer(model_name)
    print(f"\n✓ Loaded embedding model: {model_name}")
    
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    print(f"✓ Initialized persistent Chroma client")
    
    # Cosine space: Chroma returns distance where similarity = 1 - distance
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"✓ Using collection: '{collection_name}'")
    
    # Step 3: Generate embeddings and prepare storage data
    print("\n" + "=" * 80)
    print("STEP 3: GENERATE EMBEDDINGS")
    print("=" * 80)
    
    ids_to_store = []
    embeddings_to_store = []
    documents_to_store = []
    metadatas_to_store = []
    
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        text = chunk["text"]
        
        # Generate embedding
        embedding = embedding_model.encode(text)
        
        # Prepare batch data
        ids_to_store.append(chunk_id)
        embeddings_to_store.append(embedding)
        documents_to_store.append(text)
        
        # Store metadata (without the text)
        metadata = {
            "source_file": chunk["source_file"],
            "section_title": chunk["section_title"],
        }
        metadatas_to_store.append(metadata)
    
    print(f"\n✓ Generated {len(ids_to_store)} embeddings")
    print(f"  Embedding dimension: {len(embeddings_to_store[0])}")
    
    # Step 4: Store all embeddings in Chroma
    print("\n" + "=" * 80)
    print("STEP 4: STORE IN CHROMA")
    print("=" * 80)
    
    collection.add(
        ids=ids_to_store,
        embeddings=embeddings_to_store,
        documents=documents_to_store,
        metadatas=metadatas_to_store,
    )
    
    print(f"\n✓ Stored {len(ids_to_store)} embeddings in Chroma")
    
    # Step 5: Verify storage
    print("\n" + "=" * 80)
    print("STEP 5: VERIFICATION")
    print("=" * 80)
    
    collection_count = collection.count()
    print(f"\n✓ Total documents in collection: {collection_count}")
    
    # Show sample of what was stored
    sample = collection.get(
        limit=2,
        include=["embeddings", "documents", "metadatas"]
    )
    
    print(f"\nSample stored chunks:")
    for chunk_id, doc, meta in zip(sample["ids"], sample["documents"], sample["metadatas"]):
        print(f"\n  ID: {chunk_id}")
        print(f"  Source: {meta['source_file']} - {meta['section_title']}")
        print(f"  Preview: {doc[:80]}...")
    
    # Summary
    print("\n" + "=" * 80)
    print("✓ COMPLETE")
    print("=" * 80)
    
    result = {
        "collection_name": collection_name,
        "num_chunks": len(chunks),
        "num_embeddings_stored": collection_count,
        "embedding_model": model_name,
    }
    
    print(f"\nSummary:")
    print(f"  Collection: {result['collection_name']}")
    print(f"  Chunks: {result['num_chunks']}")
    print(f"  Embeddings stored: {result['num_embeddings_stored']}")
    print(f"  Model: {result['embedding_model']}")
    
    return result


if __name__ == "__main__":
    embed_and_store_chunks()
