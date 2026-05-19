# Enterprise RAG Assistant

A production-style Retrieval-Augmented Generation (RAG) system built for enterprise security document retrieval and evaluation.

This project is engineered to prioritize retrieval quality over free-form generation: it uses pure vector search, measurable evaluation metrics, and structured debugging to improve real-world enterprise knowledge access.

## Project Overview

Enterprise teams use this system to query security documentation such as:
- NIST SP 800-53
- NIST SP 800-61
- Internal access control and incident response policies

The system was built to solve a practical problem: enterprise documentation is fragmented, and retrieval quality must be measured and improved before any generative layer is trusted.

## Key Features

- Semantic search over enterprise security content
- ChromaDB vector indexing for persistent retrieval
- Chunk-based retrieval with source metadata
- Evaluation framework for precision, recall, and hit rate
- Retrieval-only mode with no API key required
- Observability and trace debugging for retrieval failures

## Architecture

```
ingestion → embedding → vector DB → retrieval → rerank → evaluation
```

- **Ingestion**: Markdown documents are parsed, chunked, and embedded
- **Embedding**: SentenceTransformers convert text into vector representations
- **Vector DB**: ChromaDB stores document vectors and metadata
- **Retrieval**: Bi-encoder search (`all-MiniLM-L6-v2`) over top candidates (`retrieve_k=15`)
- **Reranking**: Cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) rescores filtered chunks
- **Evaluation**: Metrics drive tuning and monitor retrieval quality

## Tech Stack

- Python
- ChromaDB
- SentenceTransformers
- FastAPI
- SQLite (observability/tracing)

## Evaluation Framework

Retrieval quality is the core measure in this system. The evaluation framework computes:

- **precision@k**: Relevant chunks in the top-k slots / k
- **recall@k**: Expected documents found within the top-k chunks
- **MRR** (mean reciprocal rank): Average of 1/rank for the first relevant chunk
- **abstention_rate**: Fraction of queries where retrieval returns no chunks (below similarity threshold)
- **retrieval_hit / precision / recall**: File-level source overlap metrics (legacy summary)

Run a parameter sweep over `final_k` and `min_chunk_similarity` with `--sweep` to find the best retrieval config.

These metrics enable targeted improvements and expose retrieval tradeoffs clearly.

## Retrieval Optimization (Sprint 3)

This project uses evaluation signals to tune retrieval behavior:

- **retrieve_k=15** bi-encoder candidates, **final_k=3** after rerank and caps
- Cosine similarity threshold (`similarity = 1 - distance`, default `0.40`) filters weak matches (no fallback)
- **Cross-encoder rerank** (`ms-marco-MiniLM-L-6-v2`) improves ordering before per-document caps
- Max **2 chunks per source file** to reduce cross-document noise
- Evaluation feedback balances precision and recall for real data

Disable reranking with `RERANK_ENABLED=false` if you need faster CPU-only runs.

## Performance Results

From the latest evaluation on 65 test questions:

- **Retrieval Hit Rate**: 0.85
- **Average Precision**: 0.55
- **Average Recall**: 0.85

**Interpretation**:
- Strong recall indicates good document coverage
- Moderate precision reveals semantic over-retrieval noise
- This makes the system reliable for finding relevant enterprise context while still highlighting areas for tuning

## How to Run

### Install dependencies

```bash
git clone https://github.com/rudzy123/Enterprise_Rag_Assistant.git
cd Enterprise_Rag_Assistant
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Ingest documents

```bash
python ingestion/embed_and_store.py
```

### Run retrieval

```bash
python retrieval/retrieve_chunks.py
```

### Run evaluation

```bash
# Single config (precision@k, MRR, abstention rate)
PYTHONPATH=. python -m evals.run_evals

# Sweep final_k and min similarity
PYTHONPATH=. python -m evals.run_evals --sweep

# Custom sweep grid
PYTHONPATH=. python -m evals.run_evals --sweep --final-k-values 2,3,5 --min-similarity-values 0.35,0.40,0.45
```

### Run API server

```bash
python main.py
```

## Project Structure

- `retrieval/` — vector search and retrieval logic
- `evals/` — evaluation harness, metrics, and test questions
- `ingestion/` — document processing, embedding, and storage pipeline
- `data/` — enterprise security documents and curated sources
- `app/` — application code and UI integration

## Future Improvements

- Add reranking models to improve precision
- Implement hybrid keyword + vector search
- Layer LLM generation on top of retrieval with grounded citations
- Add adaptive monitoring for retrieval drift and document coverage

## Conclusion

This repository demonstrates a practical, evaluation-driven RAG architecture for enterprise knowledge retrieval. It emphasizes measurable retrieval quality, data-driven tuning, and production-ready observability rather than relying purely on generative output.
