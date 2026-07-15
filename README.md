# Enterprise RAG Assistant

A retrieval-first RAG system for enterprise security documentation. The stack prioritizes **measurable retrieval quality**, **grounded answers with citations**, and **per-request observability** before trusting generative output.

---

## Problem Statement

Enterprise security teams rely on fragmented policy corpora—NIST publications, internal runbooks, and access-control standards spread across many Markdown files. Practitioners need fast, trustworthy answers, but face recurring gaps:

| Challenge | Impact |
|-----------|--------|
| **Fragmented knowledge** | Critical guidance is split across sections and documents with no single search surface. |
| **Unverifiable AI answers** | Generic chatbots invent policies or omit sources, creating compliance risk. |
| **Opaque retrieval** | Without logged chunks, prompts, and scores, failures are hard to debug. |
| **Weak context** | Low-similarity matches lead to confident but wrong answers. |

This project addresses those gaps with a pipeline that **retrieves evidence first**, **refuses to answer when context is weak**, **requires inline citations**, and **evaluates retrieval and answer quality** on a fixed question set.

**Corpus (curated):** `access_control_policy.md`, `incident_response_runbook.md`, `nist_800_53_selected_controls.md`, `nist_800_61_incident_response.md`

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OFFLINE: INGESTION                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  data/docs/curated/*.md                                                      │
│       │                                                                      │
│       ▼                                                                      │
│  ingestion/ingest_curated_md.py  ──►  section split (### / ##)              │
│       │                                  word chunks (250 words, 50 overlap) │
│       ▼                                                                      │
│  ingestion/pipeline.py  ──►  core/embeddings.py  ──►  core/vector_store.py  │
│                              all-MiniLM-L6-v2         ChromaDB (cosine)      │
│                              collection: enterprise_docs                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           ONLINE: QUERY PATH                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Client (FastAPI /ask, Streamlit → API, CLI)                                │
│       │                                                                      │
│       ▼                                                                      │
│   ┌──────────────────┐     trace_id + RequestLogger                          │
│   │ retrieve_chunks  │────► core/vector_store + core/embeddings             │
│   │                  │────► vector search (k=15)                             │
│   │                  │────► min similarity filter (≥ 0.40)                   │
│   │                  │────► cross-encoder rerank (optional)                  │
│   │                  │────► per-file cap (max 2) ──► final_k (default 3)     │
│   └────────┬─────────┘                                                       │
│            │ enriched chunks: rank, document_source, text_preview, scores    │
│            ▼                                                                 │
│   ┌──────────────────┐                                                       │
│   │ answer generation│────► confidence gate (weak context → "Not found")    │
│   │                  │────► LLM (OpenAI) OR retrieval-only cited snippets    │
│   │                  │────► mandatory [file.md - Section] citations          │
│   └────────┬─────────┘                                                       │
│            ▼                                                                 │
│   Response + trace_id  │  SQLite traces.db  │  traces/requests/{id}.json    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tech stack

| Layer | Technology |
|-------|------------|
| API | FastAPI, Pydantic, slowapi (rate limiting) |
| Core services | `core/` — config, embeddings, vector store, auth |
| UI | Streamlit (calls FastAPI backend via httpx) |
| Vectors | ChromaDB (persistent, cosine HNSW) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation | OpenAI API (`gpt-3.5-turbo`, optional) |
| Observability | JSON logs, SQLite traces, per-request JSON files |

---

## Security

| Control | Implementation |
|---------|----------------|
| **API key auth** | `API_KEY` env var; send via `X-API-Key` or `Authorization: Bearer` header |
| **Rate limiting** | slowapi — default `30/minute` per API key (or IP if no key) |
| **Input validation** | Question max length (`MAX_QUESTION_LENGTH=2000`); `final_k` capped at `MAX_FINAL_K` |
| **Health probes** | `GET /health` (liveness), `GET /ready` (corpus indexed) — no auth required |
| **Dev mode** | If `API_KEY` is unset, endpoints are unauthenticated (warning logged at startup) |

> **Production:** Always set `API_KEY`, disable `DEBUG_MODE`, and run without `--reload`.

Protected endpoints: `/ask`, `/ingest` (deprecated), `/traces/*`

---

## Ingestion Pipeline

**Canonical path:** `ingestion/pipeline.py` (used by CLI and deprecated API).

### 1. Section-aware parsing (`ingestion/ingest_curated_md.py`)

- Reads all `*.md` files under `data/docs/curated/`.
- Splits on `###` headers when present; otherwise falls back to `##`.
- Each section becomes a logical unit with `source_file` and `section_title` metadata.
- Long sections are word-chunked (**250 words**, **50-word overlap**); multi-part sections are labeled e.g. `Purpose (part 2/3)`.

### 2. Embedding and storage (`ingestion/pipeline.py` → `core/`)

- Uses shared `core/embeddings.py` singleton (no duplicate model loads).
- Stores in Chroma via `core/vector_store.py` with cosine distance and idempotent `reset=True` re-ingest.

```bash
python ingestion/embed_and_store.py
# or
python -c "from ingestion.pipeline import ingest_corpus; ingest_corpus(verbose=True)"
```

> **Note:** `POST /ingest` is **deprecated**. It now delegates to the unified pipeline but prefer the CLI for production ingestion.

---

## Retrieval + Generation Flow

### Retrieval pipeline (`retrieval/retrieve_chunks.py`)

**Dense-only (default, `HYBRID_SEARCH=false`):**

| Stage | Default | Purpose |
|-------|---------|---------|
| Bi-encoder search | `retrieve_k = 15` | Fetch vector candidates from Chroma |
| Similarity filter | `min_similarity ≥ 0.40` | Drop weak dense matches |
| Cross-encoder rerank | `rerank_top_n = 15` | Rescore query–passage pairs |
| Per-document cap | `max 2 per file` | Reduce single-doc dominance |
| Final selection | `final_k = 3` | Chunks passed to generation |

**Hybrid mode (`HYBRID_SEARCH=true`):**

| Stage | Default | Purpose |
|-------|---------|---------|
| Dense search | `retrieve_k = 15` | Chroma bi-encoder candidates |
| Sparse search | `bm25_retrieve_k = 15` | BM25 candidates via `rank_bm25` |
| Weighted RRF fusion | `alpha = 0.7` | `score = α/(k+rank_dense) + (1-α)/(k+rank_sparse)` |
| Threshold filter | dense ≥ 0.40 **or** BM25 ≥ 0.30 | Keeps keyword-strong matches |
| Cross-encoder rerank | optional | Same reranker as dense-only |
| Per-document cap + final_k | unchanged | Same downstream pipeline |

Metadata filtering is supported on both legs via `source_file`, `doc_type`, and `section_title`.

### API response (`POST /ask`)

```bash
# Dense-only (default)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"question": "What is the incident response process?"}'

# Hybrid + metadata filter
curl -X POST "http://localhost:8000/ask?hybrid_search=true&doc_type=policy" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"question": "What is least privilege?"}'
```

```json
{
  "answer": "...",
  "sources": ["access_control_policy.md - Purpose"],
  "confidence": 0.72,
  "confidence_reason": "Multiple sections retrieved; high similarity to query",
  "top_k": 3,
  "retrieved_chunks": [{ "rank": 1, "document_source": "...", "similarity_score": 0.81 }],
  "trace_id": "uuid"
}
```

---

## Quick Start

```bash
git clone https://github.com/rudzy123/Enterprise_Rag_Assistant.git
cd Enterprise_Rag_Assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env: set API_KEY and optionally OPENAI_API_KEY

# Index documents (canonical ingestion)
python ingestion/embed_and_store.py

# API server
uvicorn main:app --host 0.0.0.0 --port 8000

# Streamlit UI (calls API — ensure server is running)
streamlit run app/app.py

# Evaluation
PYTHONPATH=. python evals/run_evals.py -v

# Hybrid eval (requires re-ingest after enabling hybrid)
HYBRID_SEARCH=true python ingestion/embed_and_store.py
PYTHONPATH=. python evals/run_evals.py --hybrid -v

# Hybrid eval scoped to policies only
PYTHONPATH=. python evals/run_evals.py --hybrid --doc-type policy -v
```

### Health checks

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

---

## Configuration (environment)

Loaded from `.env` via `core/config.py` (`python-dotenv`).

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | *(empty)* | API authentication key; required in production |
| `API_URL` | `http://localhost:8000` | Backend URL for Streamlit |
| `RATE_LIMIT` | `30/minute` | slowapi rate limit per key/IP |
| `RETRIEVE_K` | `15` | Bi-encoder candidates |
| `FINAL_K` | `3` | Chunks returned to generation |
| `MAX_FINAL_K` | `10` | Upper bound for `final_k` query param |
| `MAX_QUESTION_LENGTH` | `2000` | Max characters in question body |
| `MIN_CHUNK_SIMILARITY` | `0.40` | Post-search filter |
| `MIN_SIMILARITY_THRESHOLD` | `0.35` | Relevance gate before answering |
| `LOW_CONFIDENCE_THRESHOLD` | `0.30` | Abstention threshold |
| `RERANK_ENABLED` | `true` | Cross-encoder reranking |
| `HYBRID_SEARCH` | `false` | Enable dense+BM25 hybrid retrieval |
| `HYBRID_ALPHA` / `ALPHA` | `0.7` | Dense weight in weighted RRF |
| `BM25_RETRIEVE_K` | `15` | BM25 candidate count |
| `RRF_K` | `60` | RRF rank smoothing constant |
| `BM25_MIN_SCORE` | `0.30` | Normalized BM25 threshold in hybrid mode |
| `OPENAI_MODEL` | `gpt-3.5-turbo` | Generation model |
| `NOT_FOUND_ANSWER` | `Not found` | Abstention text |
| `DEBUG_MODE` | `false` | FastAPI debug mode |

---

## Project Structure

```
Enterprise_Rag_Assistant/
├── main.py                     # FastAPI: /ask, /health, /ready, /traces
├── config.py                   # Backward-compat shim → core.config
├── core/
│   ├── config.py               # Central config + load_dotenv
│   ├── embeddings.py           # SentenceTransformer singleton
│   ├── vector_store.py         # Chroma client + collection management
│   └── auth.py                 # API key middleware
├── ingestion/
│   ├── ingest_curated_md.py    # Section-aware Markdown chunking
│   ├── pipeline.py             # Canonical ingest_corpus()
│   └── embed_and_store.py      # CLI wrapper
├── retrieval/                  # Search, rerank, BM25, RRF, structured logs
│   ├── retrieve_chunks.py      # Dense + optional hybrid retrieval
│   ├── bm25_store.py           # BM25 index build/load/search
│   ├── hybrid.py               # Weighted Reciprocal Rank Fusion
│   └── metadata_filter.py      # Chroma where + in-memory filters
├── answer_generation/          # Prompts, citations, confidence gating
├── observability/              # RequestLogger, SQLite traces
├── evals/                      # questions.jsonl, run_evals.py, metrics
├── app/                        # Streamlit UI (httpx → API)
├── data/docs/curated/          # Source Markdown corpus
├── chroma_db/                  # Vector store (generated)
└── traces/                     # traces.db + requests/*.json
```

---

## Known Limitations

| Area | Limitation |
|------|------------|
| **Retrieval** | Hybrid search is opt-in; re-ingest required to build BM25 index |
| **Auth** | Single shared API key (no RBAC or per-tenant keys) |
| **Scale** | Small curated corpus (~4 docs); Chroma local-only |
| **Traces** | May contain full prompts and document excerpts |
| **Ingest API** | `POST /ingest` deprecated; use CLI pipeline |

---

## License

See repository license terms. Curated documents are for knowledge-retrieval demonstration; verify against official sources for compliance use.
