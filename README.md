# Enterprise RAG Assistant

**A production-oriented, retrieval-first RAG framework for any Markdown (or text) corpus.**

Ship grounded answers with citations, confidence gating, hybrid search, and per-request traces—before you trust a generative model. Swap the sample NIST/policy docs for codebases, product specs, wikis, research papers, runbooks, or compliance corpora without changing the pipeline.

[![CI](https://github.com/rudzy123/Enterprise_RAGAssist_OPENSOURCE/actions/workflows/ci.yml/badge.svg)](https://github.com/rudzy123/Enterprise_RAGAssist_OPENSOURCE/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](LICENSE)

```bash
git clone https://github.com/rudzy123/Enterprise_RAGAssist_OPENSOURCE.git
cd Enterprise_RAGAssist_OPENSOURCE
```

---

## Why this exists

Most RAG demos optimize for fluent chat. Production systems fail on quieter problems: weak retrieval, uncited generation, silent abstention failures, and no way to replay a bad answer.

| Failure mode | What this stack does |
|--------------|----------------------|
| **Hallucinated “facts”** | Answers are grounded in retrieved chunks; weak context → configured abstention (`Not found`). |
| **Black-box retrieval** | Every `/ask` gets a `trace_id`, ranked chunks, scores, and optional full prompt/response logs. |
| **Keyword-blind dense search** | Optional dense + BM25 via weighted RRF for control IDs, error codes, and exact terms. |
| **No quality bar** | Fixed eval set + CI gate on hit@k, MRR, precision, groundedness, abstention rate. |
| **Opaque deploys** | Docker Compose, Redis cache profile, health/ready probes, structured JSON logs. |

The sample corpus happens to be enterprise security / NIST Markdown. The **framework is corpus-agnostic**: point `data/docs/curated/` (or your ingest path) at whatever documents matter to you.

---

## Key features

- **Retrieval-first design** — bi-encoder recall → similarity filter → optional cross-encoder rerank → per-document cap → `final_k`
- **Hybrid search** — dense + BM25 fused with weighted Reciprocal Rank Fusion (`HYBRID_SEARCH=true`)
- **Section-aware chunking** — Markdown `###` / `##` splits, word windows with overlap, part labels for long sections
- **Confidence gating** — refuse to generate when top similarity / aggregate confidence is below threshold
- **Structured generation** — Ollama (default Llama 3.2) or OpenAI; mandatory source grounding; three-part audit layout (Response / Thought Process / Source References)
- **Retrieval-only fallback** — cited snippets if no LLM is configured or Ollama is down
- **Metadata filters** — `source_file`, `doc_type`, `section_title` on dense and hybrid paths
- **Observability** — `RequestLogger`, SQLite `traces.db`, per-request JSON under `traces/requests/`
- **API hardening** — API key auth, slowapi rate limits, input length caps, `/health` · `/ready` · `/status`
- **Evals + CI gate** — `evals/run_evals.py` + `scripts/check_eval_gate.py` in GitHub Actions
- **Ops** — Dockerfile, Compose profiles (`ingest`, `cache`, `ollama`), optional Redis embedding cache

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OFFLINE: INGESTION                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  data/docs/curated/*.md   (or your corpus)                                   │
│       │                                                                      │
│       ▼                                                                      │
│  ingestion/ingest_curated_md.py  ──►  section split (### / ##)              │
│       │                                  word chunks (250 words, 50 overlap) │
│       ▼                                                                      │
│  ingestion/pipeline.py  ──►  core/embeddings.py  ──►  core/vector_store.py  │
│                              all-MiniLM-L6-v2         ChromaDB (cosine)      │
│                              (+ BM25 index when hybrid ingest runs)          │
│                              collection: enterprise_docs                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           ONLINE: QUERY PATH                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Client (FastAPI /ask, Streamlit → API, CLI, evals)                         │
│       │                                                                      │
│       ▼                                                                      │
│   ┌──────────────────┐     trace_id + RequestLogger                          │
│   │ retrieve_chunks  │────► dense search (k=15)                              │
│   │                  │────► [optional] BM25 + weighted RRF                   │
│   │                  │────► min similarity filter (≥ 0.40)                   │
│   │                  │────► cross-encoder rerank (optional)                  │
│   │                  │────► per-file cap (max 2) ──► final_k (default 3)     │
│   └────────┬─────────┘                                                       │
│            │ enriched chunks: rank, document_source, text_preview, scores    │
│            ▼                                                                 │
│   ┌──────────────────┐                                                       │
│   │ answer generation│────► confidence gate (weak context → abstain)        │
│   │                  │────► Ollama / OpenAI / retrieval-only                 │
│   │                  │────► grounded citations + structured sections         │
│   └────────┬─────────┘                                                       │
│            ▼                                                                 │
│   Response + trace_id  │  SQLite traces.db  │  traces/requests/{id}.json    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tech stack

| Layer | Technology |
|-------|------------|
| API | FastAPI, Pydantic, slowapi |
| Core | `core/` — config, embeddings, Chroma, auth, optional Redis cache |
| UI | Streamlit → FastAPI via httpx |
| Vectors | ChromaDB (persistent, cosine HNSW) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Sparse | `rank_bm25` + on-disk BM25 index |
| Rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation | Ollama (`llama3.2` default) · OpenAI · retrieval-only |
| Observability | JSON logs, SQLite traces, request JSON dumps |
| CI | Ruff · pytest · ingest + eval quality gate |

---

## Quick start

### Local (dev)

```bash
git clone https://github.com/rudzy123/Enterprise_RAGAssist_OPENSOURCE.git
cd Enterprise_RAGAssist_OPENSOURCE

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Set API_KEY for protected routes; leave unset only for local unauthenticated experiments

# Index the curated corpus (builds Chroma; BM25 index when hybrid ingest runs)
python ingestion/embed_and_store.py

# API
uvicorn main:app --host 0.0.0.0 --port 8000

# Optional UI (requires API up)
streamlit run app/app.py
```

**Optional local LLM (recommended):**

```bash
# Install from https://ollama.com , then:
ollama pull llama3.2

# .env (defaults already favor Ollama)
LLM_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

If Ollama is unreachable, generation falls back to **retrieval-only** (cited snippets), not OpenAI.

### Smoke test

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/ready
curl -s http://localhost:8000/status | python3 -m json.tool

curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"question": "What is the incident response process?"}' | python3 -m json.tool
```

Example response shape:

```json
{
  "answer": "## 1. Response\n...\n## 2. Thought Process & Reasoning\n...\n## 3. Source References\n...",
  "sources": ["incident_response_runbook.md - Overview"],
  "confidence": 0.72,
  "confidence_reason": "Multiple sections retrieved; high similarity to query",
  "top_k": 3,
  "retrieved_chunks": [
    {
      "rank": 1,
      "document_source": "incident_response_runbook.md - Overview",
      "similarity_score": 0.81
    }
  ],
  "trace_id": "uuid"
}
```

### Docker

```bash
cp .env.example .env          # set API_KEY; DEBUG_MODE=false
docker compose build

# One-time index (persists on chroma_data volume)
docker compose --profile ingest run --rm ingest

docker compose up -d api
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8000/ready | python3 -m json.tool
# /ready returns 503 until the corpus is indexed
```

**Optional Redis** (query-embedding cache):

```bash
# .env: REDIS_URL=redis://redis:6379/0  CACHE_ENABLED=true
docker compose --profile cache up -d
```

**Optional Compose Ollama:**

```bash
docker compose --profile ollama up -d
docker compose exec ollama ollama pull llama3.2
# .env: LLM_PROVIDER=ollama  OLLAMA_HOST=http://ollama:11434
docker compose up -d api
```

Host Ollama without the profile: `OLLAMA_HOST=http://host.docker.internal:11434` (Compose maps `host.docker.internal` on Linux).

---

## Retrieval pipeline

### Dense (default, `HYBRID_SEARCH=false`)

| Stage | Default | Purpose |
|-------|---------|---------|
| Bi-encoder search | `retrieve_k = 15` | Vector candidates from Chroma |
| Similarity filter | `≥ 0.40` | Drop weak matches |
| Cross-encoder rerank | `rerank_top_n = 15` | Query–passage rescoring |
| Per-document cap | max 2 / file | Reduce single-doc dominance |
| Final selection | `final_k = 3` | Context for generation |

### Hybrid (`HYBRID_SEARCH=true` or per-request `hybrid_search: true`)

| Stage | Default | Purpose |
|-------|---------|---------|
| Dense + BM25 | `15` each | Dual candidate lists |
| Weighted RRF | `α = 0.7` | `α/(k+r_dense) + (1-α)/(k+r_sparse)` |
| Threshold | dense ≥ 0.40 **or** BM25 ≥ 0.30 | Keep keyword-strong hits |
| Rerank → cap → `final_k` | unchanged | Same tail as dense-only |

Re-ingest after enabling hybrid so the BM25 index exists:

```bash
HYBRID_SEARCH=true python ingestion/embed_and_store.py
```

```bash
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "question": "What does AC-2 require for account management?",
    "hybrid_search": true,
    "metadata_filters": {"doc_type": "nist_reference"}
  }' | python3 -m json.tool
```

---

## Generation & LLM modes

Priority in `answer_generation/generation.py` / `core.config.resolve_llm_provider()`:

| Priority | Mode | When |
|----------|------|------|
| 1 | **Ollama** | `LLM_PROVIDER=ollama` (default) |
| 2 | **OpenAI** | Provider ≠ ollama **and** `OPENAI_API_KEY` set |
| 3 | **Retrieval-only** | Explicit `retrieval_only` / `none` / `off`, missing key, or Ollama down |

```bash
curl -s http://localhost:8000/status
# → resolved_provider: "ollama" | "openai" | "retrieval_only"
```

| Goal | `.env` |
|------|--------|
| Local Llama | `LLM_PROVIDER=ollama`, `OLLAMA_MODEL=llama3.2` |
| Force retrieval-only | `LLM_PROVIDER=retrieval_only` |
| Force OpenAI | `LLM_PROVIDER=openai` + `OPENAI_API_KEY` |

Successful LLM answers use a fixed markdown contract: **## 1. Response**, **## 2. Thought Process & Reasoning**, **## 3. Source References**, with exact citation labels for grounding checks.

---

## Security

| Control | Implementation |
|---------|----------------|
| API key | `API_KEY`; send `X-API-Key` or `Authorization: Bearer` |
| Rate limit | slowapi; default `30/minute` per key (or IP) |
| Input bounds | `MAX_QUESTION_LENGTH=2000`; `final_k` ≤ `MAX_FINAL_K` |
| Probes | `/health`, `/ready`, `/status` — no auth |
| Dev caveat | Empty `API_KEY` → unauthenticated (startup warning) |

Protected: `/ask`, `/ingest` (deprecated), `/traces/*`.

**Production:** set a strong `API_KEY`, `DEBUG_MODE=false`, no `--reload`, `--workers 1` (models load in-process).

---

## Production deployment

```bash
cp .env.example .env   # API_KEY, DEBUG_MODE=false
docker compose build
docker compose --profile ingest run --rm ingest
docker compose up -d api
```

| Item | Production setting |
|------|--------------------|
| `API_KEY` | Required strong secret |
| `DEBUG_MODE` | `false` |
| Secrets | Server-side only; never ship keys in the Streamlit client |
| Logs | JSON → stdout + rotating `logs/` |
| Traces | May contain prompts/excerpts — lock down `/traces/*` |

**Without Docker:**

```bash
pip install -r requirements.txt
cp .env.example .env
python ingestion/embed_and_store.py
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

---

## Evaluation

Retrieval + answer metrics over `evals/questions.jsonl`:

```bash
PYTHONPATH=. python evals/run_evals.py -v
PYTHONPATH=. python evals/run_evals.py --hybrid -v
PYTHONPATH=. python evals/run_evals.py --no-llm --no-rerank   # CI-style
```

**CI** (`.github/workflows/ci.yml`): ruff → pytest → ingest → eval → `scripts/check_eval_gate.py`.

Default gates (override via env):

| Metric | Default floor / ceiling |
|--------|-------------------------|
| hit@k | ≥ 0.70 |
| MRR | ≥ 0.65 |
| precision@k | ≥ 0.35 |
| pct_grounded | ≥ 0.60 |
| abstention rate | ≤ 0.35 |

```bash
python ingestion/embed_and_store.py
PYTHONPATH=. python evals/run_evals.py --no-llm --no-rerank
python scripts/check_eval_gate.py evals/results.json
```

---

## How to customize / extend

| Goal | Where to change |
|------|-----------------|
| **New corpus** | Drop `.md` into `data/docs/curated/` (or extend `ingest_curated_md.py`); re-run `ingestion/embed_and_store.py` |
| **Chunking** | `CHUNK_WORD_LIMIT`, `CHUNK_OVERLAP_WORDS`; section split logic in `ingestion/ingest_curated_md.py` |
| **Embeddings / collection** | `EMBEDDING_MODEL_NAME`, `CHROMA_*` in `core/config.py` |
| **Retrieval knobs** | `RETRIEVE_K`, `FINAL_K`, `MIN_CHUNK_SIMILARITY`, `RERANK_*`, hybrid env vars |
| **Answer format / prompts** | `SYSTEM_PROMPT` + `build_generation_prompt` in `answer_generation/generation.py` |
| **Abstention policy** | `answer_generation/confidence.py`, `MIN_SIMILARITY_THRESHOLD`, `LOW_CONFIDENCE_THRESHOLD` |
| **Eval set / gates** | `evals/questions.jsonl`, `scripts/check_eval_gate.py`, `EVAL_MIN_*` |
| **Auth / limits** | `core/auth.py`, `API_KEY`, `RATE_LIMIT` |
| **Caching** | `core/cache.py`, `REDIS_URL`, `CACHE_ENABLED` |

CLI helper (retrieve + generate): `python answer_generation/generate_answer.py`.

---

## Configuration reference

Loaded from `.env` via `core/config.py`.

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | *(empty)* | Auth key; required in production |
| `API_URL` | `http://localhost:8000` | Streamlit → API base URL |
| `RATE_LIMIT` | `30/minute` | Per key/IP |
| `RETRIEVE_K` | `15` | Dense candidates |
| `FINAL_K` | `3` | Chunks to generation |
| `MAX_FINAL_K` | `10` | Cap on `final_k` query param |
| `MAX_QUESTION_LENGTH` | `2000` | Question body limit |
| `MIN_CHUNK_SIMILARITY` | `0.40` | Post-search filter |
| `MIN_SIMILARITY_THRESHOLD` | `0.35` | Relevance abstention gate |
| `LOW_CONFIDENCE_THRESHOLD` | `0.30` | Confidence abstention |
| `RERANK_ENABLED` | `true` | Cross-encoder |
| `HYBRID_SEARCH` | `false` | Dense + BM25 default |
| `HYBRID_ALPHA` / `ALPHA` | `0.7` | Dense weight in RRF |
| `BM25_RETRIEVE_K` | `15` | Sparse candidates |
| `RRF_K` | `60` | RRF smoothing |
| `BM25_MIN_SCORE` | `0.30` | Hybrid BM25 threshold |
| `LLM_PROVIDER` | `ollama` | `ollama` \| `openai` \| `retrieval_only` |
| `OLLAMA_HOST` | `http://localhost:11434` | Alias: `OLLAMA_BASE_URL` |
| `OLLAMA_MODEL` | `llama3.2` | Chat model |
| `OLLAMA_TEMPERATURE` | `0.0` | Sampling |
| `OPENAI_API_KEY` | *(empty)* | OpenAI path |
| `OPENAI_MODEL` | `gpt-3.5-turbo` | OpenAI model |
| `NOT_FOUND_ANSWER` | `Not found` | Abstention text |
| `DEBUG_MODE` | `false` | Verbose API errors |
| `REDIS_URL` / `CACHE_ENABLED` | empty / `false` | Embedding cache |
| `EVAL_MIN_HIT_AT_K` | `0.70` | CI gate |
| `EVAL_MIN_MRR` | `0.65` | CI gate |

---

## Project structure

```
Enterprise_RAGAssist_OPENSOURCE/
├── main.py                      # FastAPI: /ask, /health, /ready, /status, /traces
├── Dockerfile / docker-compose.yml
├── .github/workflows/ci.yml     # Lint + unit tests + eval gate
├── scripts/
│   ├── docker-entrypoint.sh
│   └── check_eval_gate.py
├── core/                        # config, embeddings, vector store, auth, cache
├── ingestion/                   # section-aware MD → embed → Chroma (+ BM25)
├── retrieval/                   # dense, BM25, RRF, rerank, filters, logs
├── answer_generation/           # prompts, confidence, Ollama/OpenAI/retrieval-only
├── observability/               # RequestLogger, SQLite + JSON traces
├── evals/                       # questions.jsonl, run_evals.py, metrics
├── app/                         # Streamlit UI
├── tests/                       # pytest suite
├── data/docs/curated/           # Sample Markdown corpus (replace freely)
├── chroma_db/                   # Generated vector (+ BM25) store
└── traces/                      # Runtime traces (gitignored DB)
```

---

## Known limitations

| Area | Note |
|------|------|
| Hybrid | Opt-in; re-ingest required for BM25 |
| Auth | Single shared API key (no multi-tenant RBAC) |
| Scale | Local Chroma; sized for curated / mid-size corpora |
| Traces | May retain prompts and document excerpts |
| `POST /ingest` | Deprecated — prefer CLI `ingestion/embed_and_store.py` |

---

## License

This project is licensed under **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)**. See [LICENSE](LICENSE) for terms.

Sample / curated documents are for retrieval demos. For compliance or legal use, verify against primary sources.
