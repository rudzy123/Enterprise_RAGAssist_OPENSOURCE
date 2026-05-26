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
│  ingest_curated_md.py  ──►  section split (### / ##)  ──►  word chunks      │
│       │                         (250 words, 50 overlap)                      │
│       ▼                                                                      │
│  embed_and_store.py    ──►  all-MiniLM-L6-v2  ──►  ChromaDB (cosine)       │
│                              collection: enterprise_docs                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           ONLINE: QUERY PATH                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Client (FastAPI /ask, Streamlit, CLI)                                      │
│       │                                                                      │
│       ▼                                                                      │
│   ┌──────────────────┐     trace_id + RequestLogger                          │
│   │ retrieve_chunks  │────► vector search (k=15)                             │
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

┌─────────────────────────────────────────────────────────────────────────────┐
│                           EVALUATION (offline)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  evals/questions.jsonl (65 Q&A pairs)  ──►  run_evals.py                       │
│       │                                                                      │
│       ├── retrieval: P@k, recall@k, MRR, hit rate, abstention                 │
│       └── answers: citations %, groundedness %, confidence                   │
│            ▼                                                                 │
│       evals/results_YYYYMMDD_HHMMSS.json                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tech stack

| Layer | Technology |
|-------|------------|
| API | FastAPI, Pydantic |
| UI | Streamlit |
| Vectors | ChromaDB (persistent, cosine HNSW) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation | OpenAI API (`gpt-3.5-turbo`, optional) |
| Observability | JSON logs, SQLite traces, per-request JSON files |

---

## Ingestion Pipeline

Ingestion turns curated Markdown into searchable, citation-friendly chunks.

### 1. Section-aware parsing (`ingestion/ingest_curated_md.py`)

- Reads all `*.md` files under `data/docs/curated/`.
- Splits on `###` headers when present; otherwise falls back to `##`.
- Each section becomes a logical unit with `source_file` and `section_title` metadata.
- Long sections are word-chunked (**250 words**, **50-word overlap**); multi-part sections are labeled e.g. `Purpose (part 2/3)`.

### 2. Embedding and storage (`ingestion/embed_and_store.py`)

- Embeds chunk text with **all-MiniLM-L6-v2** (384-dim).
- Stores in Chroma collection **`enterprise_docs`** with cosine distance.
- Persists IDs, embeddings, full text, and metadata (`source_file`, `section_title`).

```bash
python ingestion/embed_and_store.py
```

> **Note:** `POST /ingest` in `main.py` is a simpler alternate path (section split on `##` only). For production ingestion, use the curated pipeline above.

---

## Retrieval + Generation Flow

### Retrieval pipeline (`retrieval/retrieve_chunks.py`)

| Stage | Default | Purpose |
|-------|---------|---------|
| Bi-encoder search | `retrieve_k = 15` | Fetch vector candidates from Chroma |
| Similarity filter | `min_similarity ≥ 0.40` | Drop weak matches (no fallback) |
| Cross-encoder rerank | `rerank_top_n = 15` | Rescore query–passage pairs |
| Per-document cap | `max 2 per file` | Reduce single-doc dominance |
| Final selection | `final_k = 3` | Chunks passed to generation |

Each returned chunk includes: `similarity_score`, `distance`, optional `rerank_score`, `document_source`, `text_preview`, and `rank`.

Structured JSON logs are emitted per stage when `RETRIEVAL_STRUCTURED_LOGS=true` (candidates, similarity filter, rerank, final selection, consolidated `retrieval_results`).

### Answer generation (`answer_generation/generation.py`)

1. **Assess retrieval context** — abstain if no chunks, top similarity &lt; 0.35, or composite confidence &lt; 0.3.
2. **Generate** — OpenAI with a retrieval-only prompt, or cited snippet concatenation when no API key.
3. **Enforce citations** — every non–`Not found` answer must contain `[file.md - Section]` labels; uncited LLM output falls back to cited snippets.

### API response (`POST /ask`)

```json
{
  "answer": "...",
  "sources": ["access_control_policy.md - Purpose"],
  "confidence": 0.72,
  "confidence_reason": "Multiple sections retrieved; high similarity to query",
  "top_k": 3,
  "retrieved_chunks": [ { "rank": 1, "document_source": "...", "similarity_score": 0.81, "text_preview": "..." } ],
  "trace_id": "uuid"
}
```

### Observability (per request)

Logged and stored for each `trace_id`:

- Query text  
- Retrieved chunk summaries  
- Full LLM prompt (system + user messages)  
- Raw model response  
- Final answer and latencies (retrieval / generation / total)  

Access via `GET /traces/{trace_id}` or `traces/requests/{trace_id}.json`.

---

## Evaluation Methodology

Evaluations run **all 65 questions** from `evals/questions.jsonl`. Each line includes `question`, `expected_answer`, and `source_doc_id`.

```bash
# Full RAG eval (retrieval + answer); uses OpenAI if OPENAI_API_KEY is set
PYTHONPATH=. python evals/run_evals.py evals/questions.jsonl -v

# Retrieval-only answers (no LLM)
PYTHONPATH=. python evals/run_evals.py --no-llm

# Parameter sweep over final_k and min_similarity
PYTHONPATH=. python evals/run_evals.py --sweep
```

Results are written to:

- `evals/results_YYYYMMDD_HHMMSS.json` (timestamped)  
- `evals/results.json` (latest)

### Retrieval metrics

| Metric | Definition |
|--------|------------|
| **Precision@k** | Relevant chunks in top-k slots ÷ k |
| **Recall@k** | Expected source files found in top-k ÷ expected count |
| **Hit@k** | At least one relevant chunk in top-k |
| **MRR** | Mean reciprocal rank of first relevant chunk |
| **Abstention rate** | Queries returning zero chunks after filtering |
| **Hit rate** | File-level: expected `source_doc_id` appears in returned set |

### Answer metrics

| Metric | Definition |
|--------|------------|
| **has_citations** | Answer contains at least one `[file - section]` label from retrieved chunks |
| **grounded** (heuristic) | Token overlap ≥ 40% with chunk text, or valid abstention when context is insufficient |
| **confidence** | Retrieval-quality score from similarity, chunk count, and source consistency |

### Summary output

```
Total questions:     65
% with citations:    82.3%
% grounded:          76.9%
Hit@k:               0.800
MRR:                 0.736
Abstention rate:     0.262
```

Analyze saved runs:

```bash
python evals/analyze_results.py evals/results.json
```

---

## Known Limitations

| Area | Limitation |
|------|------------|
| **Retrieval** | Pure dense retrieval only—no BM25/hybrid search; semantic drift can pull related but wrong documents (e.g. multiple NIST docs). |
| **Chunking** | Fixed word windows may split tables, lists, or cross-references awkwardly. |
| **Reranking** | Cross-encoder adds latency and CPU/GPU cost; disabled in some eval runs for speed. |
| **Groundedness eval** | Heuristic token overlap ≠ human judgment; does not measure factual correctness against `expected_answer`. |
| **Generation** | `gpt-3.5-turbo` may paraphrase; citation enforcement is pattern-based, not claim-level verification. |
| **Abstention** | Thresholds (`0.40` chunk filter, `0.35` relevance, `0.30` confidence) are static—not calibrated per domain. |
| **Scale** | Small curated corpus (~4 docs); behavior on large multi-tenant corpora is untested. |
| **Security** | No auth on API endpoints; traces may contain full prompts and document excerpts. |
| **Ingest API** | `POST /ingest` path lacks section metadata parity with the main ingestion pipeline. |

---

## Example Outputs

### Retrieval result (chunk record)

```json
{
  "rank": 1,
  "chunk_id": "access_control_policy_0_0",
  "source_file": "access_control_policy.md",
  "section_title": "Purpose",
  "document_source": "access_control_policy.md - Purpose",
  "similarity_score": 0.812,
  "distance": 0.188,
  "text_preview": "This policy defines requirements for managing logical access to organizational systems and data...",
  "text": "<full chunk text>"
}
```

### Grounded answer (retrieval-only mode)

**Question:** What principle must access follow according to the policy?

**Answer:**

```text
[access_control_policy.md - Access Requirements] Access must follow the principle of least privilege.
```

### Weak context (abstention)

**Question:** What is the password rotation policy for contractors?

**Answer:** `Not found`

**Confidence:** `0.0` — *Top similarity score (0.28) below relevance threshold (0.35)*

### Eval snippet (per question)

```json
{
  "question": "What is the purpose of the access control policy?",
  "expected_source": "access_control_policy.md",
  "metrics": {
    "answer": "[access_control_policy.md - Purpose] This policy defines requirements for managing logical access...",
    "has_citations": true,
    "grounded": true,
    "groundedness_score": 0.67,
    "confidence": 0.71,
    "hit_at_k": true,
    "precision_at_k": 0.67,
    "reciprocal_rank": 1.0,
    "abstained": false
  }
}
```

### Structured retrieval log (stdout)

```json
{
  "event": "retrieval_results",
  "trace_id": "a1b2c3d4-...",
  "query": "What are the incident response steps?",
  "top_k": 3,
  "count": 3,
  "results": [
    {
      "rank": 1,
      "document_source": "incident_response_runbook.md - Response Steps",
      "similarity_score": 0.79,
      "text_preview": "1. Identify and classify the incident 2. Contain affected systems..."
    }
  ]
}
```

---

## Quick Start

```bash
git clone https://github.com/rudzy123/Enterprise_Rag_Assistant.git
cd Enterprise_Rag_Assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Index documents
python ingestion/embed_and_store.py

# Optional: enable LLM answers
export OPENAI_API_KEY=sk-...

# API server
uvicorn main:app --reload

# Streamlit UI
streamlit run app/app.py

# Evaluation
PYTHONPATH=. python evals/run_evals.py -v
```

### Configuration (environment)

| Variable | Default | Description |
|----------|---------|-------------|
| `RETRIEVE_K` | `15` | Bi-encoder candidates |
| `FINAL_K` | `3` | Chunks returned to generation |
| `MIN_CHUNK_SIMILARITY` | `0.40` | Post-search filter |
| `MIN_SIMILARITY_THRESHOLD` | `0.35` | Relevance gate before answering |
| `LOW_CONFIDENCE_THRESHOLD` | `0.30` | Abstention threshold |
| `RERANK_ENABLED` | `true` | Cross-encoder reranking |
| `OPENAI_MODEL` | `gpt-3.5-turbo` | Generation model |
| `NOT_FOUND_ANSWER` | `Not found` | Abstention text |

---

## Project Structure

```
Enterprise_Rag_Assistant/
├── main.py                 # FastAPI: /ask, /ingest, /traces
├── config.py               # Central configuration
├── ingestion/              # Markdown → chunks → Chroma
├── retrieval/              # Search, rerank, structured logs
├── answer_generation/      # Prompts, citations, confidence gating
├── observability/          # RequestLogger, SQLite traces
├── evals/                  # questions.jsonl, run_evals.py, metrics
├── app/                    # Streamlit UI
├── data/docs/curated/      # Source Markdown corpus
├── chroma_db/              # Vector store (generated)
└── traces/                 # traces.db + requests/*.json
```

---

## License

See repository license terms. Curated documents are for knowledge-retrieval demonstration; verify against official sources for compliance use.
