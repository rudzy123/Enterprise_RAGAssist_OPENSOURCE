"""
Central configuration for the Enterprise RAG Assistant.

Loads environment variables from a `.env` file when present.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Application
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() in ("1", "true", "yes")
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

# Security
API_KEY = os.getenv("API_KEY", "").strip()
RATE_LIMIT = os.getenv("RATE_LIMIT", "30/minute")

# Retrieval pipeline
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "15"))
FINAL_K = int(os.getenv("FINAL_K", "3"))
TOP_K = FINAL_K
MAX_FINAL_K = int(os.getenv("MAX_FINAL_K", "10"))
MAX_CHUNKS_PER_FILE = int(os.getenv("MAX_CHUNKS_PER_FILE", "2"))
MIN_CHUNK_SIMILARITY = float(os.getenv("MIN_CHUNK_SIMILARITY", "0.40"))
MIN_SIMILARITY_THRESHOLD = float(os.getenv("MIN_SIMILARITY_THRESHOLD", "0.35"))

# Cross-encoder reranking
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() in ("1", "true", "yes")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "15"))
RETRIEVAL_STRUCTURED_LOGS = os.getenv("RETRIEVAL_STRUCTURED_LOGS", "true").lower() in (
    "1",
    "true",
    "yes",
)

# Hybrid search (dense + BM25 via weighted RRF)
HYBRID_SEARCH = os.getenv("HYBRID_SEARCH", "false").lower() in ("1", "true", "yes")
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", os.getenv("ALPHA", "0.7")))
BM25_RETRIEVE_K = int(os.getenv("BM25_RETRIEVE_K", "15"))
RRF_K = int(os.getenv("RRF_K", "60"))
BM25_MIN_SCORE = float(os.getenv("BM25_MIN_SCORE", "0.30"))
BM25_INDEX_DIR = BASE_DIR / os.getenv("BM25_INDEX_DIR", "chroma_db/bm25")

# Ingestion / chunking
CHUNK_WORD_LIMIT = int(os.getenv("CHUNK_WORD_LIMIT", "250"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "50"))
CURATED_DOCS_DIR = BASE_DIR / os.getenv("CURATED_DOCS_DIR", "data/docs/curated")

# Embeddings / vector store
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
CHROMA_DB_PATH = BASE_DIR / os.getenv("CHROMA_DB_PATH", "chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "enterprise_docs")
COLLECTION_NAME = CHROMA_COLLECTION_NAME

# Generation
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.3"))
NOT_FOUND_ANSWER = os.getenv("NOT_FOUND_ANSWER", "Not found")
MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "2000"))

# Observability / logging
TRACES_DIR = BASE_DIR / "traces"
TRACE_DB_PATH = TRACES_DIR / "traces.db"
LOG_DIR = BASE_DIR / os.getenv("LOG_DIR", "logs")
LOG_FILE = LOG_DIR / os.getenv("LOG_FILE", "enterprise_rag.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024)))  # 10 MB
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))

# Optional Redis cache (embedding / hot-key cache)
REDIS_URL = os.getenv("REDIS_URL", "").strip()
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "false").lower() in ("1", "true", "yes")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))

# Server
UVICORN_HOST = os.getenv("UVICORN_HOST", "0.0.0.0")
UVICORN_PORT = int(os.getenv("UVICORN_PORT", "8000"))
UVICORN_WORKERS = int(os.getenv("UVICORN_WORKERS", "1"))

# CI / eval gate thresholds (used by scripts/check_eval_gate.py)
EVAL_MIN_HIT_AT_K = float(os.getenv("EVAL_MIN_HIT_AT_K", "0.70"))
EVAL_MIN_MRR = float(os.getenv("EVAL_MIN_MRR", "0.65"))
EVAL_MIN_PRECISION_AT_K = float(os.getenv("EVAL_MIN_PRECISION_AT_K", "0.35"))
EVAL_MAX_ABSTENTION_RATE = float(os.getenv("EVAL_MAX_ABSTENTION_RATE", "0.35"))
EVAL_MIN_PCT_GROUNDED = float(os.getenv("EVAL_MIN_PCT_GROUNDED", "0.60"))
