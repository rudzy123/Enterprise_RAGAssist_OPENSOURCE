import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() in ("1", "true", "yes")

# Retrieval pipeline
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "15"))
FINAL_K = int(os.getenv("FINAL_K", "3"))
TOP_K = FINAL_K  # alias used by API callers
MAX_CHUNKS_PER_FILE = int(os.getenv("MAX_CHUNKS_PER_FILE", "2"))
MIN_CHUNK_SIMILARITY = float(os.getenv("MIN_CHUNK_SIMILARITY", "0.40"))
# Post-retrieval relevance gate and weak-retrieval detection (cosine similarity: 1 - distance)
MIN_SIMILARITY_THRESHOLD = float(os.getenv("MIN_SIMILARITY_THRESHOLD", "0.35"))

# Cross-encoder reranking (after bi-encoder retrieval, before per-doc cap)
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() in ("1", "true", "yes")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "15"))
RETRIEVAL_STRUCTURED_LOGS = os.getenv("RETRIEVAL_STRUCTURED_LOGS", "true").lower() in (
    "1",
    "true",
    "yes",
)

# Ingestion / chunking
CHUNK_WORD_LIMIT = int(os.getenv("CHUNK_WORD_LIMIT", "250"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "50"))

# Generation
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.3"))
NOT_FOUND_ANSWER = os.getenv("NOT_FOUND_ANSWER", "Not found")

# Observability
TRACES_DIR = BASE_DIR / "traces"
TRACE_DB_PATH = TRACES_DIR / "traces.db"
