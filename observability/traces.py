import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

TRACES_DIR = Path(__file__).resolve().parents[1] / "traces"
DB_PATH = TRACES_DIR / "traces.db"
REQUEST_LOGS_DIR = TRACES_DIR / "requests"

_TRACE_EXTRA_COLUMNS = {
    "llm_prompt_json": "TEXT",
    "model_response": "TEXT",
    "generation_mode": "TEXT",
    "retrieval_latency_ms": "REAL",
    "generation_latency_ms": "REAL",
}


class JsonFormatter(logging.Formatter):
    def format(self, record):
        message = record.getMessage()
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": message,
        }

        # Include any structured data passed through record
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            payload.update(record.extra)

        # Preserve exception info if present
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def setup_json_logger(name: str = "enterprise_rag", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    return logger


class TraceStore:
    def __init__(self, db_path: Path = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_db()

    def _get_connection(self):
        return sqlite3.connect(str(self.db_path), timeout=30, check_same_thread=False)

    def _initialize_db(self):
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    query TEXT,
                    answer TEXT,
                    retrieved_chunks_json TEXT,
                    groundedness_score REAL,
                    failure_type TEXT,
                    confidence REAL,
                    confidence_reason TEXT,
                    latency_ms REAL,
                    token_usage INTEGER,
                    step_logs_json TEXT,
                    evaluation_json TEXT,
                    created_at TEXT
                )
                """
            )
            self._migrate_schema(conn)
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_traces_created_at
                ON traces(created_at DESC)
                """
            )

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(traces)")}
        for column, column_type in _TRACE_EXTRA_COLUMNS.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE traces ADD COLUMN {column} {column_type}")

    def save_trace(self, trace: dict):
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO traces (
                    trace_id,
                    query,
                    answer,
                    retrieved_chunks_json,
                    groundedness_score,
                    failure_type,
                    confidence,
                    confidence_reason,
                    latency_ms,
                    token_usage,
                    step_logs_json,
                    evaluation_json,
                    created_at,
                    llm_prompt_json,
                    model_response,
                    generation_mode,
                    retrieval_latency_ms,
                    generation_latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace.get("trace_id"),
                    trace.get("query"),
                    trace.get("answer"),
                    json.dumps(trace.get("retrieved_chunks", []), default=str),
                    trace.get("groundedness_score"),
                    trace.get("failure_type"),
                    trace.get("confidence"),
                    trace.get("confidence_reason"),
                    trace.get("latency_ms"),
                    trace.get("token_usage"),
                    json.dumps(trace.get("step_logs", []), default=str),
                    json.dumps(trace.get("evaluation", {}), default=str),
                    trace.get("created_at"),
                    json.dumps(trace.get("llm_prompt"), default=str)
                    if trace.get("llm_prompt") is not None
                    else None,
                    trace.get("model_response"),
                    trace.get("generation_mode"),
                    trace.get("retrieval_latency_ms"),
                    trace.get("generation_latency_ms"),
                ),
            )

    def get_recent_traces(self, limit: int = 20):
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT trace_id, query, answer, failure_type, groundedness_score, confidence, latency_ms, token_usage, created_at FROM traces ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()

        return [
            {
                "trace_id": row[0],
                "query": row[1],
                "answer": row[2],
                "failure_type": row[3],
                "groundedness_score": row[4],
                "confidence": row[5],
                "latency_ms": row[6],
                "token_usage": row[7],
                "created_at": row[8],
            }
            for row in rows
        ]

    def get_trace(self, trace_id: str):
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM traces WHERE trace_id = ?",
                (trace_id,),
            )
            row = cursor.fetchone()

        if not row:
            return None

        columns = [col[0] for col in cursor.description]
        trace = dict(zip(columns, row))
        trace["retrieved_chunks"] = json.loads(trace.get("retrieved_chunks_json", "[]"))
        trace["step_logs"] = json.loads(trace.get("step_logs_json", "[]"))
        trace["evaluation"] = json.loads(trace.get("evaluation_json", "{}"))
        llm_prompt_json = trace.get("llm_prompt_json")
        trace["llm_prompt"] = json.loads(llm_prompt_json) if llm_prompt_json else None
        trace.pop("retrieved_chunks_json", None)
        trace.pop("step_logs_json", None)
        trace.pop("evaluation_json", None)
        trace.pop("llm_prompt_json", None)
        return trace


def build_step_log(event: str, details: dict = None) -> dict:
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event": event,
        "details": details or {},
    }


def log_event(logger: logging.Logger, event: str, trace_id: str = None, **details):
    payload = {"event": event}
    if trace_id:
        payload["trace_id"] = trace_id
    payload.update(details)
    extra = {"extra": payload}
    logger.info(json.dumps(payload, default=str), extra=extra)
