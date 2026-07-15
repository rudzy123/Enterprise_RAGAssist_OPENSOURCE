"""
Per-request observability: query, retrieval, LLM prompt/response, and latency.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import List, Optional

from observability.traces import build_step_log, log_event
from retrieval.result_format import format_document_source, make_text_preview

REQUEST_LOGS_DIR = Path(__file__).resolve().parents[1] / "traces" / "requests"


def summarize_chunk(chunk: dict) -> dict:
    """Compact chunk summary for logs and trace storage."""
    source_file = chunk.get("source_file", "Unknown")
    section_title = chunk.get("section_title", "Unknown")
    return {
        "chunk_id": chunk.get("chunk_id"),
        "rank": chunk.get("rank"),
        "source_file": source_file,
        "section_title": section_title,
        "document_source": chunk.get("document_source")
        or format_document_source(source_file, section_title),
        "similarity_score": chunk.get("similarity_score"),
        "distance": chunk.get("distance"),
        "rerank_score": chunk.get("rerank_score"),
        "text_preview": chunk.get("text_preview")
        or make_text_preview(chunk.get("text") or ""),
    }


@dataclass
class RequestLogger:
    """Accumulates structured logs for a single request."""

    trace_id: str
    query: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat() + "Z")
    retrieved_chunks: List[dict] = field(default_factory=list)
    llm_prompt: Optional[dict] = None
    model_response: Optional[str] = None
    answer: Optional[str] = None
    generation_mode: Optional[str] = None
    latencies_ms: dict = field(default_factory=dict)
    step_logs: List[dict] = field(default_factory=list)
    _logger: Optional[logging.Logger] = field(default=None, repr=False)

    def bind_logger(self, logger: logging.Logger) -> None:
        self._logger = logger

    def add_step(self, event: str, details: dict | None = None) -> None:
        step = build_step_log(event, details)
        self.step_logs.append(step)
        if self._logger is not None:
            log_event(self._logger, event, trace_id=self.trace_id, **(details or {}))

    def log_query(self, query: str) -> None:
        self.query = query
        self.add_step("query_received", {"query": query, "query_length": len(query)})

    def log_retrieved_chunks(self, chunks: List[dict]) -> None:
        self.retrieved_chunks = [summarize_chunk(chunk) for chunk in chunks]
        self.add_step(
            "retrieved_chunks",
            {
                "count": len(self.retrieved_chunks),
                "chunks": self.retrieved_chunks,
            },
        )

    def log_llm_prompt(self, prompt: dict) -> None:
        self.llm_prompt = prompt
        self.add_step(
            "llm_prompt",
            {
                "model": prompt.get("model"),
                "message_count": len(prompt.get("messages", [])),
                "prompt": prompt,
            },
        )

    def log_model_response(self, response: str, *, token_usage: Optional[int] = None) -> None:
        self.model_response = response
        self.add_step(
            "model_response",
            {
                "response_length": len(response),
                "token_usage": token_usage,
                "response": response,
            },
        )

    def log_answer(self, answer: str, *, generation_mode: str) -> None:
        self.answer = answer
        self.generation_mode = generation_mode
        self.add_step(
            "answer_finalized",
            {
                "generation_mode": generation_mode,
                "answer_length": len(answer),
                "answer": answer,
            },
        )

    def log_latency(
        self,
        *,
        total_ms: Optional[float] = None,
        retrieval_ms: Optional[float] = None,
        generation_ms: Optional[float] = None,
    ) -> None:
        if retrieval_ms is not None:
            self.latencies_ms["retrieval"] = round(retrieval_ms, 2)
        if generation_ms is not None:
            self.latencies_ms["generation"] = round(generation_ms, 2)
        if total_ms is not None:
            self.latencies_ms["total"] = round(total_ms, 2)
        self.add_step("latency", dict(self.latencies_ms))

    def to_trace_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "created_at": self.created_at,
            "retrieved_chunks": self.retrieved_chunks,
            "llm_prompt": self.llm_prompt,
            "model_response": self.model_response,
            "answer": self.answer,
            "generation_mode": self.generation_mode,
            "retrieval_latency_ms": self.latencies_ms.get("retrieval"),
            "generation_latency_ms": self.latencies_ms.get("generation"),
            "latency_ms": self.latencies_ms.get("total"),
            "step_logs": self.step_logs,
        }

    def save_json_file(self, directory: Path | None = None) -> Path:
        """Persist a per-request JSON log file."""
        out_dir = directory or REQUEST_LOGS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.trace_id}.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_trace_dict(), handle, indent=2, ensure_ascii=False, default=str)
        return path
