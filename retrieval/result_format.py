"""
Format retrieval results with rank, document source, and text preview.
"""

from __future__ import annotations

from typing import List

TEXT_PREVIEW_LEN = 120


def format_document_source(source_file: str, section_title: str) -> str:
    """Combine file and section into a single source label."""
    file_name = source_file or "Unknown"
    section = section_title or "Unknown"
    return f"{file_name} - {section}"


def make_text_preview(text: str, max_len: int = TEXT_PREVIEW_LEN) -> str:
    """Return a short preview of chunk text."""
    body = text or ""
    if len(body) <= max_len:
        return body
    return body[:max_len] + "..."


def enrich_retrieved_chunk(chunk: dict, rank: int) -> dict:
    """Add rank, document_source, and text_preview to a chunk dict."""
    enriched = dict(chunk)
    enriched["rank"] = rank
    enriched["document_source"] = format_document_source(
        chunk.get("source_file", "Unknown"),
        chunk.get("section_title", "Unknown"),
    )
    enriched["text_preview"] = make_text_preview(chunk.get("text") or "")
    return enriched


def enrich_retrieved_chunks(chunks: List[dict]) -> List[dict]:
    """Enrich each chunk with rank (1-based), document_source, and text_preview."""
    return [enrich_retrieved_chunk(chunk, rank=i + 1) for i, chunk in enumerate(chunks)]
