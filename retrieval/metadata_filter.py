"""
Metadata filtering helpers for dense (Chroma) and sparse (BM25) retrieval.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

FilterValue = Union[str, int, float, List[Union[str, int, float]]]

FILTERABLE_METADATA_FIELDS = frozenset(
    {
        "source_file",
        "doc_type",
        "section_title",
    }
)


def normalize_metadata_filters(
    metadata_filters: Optional[Dict[str, FilterValue]],
) -> Optional[Dict[str, FilterValue]]:
    """Validate and normalize metadata filter keys."""
    if not metadata_filters:
        return None

    normalized: Dict[str, FilterValue] = {}
    for key, value in metadata_filters.items():
        if key not in FILTERABLE_METADATA_FIELDS:
            raise ValueError(
                f"Unsupported metadata filter '{key}'. "
                f"Allowed: {sorted(FILTERABLE_METADATA_FIELDS)}"
            )
        if value is None or value == "":
            continue
        if isinstance(value, list) and not value:
            continue
        normalized[key] = value

    return normalized or None


def build_chroma_where(metadata_filters: Optional[Dict[str, FilterValue]]) -> Optional[dict]:
    """Build a Chroma `where` clause from flat metadata filters."""
    filters = normalize_metadata_filters(metadata_filters)
    if not filters:
        return None

    clauses = []
    for key, value in filters.items():
        if isinstance(value, list):
            clauses.append({key: {"$in": value}})
        else:
            clauses.append({key: value})

    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _metadata_value_matches(actual: Any, expected: FilterValue) -> bool:
    if isinstance(expected, list):
        return actual in expected
    return actual == expected


def chunk_matches_filters(
    chunk: dict,
    metadata_filters: Optional[Dict[str, FilterValue]],
) -> bool:
    """Return True if a chunk dict matches all metadata filters."""
    filters = normalize_metadata_filters(metadata_filters)
    if not filters:
        return True

    metadata = {
        "source_file": chunk.get("source_file"),
        "doc_type": chunk.get("doc_type"),
        "section_title": chunk.get("section_title"),
    }
    for key, expected in filters.items():
        if not _metadata_value_matches(metadata.get(key), expected):
            return False
    return True


def filter_chunks(
    chunks: List[dict],
    metadata_filters: Optional[Dict[str, FilterValue]],
) -> List[dict]:
    """Filter an in-memory chunk list by metadata."""
    filters = normalize_metadata_filters(metadata_filters)
    if not filters:
        return chunks
    return [chunk for chunk in chunks if chunk_matches_filters(chunk, filters)]
