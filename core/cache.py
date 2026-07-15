"""
Optional Redis cache for query embeddings and other hot keys.

When REDIS_URL is unset or Redis is unreachable, operations degrade gracefully
to a no-op (direct computation, no failure).
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional

from core.config import CACHE_ENABLED, CACHE_TTL_SECONDS, REDIS_URL
from observability.logging_config import setup_json_logger

logger = setup_json_logger("enterprise_rag.cache")

_redis_client: Any = None
_redis_available: Optional[bool] = None


def _get_redis_client():
    """Lazy-connect to Redis; returns None when unavailable."""
    global _redis_client, _redis_available

    if not CACHE_ENABLED or not REDIS_URL:
        _redis_available = False
        return None

    if _redis_available is False:
        return None

    if _redis_client is not None:
        return _redis_client

    try:
        import redis

        client = redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=2)
        client.ping()
        _redis_client = client
        _redis_available = True
        logger.info("Redis cache connected.", extra={"extra": {"redis_url": REDIS_URL}})
        return _redis_client
    except Exception as exc:
        _redis_available = False
        logger.warning(
            "Redis cache unavailable; continuing without cache.",
            extra={"extra": {"error": str(exc)}},
        )
        return None


def cache_ping() -> bool:
    """Return True when Redis is configured and responding."""
    client = _get_redis_client()
    if client is None:
        return False
    try:
        client.ping()
        return True
    except Exception:
        return False


def _cache_key(namespace: str, raw_key: str) -> str:
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]
    return f"rag:{namespace}:{digest}"


def cache_get(namespace: str, key: str) -> Optional[str]:
    """Fetch a cached string value, or None on miss / error."""
    client = _get_redis_client()
    if client is None:
        return None
    try:
        return client.get(_cache_key(namespace, key))
    except Exception as exc:
        logger.warning("Cache get failed.", extra={"extra": {"error": str(exc)}})
        return None


def cache_set(namespace: str, key: str, value: str, ttl: Optional[int] = None) -> None:
    """Store a string value with TTL (defaults to CACHE_TTL_SECONDS)."""
    client = _get_redis_client()
    if client is None:
        return
    try:
        client.setex(_cache_key(namespace, key), ttl or CACHE_TTL_SECONDS, value)
    except Exception as exc:
        logger.warning("Cache set failed.", extra={"extra": {"error": str(exc)}})


def cache_get_json(namespace: str, key: str) -> Optional[Any]:
    raw = cache_get(namespace, key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def cache_set_json(namespace: str, key: str, value: Any, ttl: Optional[int] = None) -> None:
    cache_set(namespace, key, json.dumps(value), ttl=ttl)
