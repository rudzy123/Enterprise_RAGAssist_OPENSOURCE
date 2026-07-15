"""
Central logging configuration: JSON to console and rotating file.

All application loggers should be obtained via setup_json_logger().
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from core.config import LOG_BACKUP_COUNT, LOG_DIR, LOG_FILE, LOG_LEVEL, LOG_MAX_BYTES

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    """Emit single-line JSON log records for structured ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        import json
        from datetime import datetime, timezone

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if hasattr(record, "extra") and isinstance(record.extra, dict):
            payload.update(record.extra)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def _resolve_level(level_name: str) -> int:
    return getattr(logging, level_name.upper(), logging.INFO)


def configure_logging(
    *,
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
    force: bool = False,
) -> None:
    """
    Configure the root logger with console and optional rotating file handlers.

    Idempotent: safe to call multiple times unless force=True.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    resolved_level = _resolve_level(level or LOG_LEVEL)
    target_file = log_file or LOG_FILE

    root = logging.getLogger()
    root.setLevel(resolved_level)

    # Remove existing handlers when reconfiguring.
    if force:
        root.handlers.clear()

    formatter = JsonFormatter()

    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        console.setLevel(resolved_level)
        root.addHandler(console)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        file_handler = RotatingFileHandler(
            target_file,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(resolved_level)
        root.addHandler(file_handler)

    # Quiet noisy third-party loggers in production.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    _CONFIGURED = True


def setup_json_logger(
    name: str = "enterprise_rag",
    level: Optional[int] = None,
) -> logging.Logger:
    """
    Return a named logger using the shared JSON console + file configuration.
    """
    configure_logging()
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    else:
        logger.setLevel(_resolve_level(LOG_LEVEL))
    # Propagate to root handlers (console + file).
    logger.propagate = True
    return logger
