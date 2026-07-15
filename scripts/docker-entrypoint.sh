#!/usr/bin/env bash
# Docker entrypoint: start FastAPI with production-safe defaults.
set -euo pipefail

HOST="${UVICORN_HOST:-0.0.0.0}"
PORT="${UVICORN_PORT:-8000}"
WORKERS="${UVICORN_WORKERS:-1}"

echo "Starting Enterprise RAG API on ${HOST}:${PORT} (workers=${WORKERS})"

exec uvicorn main:app \
  --host "${HOST}" \
  --port "${PORT}" \
  --workers "${WORKERS}" \
  --proxy-headers \
  --forwarded-allow-ips="*"
