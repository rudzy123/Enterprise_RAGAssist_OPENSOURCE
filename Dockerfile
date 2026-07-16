# syntax=docker/dockerfile:1
#
# Lightweight API image — does not bundle Ollama.
# Run Ollama separately:
#   - Host install + OLLAMA_HOST=http://host.docker.internal:11434, or
#   - docker compose --profile ollama up -d  → OLLAMA_HOST=http://ollama:11434
# Set LLM_PROVIDER=ollama and OLLAMA_MODEL=llama3.2 in .env.
FROM python:3.12-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# curl for container health checks
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Dependencies (cached layer) ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Application ---
COPY . .

RUN mkdir -p chroma_db traces logs \
    && chmod +x scripts/docker-entrypoint.sh \
    && useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

ENTRYPOINT ["./scripts/docker-entrypoint.sh"]
