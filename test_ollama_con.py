#!/usr/bin/env python3
"""Temporary Ollama connectivity check for local Llama generative cutover."""

from __future__ import annotations

import sys

import ollama
from core.config import OLLAMA_HOST, OLLAMA_MODEL


def _model_names(client: ollama.Client) -> set[str]:
    listing = client.list()
    models = getattr(listing, "models", None)
    if models is None and isinstance(listing, dict):
        models = listing.get("models") or []
    names: set[str] = set()
    for item in models or []:
        name = getattr(item, "model", None) or getattr(item, "name", None)
        if name is None and isinstance(item, dict):
            name = item.get("model") or item.get("name")
        if name:
            names.add(str(name))
            # Ollama often lists "llama3.2:latest"; also accept bare tag
            if ":" in str(name):
                names.add(str(name).split(":", 1)[0])
    return names


def main() -> int:
    host = (OLLAMA_HOST or "http://localhost:11434").rstrip("/")
    model = (OLLAMA_MODEL or "llama3.2").strip()

    print("=" * 60)
    print("Ollama connection test")
    print("=" * 60)
    print(f"Host:  {host}")
    print(f"Model: {model}")
    print()

    try:
        client = ollama.Client(host=host)
        names = _model_names(client)
        print(f"Status: CONNECTED ({len(names)} local model tag(s) visible)")
        if names:
            preview = ", ".join(sorted(names)[:12])
            print(f"Local models (sample): {preview}")
    except Exception as exc:
        print(f"Status: FAILED — cannot reach Ollama at {host}")
        print(f"Error:  {exc}")
        return 1

    target_ok = model in names or any(
        n == model or n.startswith(f"{model}:") for n in names
    )
    if target_ok:
        print(f"Model '{model}': FOUND locally")
    else:
        print(f"Model '{model}': NOT FOUND — pulling now...")
        try:
            for event in client.pull(model, stream=True):
                status = getattr(event, "status", None)
                if status is None and isinstance(event, dict):
                    status = event.get("status")
                if status:
                    print(f"  pull: {status}")
            print(f"Model '{model}': PULL COMPLETE")
        except Exception as exc:
            print(f"Status: FAILED — pull error for '{model}'")
            print(f"Error:  {exc}")
            return 1

    print()
    print("Smoke chat (one short turn)...")
    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            options={"temperature": 0.0, "num_predict": 16},
        )
        message = getattr(response, "message", None)
        if message is not None:
            content = (getattr(message, "content", None) or "").strip()
        elif isinstance(response, dict):
            content = (response.get("message") or {}).get("content", "").strip()
        else:
            content = ""
        print(f"Chat reply: {content!r}")
        print("Status: READY for generative pipeline")
        return 0
    except Exception as exc:
        print(f"Status: FAILED — chat error")
        print(f"Error:  {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
