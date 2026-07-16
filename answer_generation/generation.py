"""
Answer generation with mandatory chunk citations and low-confidence abstention.

All answers are grounded exclusively in retrieved chunks. Weak retrieval context
returns a configured not-found response with a retrieval-quality confidence score.

LLM priority (``LLM_PROVIDER`` from ``core.config``):
  1. ``LLM_PROVIDER=ollama`` → ``ollama.chat()`` with ``llama3.2``
  2. Else if ``OPENAI_API_KEY`` exists → OpenAI
  3. Else → retrieval-only

If Ollama is not running, log a warning and fall back to retrieval-only.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import ollama
from core.config import (
    LLM_PROVIDER,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    OPENAI_API_KEY,
)

import openai

from answer_generation.confidence import assess_retrieval_context
from config import NOT_FOUND_ANSWER, OPENAI_MODEL

logger = logging.getLogger("enterprise_rag.generation")


def chunk_citation(chunk: dict) -> str:
    """Canonical inline citation label for a retrieved chunk."""
    source = chunk.get("document_source")
    if not source:
        source = f"{chunk['source_file']} - {chunk.get('section_title', 'section')}"
    return f"[{source}]"


def format_chunks_for_prompt(chunks: List[dict]) -> Tuple[str, List[str]]:
    """Build labeled context block and list of citation labels."""
    parts = []
    labels = []
    for chunk in chunks:
        label = chunk_citation(chunk)
        labels.append(label)
        parts.append(f"{label}\n{chunk['text'].strip()}")
    return "\n\n".join(parts), labels


def answer_has_chunk_citations(answer: str, chunks: List[dict]) -> bool:
    """True if the answer cites at least one retrieved chunk label."""
    if not chunks:
        return True
    return any(chunk_citation(chunk) in answer for chunk in chunks)


def build_generation_prompt(query: str, context: str, citation_labels: List[str]) -> str:
    labels_text = "\n".join(f"- {label}" for label in citation_labels)
    return f"""You are a careful assistant that answers questions using ONLY the retrieved chunks below.

Retrieved chunks (citation labels must be copied exactly):
{context}

Allowed citation labels:
{labels_text}

Question: {query}

Rules:
- Use ONLY information from the retrieved chunks above. Do not use outside knowledge.
- Every sentence in your answer MUST include at least one inline citation using an allowed label.
- Do not invent sources or cite labels that are not listed above.
- If the chunks do not contain enough information to answer, respond with exactly: {NOT_FOUND_ANSWER}

Answer:"""


def _chat_messages(query: str, chunks: List[dict]) -> Tuple[list, dict]:
    """Build chat messages and observability prompt payload for an LLM call."""
    context, citation_labels = format_chunks_for_prompt(chunks)
    prompt = build_generation_prompt(query, context, citation_labels)
    system_message = (
        "You answer only from provided retrieved chunks. "
        "Do not use any outside knowledge. "
        "Every sentence must include an inline citation. "
        f"If unsure, respond with exactly: {NOT_FOUND_ANSWER}"
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    return messages, {"messages": messages}


def generate_retrieval_only_answer(question: str, chunks: List[dict]) -> str:
    """Return cited snippets from retrieved chunks (no LLM)."""
    if not chunks:
        return NOT_FOUND_ANSWER

    lines = []
    for chunk in chunks:
        cite = chunk_citation(chunk)
        snippet = chunk["text"].strip().replace("\n", " ")
        lines.append(f"{cite} {snippet}")

    return "\n".join(lines)


def ensure_cited_answer(answer: str, chunks: List[dict]) -> str:
    """Ensure the answer includes chunk citations or fall back to cited snippets."""
    if answer.strip().lower() == NOT_FOUND_ANSWER.lower():
        return NOT_FOUND_ANSWER
    if answer_has_chunk_citations(answer, chunks):
        return answer
    return generate_retrieval_only_answer("", chunks)


def _finalize_llm_answer(
    raw_response: str,
    chunks: List[dict],
    *,
    llm_prompt: dict,
    total_tokens: Optional[int],
    generation_mode: str,
) -> Tuple[str, Optional[int], dict]:
    if raw_response.strip().lower() == NOT_FOUND_ANSWER.lower():
        return NOT_FOUND_ANSWER, total_tokens, {
            "llm_prompt": llm_prompt,
            "model_response": raw_response,
            "generation_mode": generation_mode,
        }

    final_answer = ensure_cited_answer(raw_response, chunks)
    return final_answer, total_tokens, {
        "llm_prompt": llm_prompt,
        "model_response": raw_response,
        "generation_mode": generation_mode,
    }


def _retrieval_only_result(query: str, chunks: List[dict]) -> Tuple[str, Optional[int], dict]:
    answer = generate_retrieval_only_answer(query, chunks)
    return answer, None, {
        "llm_prompt": None,
        "model_response": answer,
        "generation_mode": "retrieval_only",
    }


def generate_with_openai(
    query: str,
    chunks: List[dict],
    *,
    api_key: Optional[str] = None,
) -> Tuple[str, Optional[int], dict]:
    """
    Generate an answer with mandatory citations from retrieved chunks only.

    Returns:
        (answer_text, total_tokens, observability)
    """
    key = api_key or OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not key:
        return _retrieval_only_result(query, chunks)

    messages, prompt_base = _chat_messages(query, chunks)
    llm_prompt = {**prompt_base, "model": OPENAI_MODEL, "provider": "openai"}

    client = openai.OpenAI(api_key=key)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        max_tokens=500,
        temperature=0.0,
    )

    raw_response = response.choices[0].message.content.strip()
    total_tokens = None
    if getattr(response, "usage", None):
        total_tokens = getattr(response.usage, "total_tokens", None)

    return _finalize_llm_answer(
        raw_response,
        chunks,
        llm_prompt=llm_prompt,
        total_tokens=total_tokens,
        generation_mode="llm",
    )


def generate_with_ollama(
    query: str,
    chunks: List[dict],
    *,
    model: Optional[str] = None,
    host: Optional[str] = None,
) -> Tuple[str, Optional[int], dict]:
    """
    Generate an answer via ``ollama.chat()`` (default model: llama3.2).

    If Ollama is not running / unreachable, logs a warning and returns retrieval-only.

    Returns:
        (answer_text, total_tokens, observability)
    """
    ollama_model = (model or OLLAMA_MODEL or "llama3.2").strip()
    ollama_host = (host or OLLAMA_HOST).rstrip("/")
    messages, prompt_base = _chat_messages(query, chunks)
    llm_prompt = {
        **prompt_base,
        "model": ollama_model,
        "provider": "ollama",
        "host": ollama_host,
        "temperature": OLLAMA_TEMPERATURE,
    }

    try:
        # ollama.chat() API; Client(host=...) respects OLLAMA_HOST
        response = ollama.Client(host=ollama_host).chat(
            model=ollama_model,
            messages=messages,
            options={
                "temperature": OLLAMA_TEMPERATURE,
                "num_predict": 500,
            },
        )
    except Exception as exc:
        logger.warning(
            "Ollama is not available (host=%s, model=%s); falling back to retrieval-only: %s",
            ollama_host,
            ollama_model,
            exc,
        )
        answer, tokens, obs = _retrieval_only_result(query, chunks)
        obs["llm_prompt"] = llm_prompt
        obs["ollama_error"] = str(exc)
        return answer, tokens, obs

    message = getattr(response, "message", None)
    if message is not None:
        raw_response = (getattr(message, "content", None) or "").strip()
    elif isinstance(response, dict):
        raw_response = (response.get("message") or {}).get("content", "").strip()
    else:
        raw_response = ""

    prompt_tokens = getattr(response, "prompt_eval_count", None)
    completion_tokens = getattr(response, "eval_count", None)
    if prompt_tokens is None and isinstance(response, dict):
        prompt_tokens = response.get("prompt_eval_count")
        completion_tokens = response.get("eval_count")

    total_tokens = None
    if prompt_tokens is not None or completion_tokens is not None:
        total_tokens = int(prompt_tokens or 0) + int(completion_tokens or 0)

    return _finalize_llm_answer(
        raw_response,
        chunks,
        llm_prompt=llm_prompt,
        total_tokens=total_tokens,
        generation_mode="llm",
    )


def generate_answer_from_chunks(
    query: str,
    chunks: List[dict],
    *,
    api_key: Optional[str] = None,
    use_llm: bool = True,
    raw_candidate_count: int = 0,
) -> Tuple[str, float, str, Optional[int], dict]:
    """
    Generate an answer from retrieved chunks with retrieval-quality confidence gating.

    Priority (respects ``LLM_PROVIDER`` from config):
      1. ``LLM_PROVIDER=ollama`` → ``ollama.chat()`` with llama3.2
         (if Ollama is down → warning + retrieval-only)
      2. Else if ``OPENAI_API_KEY`` exists → OpenAI
      3. Else → retrieval-only

    Returns:
        (answer_text, confidence, confidence_reason, token_usage, observability)
    """
    observability = {
        "llm_prompt": None,
        "model_response": None,
        "generation_mode": "abstained",
        "llm_provider": None,
    }
    confidence, confidence_reason, is_weak = assess_retrieval_context(
        chunks,
        raw_candidate_count=raw_candidate_count,
    )

    if is_weak:
        observability["model_response"] = NOT_FOUND_ANSWER
        return NOT_FOUND_ANSWER, confidence, confidence_reason, None, observability

    openai_key = (api_key if api_key is not None else OPENAI_API_KEY).strip()
    provider = (LLM_PROVIDER or "").strip().lower().replace("-", "_")

    if not use_llm or provider in ("retrieval_only", "none", "off"):
        selected = "retrieval_only"
    elif provider == "ollama":
        selected = "ollama"
    elif openai_key:
        selected = "openai"
    else:
        selected = "retrieval_only"

    observability["llm_provider"] = selected

    if selected == "ollama":
        answer, token_usage, llm_obs = generate_with_ollama(
            query,
            chunks,
            model=OLLAMA_MODEL or "llama3.2",
            host=OLLAMA_HOST,
        )
        # Graceful fallback when Ollama is not running
        if llm_obs.get("generation_mode") == "retrieval_only":
            observability["llm_provider"] = "retrieval_only"
        observability.update(llm_obs)
    elif selected == "openai":
        answer, token_usage, llm_obs = generate_with_openai(
            query, chunks, api_key=openai_key
        )
        observability.update(llm_obs)
    else:
        answer, token_usage, llm_obs = _retrieval_only_result(query, chunks)
        observability.update(llm_obs)

    answer = ensure_cited_answer(answer, chunks)
    return answer, confidence, confidence_reason, token_usage, observability
