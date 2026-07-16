"""
Answer generation with structured, grounded, audit-ready responses.

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
import string
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

# Mandated response layout for LLM (and structured retrieval-only) answers.
SYSTEM_PROMPT = f"""You are an expert enterprise security and compliance consultant.
Answer ONLY from the retrieved evidence provided by the user. Do not use outside knowledge.
Never invent controls, policies, or sources.

If the evidence is insufficient to answer, respond with exactly: {NOT_FOUND_ANSWER}

When evidence is sufficient, your entire reply MUST use this exact markdown layout
(no preface, no closing remarks outside these sections):

---
## 1. Response
[A fluid, conversational, highly professional answer. Use bolding and structured
bullet points where helpful. Stay grounded in the evidence, but do NOT use robotic
meta-talk such as "According to Document 1", "The context says", or "Based on the
retrieved chunks". Write as a consultant speaking to a client.]

## 2. Thought Process & Reasoning
[2–3 sentences explaining how you arrived at the answer: which query keywords matched
which controls/sections, how overlapping guidance was synthesized across sources, and
what unrelated noise was filtered out.]

## 3. Source References
### Reference A: [Descriptive title of the match]
* **Source Document:** [Exact source_file / section metadata from the evidence block]
* **Relevance Score:** [similarity_score from the evidence block as 0.XX]
* **Exact Context Match:**
> "[Exact relevant snippet from that source — do not invent or heavily paraphrase the proof sentence.]"

[Add Reference B, C, … for each retrieved evidence block you relied on, in order.]
---

Rules:
- Use ONLY the retrieved evidence. Copy Source Document names and Relevance Scores from the evidence metadata.
- Include one Source Reference per evidence block that supports the Response (typically all provided blocks).
- Do not invent similarity scores or document names.
"""


def chunk_citation(chunk: dict) -> str:
    """Canonical inline citation label for a retrieved chunk."""
    source = chunk.get("document_source")
    if not source:
        source = f"{chunk['source_file']} - {chunk.get('section_title', 'section')}"
    return f"[{source}]"


def _format_relevance_score(chunk: dict) -> str:
    """Format similarity (or rerank) score as 0.XX for Source References."""
    score = chunk.get("similarity_score")
    if score is None and chunk.get("rerank_score") is not None:
        score = chunk["rerank_score"]
    try:
        return f"{float(score):.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _source_document_label(chunk: dict) -> str:
    """Human-readable source label including optional chunk sequence."""
    base = chunk.get("document_source") or (
        f"{chunk.get('source_file', 'unknown')} - {chunk.get('section_title', 'section')}"
    )
    rank = chunk.get("rank")
    chunk_id = chunk.get("chunk_id")
    parts = [base]
    if rank is not None:
        parts.append(f"rank {rank}")
    if chunk_id:
        parts.append(f"id {chunk_id}")
    if len(parts) == 1:
        return base
    return f"{base} ({', '.join(parts[1:])})"


def _reference_letter(index: int) -> str:
    """0 → A, 1 → B, …"""
    if 0 <= index < 26:
        return string.ascii_uppercase[index]
    return str(index + 1)


def format_chunks_for_prompt(chunks: List[dict]) -> Tuple[str, List[str]]:
    """
    Build evidence block with text + metadata for the LLM prompt.

    Each chunk includes source file, section, rank, similarity score, and raw text
    so the model can populate the Source References section accurately.
    """
    parts = []
    labels = []
    for i, chunk in enumerate(chunks):
        label = chunk_citation(chunk)
        labels.append(label)
        letter = _reference_letter(i)
        score = _format_relevance_score(chunk)
        source_doc = _source_document_label(chunk)
        text = (chunk.get("text") or "").strip()
        parts.append(
            f"### Evidence {letter}\n"
            f"- citation_label: {label}\n"
            f"- source_document: {source_doc}\n"
            f"- source_file: {chunk.get('source_file', '')}\n"
            f"- section_title: {chunk.get('section_title', '')}\n"
            f"- rank: {chunk.get('rank', i + 1)}\n"
            f"- similarity_score: {score}\n"
            f"- chunk_id: {chunk.get('chunk_id', '')}\n"
            f"- text:\n{text}"
        )
    return "\n\n".join(parts), labels


def answer_has_chunk_citations(answer: str, chunks: List[dict]) -> bool:
    """True if the answer cites retrieved sources (inline labels or Source References)."""
    if not chunks:
        return True
    if any(chunk_citation(chunk) in answer for chunk in chunks):
        return True
    # Structured layout: Source Document lines must mention retrieved sources
    if "## 3. Source References" in answer or "### Reference " in answer:
        return any(
            (chunk.get("source_file") or "") in answer
            or (chunk.get("document_source") or "") in answer
            for chunk in chunks
        )
    return False


def answer_has_structured_layout(answer: str) -> bool:
    """True if the answer includes the mandated three-section headings."""
    required = ("## 1. Response", "## 2. Thought Process", "## 3. Source References")
    return all(section in answer for section in required)


def build_generation_prompt(query: str, context: str, citation_labels: List[str]) -> str:
    labels_text = "\n".join(f"- {label}" for label in citation_labels)
    return f"""Retrieved evidence (use metadata for Source References; ground the Response only on text):

{context}

Citation labels (for traceability):
{labels_text}

User question: {query}

Produce the mandated three-section markdown answer (Response, Thought Process & Reasoning,
Source References). If evidence is insufficient, respond with exactly: {NOT_FOUND_ANSWER}
"""


def _chat_messages(query: str, chunks: List[dict]) -> Tuple[list, dict]:
    """Build chat messages and observability prompt payload for an LLM call."""
    context, citation_labels = format_chunks_for_prompt(chunks)
    prompt = build_generation_prompt(query, context, citation_labels)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    return messages, {"messages": messages}


def generate_retrieval_only_answer(question: str, chunks: List[dict]) -> str:
    """
    Structured fallback when no LLM is used: same three-section layout,
    with Response built from cited snippets.
    """
    if not chunks:
        return NOT_FOUND_ANSWER

    response_lines = []
    for chunk in chunks:
        cite = chunk_citation(chunk)
        snippet = chunk["text"].strip().replace("\n", " ")
        response_lines.append(f"- {cite} {snippet}")

    sources = []
    for i, chunk in enumerate(chunks):
        letter = _reference_letter(i)
        title = chunk.get("section_title") or chunk.get("source_file") or "Retrieved passage"
        sources.append(
            f"### Reference {letter}: {title}\n"
            f"* **Source Document:** {_source_document_label(chunk)}\n"
            f"* **Relevance Score:** {_format_relevance_score(chunk)}\n"
            f"* **Exact Context Match:**\n"
            f"> \"{chunk['text'].strip()}\""
        )

    thought = (
        "Retrieved passages were ranked by semantic similarity to the query; "
        "the Response lists the highest-scoring grounded snippets. "
        "No generative model was used (retrieval-only mode)."
    )

    return (
        "---\n"
        "## 1. Response\n"
        + "\n".join(response_lines)
        + "\n\n## 2. Thought Process & Reasoning\n"
        + thought
        + "\n\n## 3. Source References\n"
        + "\n\n".join(sources)
        + "\n---"
    )


def ensure_cited_answer(answer: str, chunks: List[dict]) -> str:
    """Ensure the answer is grounded; fall back to structured retrieval-only if not."""
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
    # If the model omitted the mandated layout but was still grounded, keep content;
    # structured layout is preferred when the model followed instructions.
    return final_answer, total_tokens, {
        "llm_prompt": llm_prompt,
        "model_response": raw_response,
        "generation_mode": generation_mode,
        "structured_layout": answer_has_structured_layout(final_answer),
    }


def _retrieval_only_result(query: str, chunks: List[dict]) -> Tuple[str, Optional[int], dict]:
    answer = generate_retrieval_only_answer(query, chunks)
    return answer, None, {
        "llm_prompt": None,
        "model_response": answer,
        "generation_mode": "retrieval_only",
        "structured_layout": answer_has_structured_layout(answer),
    }


def generate_with_openai(
    query: str,
    chunks: List[dict],
    *,
    api_key: Optional[str] = None,
) -> Tuple[str, Optional[int], dict]:
    """
    Generate a structured, grounded answer from retrieved chunks only.

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
        max_tokens=1200,
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
    Generate a structured answer via ``ollama.chat()`` (default model: llama3.2).

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
        response = ollama.Client(host=ollama_host).chat(
            model=ollama_model,
            messages=messages,
            options={
                "temperature": OLLAMA_TEMPERATURE,
                "num_predict": 1200,
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
