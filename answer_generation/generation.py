"""
Answer generation with mandatory chunk citations and low-confidence abstention.

All answers are grounded exclusively in retrieved chunks. Weak retrieval context
returns a configured not-found response with a retrieval-quality confidence score.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import openai

from config import NOT_FOUND_ANSWER, OPENAI_MODEL
from answer_generation.confidence import assess_retrieval_context


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
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        answer = generate_retrieval_only_answer(query, chunks)
        return answer, None, {
            "llm_prompt": None,
            "model_response": answer,
            "generation_mode": "retrieval_only",
        }

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
    llm_prompt = {"model": OPENAI_MODEL, "messages": messages}

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

    if raw_response.strip().lower() == NOT_FOUND_ANSWER.lower():
        return NOT_FOUND_ANSWER, total_tokens, {
            "llm_prompt": llm_prompt,
            "model_response": raw_response,
            "generation_mode": "llm",
        }

    final_answer = ensure_cited_answer(raw_response, chunks)
    return final_answer, total_tokens, {
        "llm_prompt": llm_prompt,
        "model_response": raw_response,
        "generation_mode": "llm",
    }


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

    Returns:
        (answer_text, confidence, confidence_reason, token_usage, observability)
    """
    observability = {
        "llm_prompt": None,
        "model_response": None,
        "generation_mode": "abstained",
    }
    confidence, confidence_reason, is_weak = assess_retrieval_context(
        chunks,
        raw_candidate_count=raw_candidate_count,
    )

    if is_weak:
        observability["model_response"] = NOT_FOUND_ANSWER
        return NOT_FOUND_ANSWER, confidence, confidence_reason, None, observability

    if use_llm and (api_key or os.getenv("OPENAI_API_KEY")):
        answer, token_usage, llm_obs = generate_with_openai(query, chunks, api_key=api_key)
        observability.update(llm_obs)
    else:
        answer = generate_retrieval_only_answer(query, chunks)
        token_usage = None
        observability["generation_mode"] = "retrieval_only"
        observability["model_response"] = answer

    answer = ensure_cited_answer(answer, chunks)
    return answer, confidence, confidence_reason, token_usage, observability
