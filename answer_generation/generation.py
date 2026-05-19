"""
Answer generation with mandatory chunk citations and low-confidence abstention.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import openai

from config import NOT_FOUND_ANSWER, OPENAI_MODEL
from answer_generation.confidence import compute_retrieval_confidence, is_low_confidence
from retrieval.similarity import chunk_similarity_score


def chunk_citation(chunk: dict) -> str:
    """Canonical inline citation label for a retrieved chunk."""
    return f"[{chunk['source_file']} - {chunk.get('section_title', 'section')}]"


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
- Answer using only information from the retrieved chunks above.
- Every factual statement MUST include an inline citation using an allowed label (e.g. [file.md - Section Name]).
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


def generate_with_openai(
    query: str,
    chunks: List[dict],
    *,
    api_key: Optional[str] = None,
) -> Tuple[str, Optional[int]]:
    """
    Generate an answer with mandatory citations.

    Returns:
        (answer_text, total_tokens)
    """
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        return "Error: OPENAI_API_KEY environment variable not set.", None

    context, citation_labels = format_chunks_for_prompt(chunks)
    prompt = build_generation_prompt(query, context, citation_labels)

    client = openai.OpenAI(api_key=key)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer only from provided retrieved chunks. "
                    f"Every claim needs an inline citation. If unsure, say exactly: {NOT_FOUND_ANSWER}"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=500,
        temperature=0.1,
    )

    answer = response.choices[0].message.content.strip()
    total_tokens = None
    if getattr(response, "usage", None):
        total_tokens = getattr(response.usage, "total_tokens", None)

    if answer.strip().lower() == NOT_FOUND_ANSWER.lower():
        return NOT_FOUND_ANSWER, total_tokens

    if not answer_has_chunk_citations(answer, chunks):
        return generate_retrieval_only_answer(query, chunks), total_tokens

    return answer, total_tokens


def generate_answer_from_chunks(
    query: str,
    chunks: List[dict],
    *,
    api_key: Optional[str] = None,
    confidence: Optional[float] = None,
    use_llm: bool = True,
) -> Tuple[str, float, str]:
    """
    Generate an answer from retrieved chunks with confidence gating.

    Returns:
        (answer_text, confidence, confidence_reason)
    """
    metadatas = [
        {"source_file": c["source_file"], "section_title": c.get("section_title")}
        for c in chunks
    ]
    similarity_scores = [chunk_similarity_score(c) for c in chunks]

    if confidence is None:
        confidence, confidence_reason = compute_retrieval_confidence(
            num_docs=len(chunks),
            similarity_scores=similarity_scores,
            metadatas=metadatas,
        )
    else:
        _, confidence_reason = compute_retrieval_confidence(
            num_docs=len(chunks),
            similarity_scores=similarity_scores,
            metadatas=metadatas,
        )

    if not chunks or is_low_confidence(confidence):
        return NOT_FOUND_ANSWER, confidence, confidence_reason

    if use_llm and (api_key or os.getenv("OPENAI_API_KEY")):
        answer, _ = generate_with_openai(query, chunks, api_key=api_key)
    else:
        answer = generate_retrieval_only_answer(query, chunks)

    return answer, confidence, confidence_reason
