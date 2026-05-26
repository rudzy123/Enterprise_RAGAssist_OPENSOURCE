"""
Answer generation with citations from retrieved chunks.

Uses OpenAI API to generate answers based only on retrieved context,
with mandatory inline citations and low-confidence abstention.
"""

import os

from answer_generation.generation import generate_answer_from_chunks
from config import NOT_FOUND_ANSWER
from retrieval.retrieve_chunks import retrieve_similar_chunks


def generate_answer_with_citations(query: str):
    """
    Generate an answer from retrieved chunks with citations.

    Args:
        query: The user's question

    Returns:
        Formatted answer with inline chunk citations, or "Not found"
    """
    print("\n" + "=" * 80)
    print("STEP 1: RETRIEVE RELEVANT CHUNKS")
    print("=" * 80)

    chunks = retrieve_similar_chunks(query)

    if not chunks:
        return NOT_FOUND_ANSWER

    print(f"\n✓ Retrieved {len(chunks)} chunks")

    print("\n" + "=" * 80)
    print("STEP 2: GENERATE ANSWER")
    print("=" * 80)

    answer, confidence, confidence_reason, _, _ = generate_answer_from_chunks(
        query,
        chunks,
        use_llm=bool(os.getenv("OPENAI_API_KEY")),
    )

    print(f"✓ Confidence: {confidence:.2f} ({confidence_reason})")
    print(f"✓ Answer length: {len(answer)} characters")

    return answer


if __name__ == "__main__":
    test_query = "What is the incident response process?"

    print(f"Query: {test_query}")
    print("\n" + "=" * 100)

    answer = generate_answer_with_citations(test_query)

    print("\n" + "=" * 100)
    print("FINAL ANSWER:")
    print("=" * 100)
    print(answer)
