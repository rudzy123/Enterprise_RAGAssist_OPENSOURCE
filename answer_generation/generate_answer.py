"""
Answer generation with citations from retrieved chunks.

Uses OpenAI API to generate answers based only on retrieved context,
with proper source citations.
"""

import os
import openai
from retrieval.retrieve_chunks import retrieve_similar_chunks


def generate_answer_with_citations(query: str):
    """
    Generate an answer from retrieved chunks with citations.

    Args:
        query: The user's question

    Returns:
        Formatted answer with citations
    """

    # Step 1: Retrieve relevant chunks
    print("\n" + "=" * 80)
    print("STEP 1: RETRIEVE RELEVANT CHUNKS")
    print("=" * 80)

    retrieved_results = retrieve_similar_chunks(query)

    if not retrieved_results:
        return "Error: Could not retrieve any chunks from the knowledge base."

    # Step 2: Format context for the LLM
    print("\n" + "=" * 80)
    print("STEP 2: FORMAT CONTEXT FOR LLM")
    print("=" * 80)

    context_parts = []
    sources = []

    for result in retrieved_results:
        # Format each chunk with its metadata
        chunk_text = f"""
Section: {result['section_title']}
Source: {result['source_file']}
Content: {result['text']}
"""
        context_parts.append(chunk_text)

        # Collect sources for citation
        source_ref = f"- {result['section_title']} ({result['source_file']})"
        sources.append(source_ref)

    context = "\n".join(context_parts)
    sources_list = "\n".join(sources)

    print(f"✓ Formatted {len(retrieved_results)} chunks for context")
    print(f"Context length: {len(context)} characters")

    # Step 3: Generate answer using OpenAI
    print("\n" + "=" * 80)
    print("STEP 3: GENERATE ANSWER WITH OPENAI")
    print("=" * 80)

    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Error: OPENAI_API_KEY environment variable not set."

    client = openai.OpenAI(api_key=api_key)

    # Simple prompt template
    prompt = f"""
You are a helpful assistant that answers questions based ONLY on the provided context.
If the answer is not in the context, say "Not found in provided documents".

Context:
{context}

Question: {query}

Instructions:
- Answer based only on the provided context
- Be concise but complete
- If information is not in the context, say "Not found in provided documents"
- Do not add external knowledge or assumptions

Answer:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based only on provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.1  # Low temperature for factual answers
        )

        answer = response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error calling OpenAI API: {e}"

    # Step 4: Format final output with citations
    print("\n" + "=" * 80)
    print("STEP 4: FORMAT FINAL OUTPUT")
    print("=" * 80)

    final_output = f"{answer}\n\nSources:\n{sources_list}"

    print("✓ Generated answer with citations")
    print(f"Answer length: {len(answer)} characters")
    print(f"Sources: {len(sources)} references")

    return final_output


if __name__ == "__main__":
    # Example usage - requires OPENAI_API_KEY environment variable
    test_query = "What is the incident response process?"

    print(f"Query: {test_query}")
    print("\n" + "=" * 100)

    answer = generate_answer_with_citations(test_query)

    print("\n" + "=" * 100)
    print("FINAL ANSWER:")
    print("=" * 100)
    print(answer)