"""
Enterprise RAG Assistant
Author: Rudolf Musika
License: CC BY-NC-ND 4.0

This software may be used as-is with user-supplied API keys.
Modification and redistribution are restricted.

Minimal Streamlit UI for Enterprise RAG Assistant.

Provides a simple interface to ask questions and get answers with citations.
API key is used in memory only and never saved.
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from retrieval.retrieve_chunks import retrieve_similar_chunks

def generate_answer_with_citations(query: str, api_key: str):
    """
    Generate an answer from retrieved chunks with citations.
    This is a simplified version of the answer_generation module.
    """
    import openai
    from sentence_transformers import SentenceTransformer
    import chromadb

    # Step 1: Retrieve relevant chunks
    retrieved_results = retrieve_similar_chunks(query)

    if not retrieved_results:
        return "Error: Could not retrieve any chunks from the knowledge base."

    # Step 2: Format context for the LLM
    context_parts = []
    sources = []

    for result in retrieved_results:
        chunk_text = f"""
Section: {result['section_title']}
Source: {result['source_file']}
Content: {result['text']}
"""
        context_parts.append(chunk_text)
        source_ref = f"- {result['section_title']} ({result['source_file']})"
        sources.append(source_ref)

    context = "\n".join(context_parts)
    sources_list = "\n".join(sources)

    # Step 3: Generate answer using OpenAI
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
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based only on provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.1
        )

        answer = response.choices[0].message.content.strip()
        return f"{answer}\n\nSources:\n{sources_list}"

    except Exception as e:
        return f"Error calling OpenAI API: {e}"


def main():
    st.set_page_config(
        page_title="Enterprise RAG Assistant",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 Enterprise RAG Assistant")
    st.markdown("Ask questions about enterprise security policies and incident response procedures.")

    # API Key input (in memory only)
    with st.sidebar:
        st.header("🔑 API Configuration")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key. Used only in memory for this session."
        )

        if not api_key:
            st.warning("⚠️ No API key provided. Answer generation will be disabled, but you can still see retrieved chunks.")
        else:
            st.success("✅ API key provided. Full RAG functionality enabled.")

    # Main question input
    st.header("❓ Ask a Question")
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is the incident response process?"
    )

    if st.button("🔍 Search", type="primary"):
        if not question.strip():
            st.error("Please enter a question.")
            return

        with st.spinner("Searching knowledge base..."):
            try:
                # Always perform retrieval
                retrieved_results = retrieve_similar_chunks(question)

                if not retrieved_results:
                    st.error("No relevant information found in the knowledge base.")
                    return

                # Display retrieved chunks
                st.header("📄 Retrieved Information")
                for i, result in enumerate(retrieved_results, 1):
                    score_hint = (
                        f"rerank {result['rerank_score']:.3f}"
                        if "rerank_score" in result
                        else f"sim {result['similarity_score']:.3f}"
                    )
                    with st.expander(f"📋 Result {i}: {result['section_title']} ({score_hint})"):
                        st.markdown(f"**Source:** {result['source_file']}")
                        st.markdown(f"**Section:** {result['section_title']}")
                        st.markdown(f"**Bi-encoder similarity:** {result['similarity_score']:.4f}")
                        if "rerank_score" in result:
                            st.markdown(f"**Rerank score:** {result['rerank_score']:.4f}")
                        st.text_area(
                            "Content:",
                            result['text'],
                            height=100,
                            disabled=True
                        )

                # Generate answer if API key provided
                if api_key:
                    st.header("🤖 Generated Answer")
                    with st.spinner("Generating answer..."):
                        answer = generate_answer_with_citations(question, api_key)

                        if "Error calling OpenAI API" in answer:
                            st.error(answer)
                        else:
                            st.success("Answer generated successfully!")
                            st.markdown(answer)
                else:
                    st.info("💡 **Tip:** Provide an OpenAI API key above to get AI-generated answers with citations.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit, ChromaDB, Sentence Transformers, and OpenAI*")


if __name__ == "__main__":
    main()
