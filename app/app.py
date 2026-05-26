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

import os
import sys
from pathlib import Path

import streamlit as st

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from answer_generation.generation import generate_answer_from_chunks
from config import NOT_FOUND_ANSWER
from retrieval.retrieve_chunks import retrieve_similar_chunks


def main():
    st.set_page_config(
        page_title="Enterprise RAG Assistant",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 Enterprise RAG Assistant")
    st.markdown("Ask questions about enterprise security policies and incident response procedures.")

    with st.sidebar:
        st.header("🔑 API Configuration")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key. Used only in memory for this session."
        )

        if not api_key:
            st.warning("⚠️ No API key provided. Answers use cited retrieval snippets only.")
        else:
            st.success("✅ API key provided. Full RAG functionality enabled.")

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
                retrieved_results = retrieve_similar_chunks(question)

                if not retrieved_results:
                    st.warning(NOT_FOUND_ANSWER)
                    return

                st.header("📄 Retrieved Information")
                for result in retrieved_results:
                    score_hint = (
                        f"rerank {result['rerank_score']:.3f}"
                        if "rerank_score" in result
                        else f"sim {result['similarity_score']:.3f}"
                    )
                    with st.expander(
                        f"📋 Result {result['rank']}: {result['document_source']} ({score_hint})"
                    ):
                        st.markdown(f"**Source:** {result['document_source']}")
                        st.markdown(f"**Bi-encoder similarity:** {result['similarity_score']:.4f}")
                        if "rerank_score" in result:
                            st.markdown(f"**Rerank score:** {result['rerank_score']:.4f}")
                        st.markdown(f"**Preview:** {result['text_preview']}")
                        st.text_area(
                            "Content:",
                            result["text"],
                            height=100,
                            disabled=True,
                        )

                st.header("🤖 Generated Answer")
                with st.spinner("Generating answer..."):
                    answer, confidence, confidence_reason, _, _ = generate_answer_from_chunks(
                        question,
                        retrieved_results,
                        api_key=api_key or None,
                        use_llm=bool(api_key),
                    )

                    if answer.startswith("Error calling OpenAI"):
                        st.error(answer)
                    elif answer == NOT_FOUND_ANSWER:
                        st.warning(f"{NOT_FOUND_ANSWER} (confidence: {confidence:.2f})")
                    else:
                        st.success(f"Answer generated (confidence: {confidence:.2f})")
                        st.markdown(answer)
                        st.caption(confidence_reason)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    st.markdown("---")
    st.markdown("*Built with Streamlit, ChromaDB, Sentence Transformers, and OpenAI*")


if __name__ == "__main__":
    main()
