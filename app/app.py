"""
Enterprise RAG Assistant
Author: Rudolf Musika
License: CC BY-NC-ND 4.0

This software may be used as-is with user-supplied API keys.
Modification and redistribution are restricted.

Minimal Streamlit UI for Enterprise RAG Assistant.

Calls the FastAPI backend for consistent tracing, auth, and rate limiting.
"""

import os

import httpx
import streamlit as st

from core.config import API_KEY, API_URL, NOT_FOUND_ANSWER


def _api_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    return headers


def ask_api(question: str, final_k: int = 3) -> dict:
    """Submit a question to the FastAPI /ask endpoint."""
    response = httpx.post(
        f"{API_URL}/ask",
        headers=_api_headers(),
        json={"question": question},
        params={"final_k": final_k},
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()


def main():
    st.set_page_config(
        page_title="Enterprise RAG Assistant",
        page_icon="🤖",
        layout="wide",
    )

    st.title("🤖 Enterprise RAG Assistant")
    st.markdown(
        "Ask questions about enterprise security policies and incident response procedures."
    )

    with st.sidebar:
        st.header("⚙️ Configuration")
        st.text_input("API URL", value=API_URL, disabled=True)
        if API_KEY:
            st.success("✅ API key configured (from environment).")
        else:
            st.warning("⚠️ API_KEY not set. Backend must allow unauthenticated access.")

        st.header("🔑 OpenAI")
        st.caption(
            "OpenAI generation is configured on the API server via OPENAI_API_KEY. "
            "Set it in the server environment for LLM answers."
        )
        if os.getenv("OPENAI_API_KEY"):
            st.success("✅ OPENAI_API_KEY detected in this environment.")
        else:
            st.info("ℹ️ No OPENAI_API_KEY here. Server may still use retrieval-only mode.")

    st.header("❓ Ask a Question")
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is the incident response process?",
    )

    if st.button("🔍 Search", type="primary"):
        if not question.strip():
            st.error("Please enter a question.")
            return

        with st.spinner("Querying API..."):
            try:
                result = ask_api(question.strip())

                if result.get("trace_id"):
                    st.caption(f"Trace ID: `{result['trace_id']}`")

                answer = result.get("answer", NOT_FOUND_ANSWER)
                confidence = result.get("confidence", 0.0)
                confidence_reason = result.get("confidence_reason", "")
                retrieved_chunks = result.get("retrieved_chunks") or []

                if not retrieved_chunks and answer == NOT_FOUND_ANSWER:
                    st.warning(NOT_FOUND_ANSWER)
                    if confidence_reason:
                        st.caption(confidence_reason)
                    return

                st.header("📄 Retrieved Information")
                for chunk in retrieved_chunks:
                    score_hint = (
                        f"rerank {chunk['rerank_score']:.3f}"
                        if chunk.get("rerank_score") is not None
                        else f"sim {chunk['similarity_score']:.3f}"
                    )
                    with st.expander(
                        f"📋 Result {chunk['rank']}: {chunk['document_source']} ({score_hint})"
                    ):
                        st.markdown(f"**Source:** {chunk['document_source']}")
                        st.markdown(
                            f"**Bi-encoder similarity:** {chunk['similarity_score']:.4f}"
                        )
                        if chunk.get("rerank_score") is not None:
                            st.markdown(f"**Rerank score:** {chunk['rerank_score']:.4f}")
                        st.markdown(f"**Preview:** {chunk['text_preview']}")
                        st.text_area(
                            "Content:",
                            chunk["text"],
                            height=100,
                            disabled=True,
                        )

                st.header("🤖 Generated Answer")
                if answer == NOT_FOUND_ANSWER:
                    st.warning(f"{NOT_FOUND_ANSWER} (confidence: {confidence:.2f})")
                else:
                    st.success(f"Answer generated (confidence: {confidence:.2f})")
                    st.markdown(answer)
                if confidence_reason:
                    st.caption(confidence_reason)

            except httpx.HTTPStatusError as exc:
                detail = exc.response.text
                try:
                    detail = exc.response.json().get("detail", detail)
                except Exception:
                    pass
                st.error(f"API error ({exc.response.status_code}): {detail}")
            except httpx.ConnectError:
                st.error(
                    f"Cannot connect to API at {API_URL}. "
                    "Start the server with: `uvicorn main:app --reload`"
                )
            except Exception as exc:
                st.error(f"An error occurred: {exc}")

    st.markdown("---")
    st.markdown("*Built with Streamlit, FastAPI, ChromaDB, and OpenAI*")


if __name__ == "__main__":
    main()
