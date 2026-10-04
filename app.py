import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agent import ResearchAgent
from knowledge_base import KnowledgeBase

load_dotenv()

st.set_page_config(
    page_title="AI Research & Analysis Agent",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 AI Research & Analysis Agent")
st.caption("GenAI + Agentic AI + RAG + Web Research")

with st.sidebar:
    st.header("Project")
    st.write(
        "A multi-tool research agent that plans a task, searches the web, "
        "retrieves information from uploaded documents, and generates a cited report."
    )

    st.divider()
    st.subheader("Knowledge Base")
    uploaded = st.file_uploader(
        "Upload TXT, PDF, or DOCX files",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
    )

    if "kb" not in st.session_state:
        st.session_state.kb = KnowledgeBase()

    if uploaded:
        added = 0
        for file in uploaded:
            try:
                text = st.session_state.kb.extract_text(file)
                if text.strip():
                    st.session_state.kb.add_document(file.name, text)
                    added += 1
            except Exception as exc:
                st.error(f"Could not read {file.name}: {exc}")
        if added:
            st.success(f"Added {added} document(s) to the knowledge base.")

    st.metric("Indexed chunks", st.session_state.kb.chunk_count)

    if st.button("Clear knowledge base", use_container_width=True):
        st.session_state.kb.clear()
        st.rerun()

    st.divider()
    st.subheader("API status")
    st.write("✅ Gemini API key loaded" if os.getenv("GEMINI_API_KEY") else "⚠️ Gemini API key missing")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input(
    "Example: Research how generative AI is changing cybersecurity in 2026."
)

if prompt:
    if not os.getenv("GEMINI_API_KEY"):
        st.error(
            "GEMINI_API_KEY is not configured. Add it to Streamlit Secrets or your .env file."
        )
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Researching and analyzing..."):
            try:
                agent = ResearchAgent(st.session_state.kb)
                answer = agent.run(prompt)
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

                st.download_button(
                    "⬇️ Download report",
                    data=answer,
                    file_name="ai_research_report.md",
                    mime="text/markdown",
                )
            except Exception as exc:
                error_message = (
                    "I couldn't complete the research task. "
                    f"Please check your API key and configuration. Details: {exc}"
                )
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )
