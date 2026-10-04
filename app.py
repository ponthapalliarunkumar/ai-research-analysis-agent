import os

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


def get_setting(name: str, default: str | None = None) -> str | None:
    """Read a setting from Streamlit Secrets first, then environment / .env."""
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name) or default


api_key = get_setting("GEMINI_API_KEY")
model_name = get_setting("GEMINI_MODEL")

# ------------------------------------------------------------- session state
if "kb" not in st.session_state:
    st.session_state.kb = KnowledgeBase(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed" not in st.session_state:
    st.session_state.indexed = set()

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# ------------------------------------------------------------------- sidebar
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
        key=f"uploader_{st.session_state.uploader_key}",
    )

    # Streamlit reruns on every interaction, so only index files we haven't seen.
    if uploaded and not api_key:
        st.warning("Add GEMINI_API_KEY before uploading documents.")
    elif uploaded:
        for file in uploaded:
            file_key = (file.name, file.size)
            if file_key in st.session_state.indexed:
                continue
            try:
                with st.spinner(f"Indexing {file.name}..."):
                    text = st.session_state.kb.extract_text(file)
                    if not text.strip():
                        st.warning(f"No readable text found in {file.name}.")
                        continue
                    st.session_state.kb.add_document(file.name, text)
                st.session_state.indexed.add(file_key)
            except Exception as exc:
                st.error(f"Could not index {file.name}: {exc}")

    st.metric("Indexed chunks", st.session_state.kb.chunk_count)

    if st.button("Clear knowledge base", use_container_width=True):
        st.session_state.kb.clear()
        st.session_state.indexed = set()
        st.session_state.uploader_key += 1  # resets the file uploader widget
        st.rerun()

    st.divider()
    st.subheader("API status")

    if api_key:
        st.success("✅ Gemini API key loaded")
    else:
        st.warning("⚠️ Gemini API key missing")

# --------------------------------------------------------------- chat history
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            st.download_button(
                "⬇️ Download report",
                data=message["content"],
                file_name="ai_research_report.md",
                mime="text/markdown",
                key=f"download_{i}",
            )

prompt = st.chat_input(
    "Example: Research how generative AI is changing cybersecurity in 2026."
)

if prompt:
    if not api_key:
        st.error(
            "GEMINI_API_KEY is not configured. "
            "Add it to Streamlit Secrets or your .env file."
        )
        st.stop()

    history = list(st.session_state.messages)  # earlier turns only

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔎 Researching and analyzing..."):
            try:
                agent = ResearchAgent(
                    st.session_state.kb,
                    api_key=api_key,
                    model=model_name,
                )
                answer = agent.run(prompt, history=history)
            except Exception as exc:
                st.error("Research agent error:")
                st.exception(exc)
                st.stop()

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.download_button(
            "⬇️ Download report",
            data=answer,
            file_name="ai_research_report.md",
            mime="text/markdown",
            key=f"download_{len(st.session_state.messages) - 1}",
        )