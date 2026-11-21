"""Streamlit chat UI.  Run: streamlit run app.py"""
import tempfile
from pathlib import Path

import streamlit as st

from src import config
from src.rag import ResumeRAG

st.set_page_config(page_title="Resume Chatbot", page_icon="📄", layout="centered")


@st.cache_resource(show_spinner="Loading embedding model and vector DB...")
def get_rag(model: str) -> ResumeRAG:
    return ResumeRAG(model=model)


with st.sidebar:
    st.header("Knowledge base")
    model = st.text_input("Ollama model", value=config.LLM_MODEL)
    rag = get_rag(model)
    rag.llm.model = model

    upload = st.file_uploader("Upload resume", type=["pdf", "docx", "txt", "md"])
    replace = st.checkbox("Replace existing index", value=True)
    if st.button("Index resume", disabled=upload is None, type="primary"):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / upload.name
            path.write_bytes(upload.getvalue())
            if replace:
                rag.store.reset()
            n = rag.ingest_file(path)
        st.success(f"Indexed {n} chunks from {upload.name}")

    if st.button("Index files in /data"):
        report = rag.ingest_folder(reset=replace)
        st.success(f"Indexed: {report}" if report else "No files found in /data")

    st.caption(f"Chunks in store: {rag.store.count()}")
    if st.button("Clear chat"):
        st.session_state.pop("messages", None)
        st.rerun()

st.title("📄 Resume Chatbot")
st.caption("Ask anything about the resume. Answers are grounded in the indexed document.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("e.g. What are the key skills and projects?"):
    if rag.store.count() == 0:
        st.warning("Index a resume first (sidebar).")
        st.stop()

    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            stream, contexts = rag.ask_stream(question, history)
            answer = st.write_stream(stream)
        except RuntimeError as e:
            st.error(str(e))
            st.stop()
        with st.expander("Retrieved context"):
            for c in contexts:
                st.markdown(f"**{c['source']}** · similarity {c['score']:.2f}")
                st.text(c["text"])
    st.session_state.messages.append({"role": "assistant", "content": answer})
