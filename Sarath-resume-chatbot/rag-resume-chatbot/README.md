# Resume RAG Chatbot

A fully local, open-source chatbot that uses your resume as its knowledge base.

| Piece        | Choice                                   |
|--------------|------------------------------------------|
| LLM          | Llama 3.2 3B (any Ollama model works)    |
| Embeddings   | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector DB    | ChromaDB (persistent, cosine similarity) |
| UI           | Streamlit + CLI                          |
| Resume files | PDF, DOCX, TXT, MD                       |

## How it works

```
resume -> loader -> chunker -> embeddings -> ChromaDB
                                                |
question -> embed -> top-k similar chunks ------+
                         |
              prompt (resume-only rules) -> Ollama LLM -> streamed answer
```

## Setup

1. Install [Ollama](https://ollama.com) and pull a model:
   ```bash
   ollama pull llama3.2:3b
   ```
2. Create an environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. (Optional) `cp .env.example .env` to change the model or chunk settings.

## Run

**Web UI**
```bash
streamlit run app.py
```
Upload your resume in the sidebar, click **Index resume**, then chat.

**CLI**
```bash
# put your resume in data/ first (a sample_resume.txt is included)
python cli.py ingest --reset
python cli.py chat
python cli.py ask "What projects has the candidate worked on?"
```

## Project layout

```
app.py              Streamlit UI
cli.py              ingest / chat / ask commands
src/config.py       settings (env-overridable)
src/loader.py       PDF/DOCX/TXT reader
src/chunker.py      paragraph-aware chunking with overlap
src/vectorstore.py  ChromaDB + embeddings
src/llm.py          Ollama client (streaming)
src/rag.py          retrieval + grounded prompt
tests/              chunker tests (pytest)
```

## Tuning tips

- Answers too vague? Raise `TOP_K` (e.g. 6) or lower `CHUNK_SIZE`.
- Better quality: use `llama3.1:8b` or `mistral` via `LLM_MODEL`.
- Low-RAM machine: `llama3.2:1b` or `qwen2.5:1.5b`.
- Scanned (image-only) PDFs have no text layer; run OCR first.
- To swap the vector DB (FAISS, Qdrant), only `src/vectorstore.py` needs changing.
