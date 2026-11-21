"""RAG pipeline: ingest resume -> retrieve -> grounded answer."""
from pathlib import Path
from typing import Iterator

from . import config
from .chunker import chunk_text
from .llm import LLM
from .loader import find_documents, load_file
from .vectorstore import VectorStore

SYSTEM_PROMPT = """You are a resume assistant. You answer questions about the \
person whose resume is provided in the CONTEXT.

Rules:
- Use ONLY the information in the CONTEXT. Do not invent employers, dates, \
skills, degrees or numbers.
- If the answer is not in the CONTEXT, say: "That isn't mentioned in the resume."
- Be concise and factual. Use bullet points for lists of skills or projects.
- Refer to the person in the third person unless the user asks for a first-person \
answer (for example, a cover-letter style reply)."""


class ResumeRAG:
    def __init__(self, model: str = config.LLM_MODEL):
        self.store = VectorStore()
        self.llm = LLM(model=model)

    # ---- ingestion -------------------------------------------------------
    def ingest_file(self, path) -> int:
        path = Path(path)
        text = load_file(path)
        chunks = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        return self.store.add_chunks(chunks, source=path.name)

    def ingest_folder(self, folder=config.DATA_DIR, reset: bool = False) -> dict:
        if reset:
            self.store.reset()
        report = {}
        for doc in find_documents(folder):
            report[doc.name] = self.ingest_file(doc)
        return report

    # ---- querying --------------------------------------------------------
    def retrieve(self, question: str) -> list[dict]:
        return self.store.query(question, k=config.TOP_K)

    def _messages(self, question: str, contexts: list[dict], history: list[dict]):
        context = "\n\n---\n\n".join(c["text"] for c in contexts) or "(no context found)"
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        msgs += history[-6:]  # last 3 exchanges for follow-up questions
        msgs.append(
            {
                "role": "user",
                "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}",
            }
        )
        return msgs

    def ask_stream(
        self, question: str, history: list[dict] | None = None
    ) -> tuple[Iterator[str], list[dict]]:
        contexts = self.retrieve(question)
        stream = self.llm.stream(self._messages(question, contexts, history or []))
        return stream, contexts

    def ask(self, question: str, history: list[dict] | None = None) -> tuple[str, list[dict]]:
        stream, contexts = self.ask_stream(question, history)
        return "".join(stream), contexts
