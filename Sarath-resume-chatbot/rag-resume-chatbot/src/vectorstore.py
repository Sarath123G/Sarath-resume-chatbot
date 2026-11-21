"""ChromaDB wrapper (persistent, cosine similarity) with local embeddings."""
import hashlib

import chromadb

from . import config


class VectorStore:
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.embedder = SentenceTransformer(config.EMBED_MODEL)
        self.client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self.collection = self._get_collection()

    def _get_collection(self):
        return self.client.get_or_create_collection(
            name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.encode(texts, normalize_embeddings=True).tolist()

    def add_chunks(self, chunks: list[str], source: str) -> int:
        if not chunks:
            return 0
        ids = [
            hashlib.md5(f"{source}:{i}:{c}".encode()).hexdigest()
            for i, c in enumerate(chunks)
        ]
        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=self._embed(chunks),
            metadatas=[{"source": source, "chunk": i} for i in range(len(chunks))],
        )
        return len(chunks)

    def query(self, question: str, k: int = config.TOP_K) -> list[dict]:
        total = self.count()
        if total == 0:
            return []
        res = self.collection.query(
            query_embeddings=self._embed([question]),
            n_results=min(k, total),
        )
        return [
            {"text": doc, "source": meta["source"], "score": 1 - dist}
            for doc, meta, dist in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(config.COLLECTION_NAME)
        self.collection = self._get_collection()
