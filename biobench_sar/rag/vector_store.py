from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from biobench_sar.rag.ingestor import Chunk
from biobench_sar.schemas import RetrievedChunk

Embedder = Callable[[list[str]], list[list[float]]]


class VectorStore:
    def __init__(
        self,
        persist_dir: Path,
        collection: str = "biobench",
        embed_model: str = "all-MiniLM-L6-v2",
        embedder: Embedder | None = None,
    ) -> None:
        import chromadb

        self._lock = threading.RLock()
        # Separate lock: loading the embedding model takes seconds and must not block count()/status polls.
        self._model_lock = threading.Lock()
        self._embed_model = embed_model
        self._embedder = embedder
        self._st_model: Any = None
        self._collection_name = collection
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._col = self._open()
        self._count = int(self._col.count())

    def _open(self) -> Any:
        return self._client.get_or_create_collection(
            self._collection_name, metadata={"hnsw:space": "cosine"}
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._embedder is not None:
            return [list(map(float, v)) for v in self._embedder(texts)]
        with self._model_lock:
            if self._st_model is None:
                from sentence_transformers import SentenceTransformer

                self._st_model = SentenceTransformer(self._embed_model)
            model = self._st_model
        return model.encode(texts, normalize_embeddings=True).tolist()

    @staticmethod
    def _id(chunk: Chunk, index: int) -> str:
        return hashlib.sha1(f"{chunk.source}{index}{chunk.text}".encode()).hexdigest()[:16]

    def add_chunks(self, chunks: list[Chunk]) -> int:
        chunks = [c for c in chunks if c.text.strip()]
        if not chunks:
            return 0
        embeddings = self._embed([c.text for c in chunks])
        with self._lock:
            self._col.upsert(
                ids=[self._id(c, i) for i, c in enumerate(chunks)],
                documents=[c.text for c in chunks],
                embeddings=embeddings,
                metadatas=[{**c.metadata, "source": c.source} for c in chunks],
            )
            self._count = int(self._col.count())
        return len(chunks)

    def query(self, text: str, k: int = 4) -> list[RetrievedChunk]:
        n = self._count
        if n == 0 or not text.strip() or k <= 0:
            return []
        emb = self._embed([text])
        with self._lock:
            res = self._col.query(
                query_embeddings=emb,
                n_results=min(k, n),
                include=["documents", "metadatas", "distances"],
            )
        out: list[RetrievedChunk] = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            meta = dict(meta or {})
            out.append(RetrievedChunk(doc, str(meta.get("source", "")), 1.0 - float(dist), meta))
        return out

    def count(self) -> int:
        """Cached, lock-free: safe to call from a status poll while an upload is embedding."""
        return self._count

    def clear(self) -> None:
        with self._lock:
            self._client.delete_collection(self._collection_name)
            self._col = self._open()
            self._count = 0
