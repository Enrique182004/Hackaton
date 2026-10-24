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
        self._count = 0
        self._sources: list[dict[str, Any]] = []
        with self._lock:
            self._refresh()

    def _open(self) -> Any:
        return self._client.get_or_create_collection(self._collection_name, metadata={"hnsw:space": "cosine"})

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._embedder is not None:
            return [list(map(float, v)) for v in self._embedder(texts)]
        with self._model_lock:
            if self._st_model is None:
                from sentence_transformers import SentenceTransformer

                try:
                    # Cached copy first: venues are often offline, and the hub check adds seconds per start.
                    self._st_model = SentenceTransformer(self._embed_model, local_files_only=True)
                except Exception:
                    self._st_model = SentenceTransformer(self._embed_model)
            model = self._st_model
        return model.encode(texts, normalize_embeddings=True).tolist()

    @staticmethod
    def _id(chunk: Chunk) -> str:
        # Content-addressed per source: re-uploading the same file (in any batch order) is a no-op.
        return hashlib.sha1(f"{chunk.source}\x00{chunk.text}".encode()).hexdigest()[:16]

    def _unique(self, chunks: list[Chunk]) -> dict[str, Chunk]:
        unique: dict[str, Chunk] = {}
        for c in chunks:
            if c.text.strip():
                unique.setdefault(self._id(c), c)
        return unique

    def _upsert(self, unique: dict[str, Chunk], embeddings: list[list[float]]) -> None:
        self._col.upsert(
            ids=list(unique),
            documents=[c.text for c in unique.values()],
            embeddings=embeddings,
            metadatas=[{**c.metadata, "source": c.source} for c in unique.values()],
        )

    def _source_ids(self, source: str) -> list[str]:
        return list(self._col.get(where={"source": source}, include=[])["ids"])

    def _refresh(self) -> None:
        """Recompute the cached count and per-file summary. Caller holds the lock."""
        self._count = int(self._col.count())
        metadatas = self._col.get(include=["metadatas"])["metadatas"] or []
        by_source: dict[str, dict[str, Any]] = {}
        for meta in metadatas:
            meta = meta or {}
            source = str(meta.get("source", ""))
            entry = by_source.setdefault(source, {"source": source, "type": str(meta.get("type", "")), "chunks": 0})
            entry["chunks"] += 1
        self._sources = sorted(by_source.values(), key=lambda d: (-d["chunks"], d["source"]))

    def add_chunks(self, chunks: list[Chunk]) -> int:
        unique = self._unique(chunks)
        if not unique:
            return 0
        embeddings = self._embed([c.text for c in unique.values()])
        with self._lock:
            self._upsert(unique, embeddings)
            self._refresh()
        return len(unique)

    def replace_source(self, source: str, chunks: list[Chunk]) -> tuple[int, int]:
        """Make `chunks` the only ones stored for `source`. Returns (newly added, stale removed).

        Ids are content hashes, so plain add_chunks() on an edited file would keep serving its old text.
        """
        if any(c.source != source for c in chunks):
            raise ValueError(f"every chunk must have source {source!r}")
        unique = self._unique(chunks)
        embeddings = self._embed([c.text for c in unique.values()]) if unique else []
        with self._lock:
            existing = set(self._source_ids(source))
            if unique:
                self._upsert(unique, embeddings)
            stale = list(existing - unique.keys())
            if stale:
                self._col.delete(ids=stale)
            self._refresh()
        return len(unique.keys() - existing), len(stale)

    def source_count(self, source: str) -> int:
        with self._lock:
            return len(self._source_ids(source))

    def query(self, text: str, k: int = 4, sources: list[str] | None = None) -> list[RetrievedChunk]:
        """Top-k chunks for `text`. `sources=None` searches every file; a list searches only those files."""
        if sources is not None and not sources:
            return []
        if self._count == 0 or not text.strip() or k <= 0:
            return []
        where: dict[str, Any] | None = None
        names: set[str] = set()
        if sources is not None:
            names = set(sources)
            ordered = sorted(names)
            where = {"source": ordered[0]} if len(ordered) == 1 else {"source": {"$in": ordered}}
        emb = self._embed([text])
        with self._lock:
            # Count under the same lock as the query: some chroma versions raise if n_results > matches.
            n = self._count if where is None else sum(d["chunks"] for d in self._sources if d["source"] in names)
            if n == 0:
                return []
            res = self._col.query(
                query_embeddings=emb,
                n_results=min(k, n),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        out: list[RetrievedChunk] = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0], strict=False):
            meta = dict(meta or {})
            out.append(RetrievedChunk(doc, str(meta.get("source", "")), 1.0 - float(dist), meta))
        return out

    def count(self) -> int:
        """Cached, lock-free: safe to call from a status poll while an upload is embedding."""
        return self._count

    def list_sources(self) -> list[dict[str, Any]]:
        """One entry per ingested file: name, type and chunk count, largest first.

        Cached and lock-free like count(): /api/status reads it on every poll.
        """
        return [dict(d) for d in self._sources]

    def delete_source(self, source: str) -> int:
        with self._lock:
            ids = self._col.get(where={"source": source})["ids"]
            if ids:
                self._col.delete(ids=ids)
            self._refresh()
        return len(ids)

    def clear(self) -> None:
        with self._lock:
            self._client.delete_collection(self._collection_name)
            self._col = self._open()
            self._count = 0
            self._sources = []
