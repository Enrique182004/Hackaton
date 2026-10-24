from __future__ import annotations

import hashlib
import os
import re
import shutil
import threading
from pathlib import Path
from typing import Any

import pytest

from biobench_sar.rag.ingestor import (
    SUPPORTED_EXTENSIONS,
    Chunk,
    UnsupportedFileType,
    chunk_paragraphs,
    ingest_file,
)
from biobench_sar.rag.vector_store import VectorStore

DIM = 256
KNOWLEDGE = Path(__file__).resolve().parent.parent / "biobench_sar" / "knowledge" / "biology"


def fake_embed(texts: list[str]) -> list[list[float]]:
    out = []
    for t in texts:
        v = [0.0] * DIM
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        out.append([x / norm for x in v])
    return out


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "chroma", embedder=fake_embed)


def test_chunk_paragraphs_merge_and_split() -> None:
    assert chunk_paragraphs("a\n\nb\n\n\nc", 100) == ["a\n\nb\n\nc"]
    assert chunk_paragraphs("a" * 30 + "\n\n" + "b" * 30, 40) == ["a" * 30, "b" * 30]
    long = chunk_paragraphs("word " * 100, 50)
    assert len(long) > 5 and all(len(c) <= 50 for c in long)
    assert chunk_paragraphs("  \n\n  ") == []


def test_supported_extensions() -> None:
    assert {".pdf", ".txt", ".md", ".csv", ".docx", ".pptx", ".png", ".jpg", ".jpeg"} == set(SUPPORTED_EXTENSIONS)


def test_unsupported(tmp_path: Path) -> None:
    f = tmp_path / "x.exe"
    f.write_text("hi")
    with pytest.raises(UnsupportedFileType):
        ingest_file(f)


def test_txt_md(tmp_path: Path) -> None:
    f = tmp_path / "n.md"
    f.write_text("# Title\n\nHello amoeba.", encoding="utf-8")
    chunks = ingest_file(f)
    assert chunks and chunks[0].source == "n.md" and chunks[0].metadata["type"] == "md"
    (tmp_path / "e.txt").write_text("  \n")
    assert ingest_file(tmp_path / "e.txt") == []


def test_csv(tmp_path: Path) -> None:
    f = tmp_path / "s.csv"
    f.write_text("site,soil\nA,Leptosol\nB,Feozem\n")
    chunks = ingest_file(f)
    joined = " ".join(c.text for c in chunks)
    assert "Row 1: site is A, soil is Leptosol." in joined
    assert "Row 2: site is B, soil is Feozem." in joined
    assert chunks[0].metadata["type"] == "csv" and "row" in chunks[0].metadata


def test_docx(tmp_path: Path) -> None:
    import docx

    d = docx.Document()
    d.add_paragraph("First paragraph about sterile technique.")
    d.add_paragraph("Second paragraph.")
    f = tmp_path / "a.docx"
    d.save(str(f))
    chunks = ingest_file(f)
    assert "sterile technique" in chunks[0].text and chunks[0].metadata["type"] == "docx"


def test_pptx(tmp_path: Path) -> None:
    from pptx import Presentation

    prs = Presentation()
    for title in ("Slide one text", "Slide two text"):
        s = prs.slides.add_slide(prs.slide_layouts[5])
        s.shapes.title.text = title
    f = tmp_path / "a.pptx"
    prs.save(str(f))
    chunks = ingest_file(f)
    assert [c.metadata["slide"] for c in chunks] == [1, 2]
    assert "Slide two" in chunks[1].text


def test_pdf(tmp_path: Path) -> None:
    import pymupdf

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Acanthamoeba cysts resist drying.")
    doc.new_page().insert_text((72, 72), "Second page content.")
    f = tmp_path / "a.pdf"
    doc.save(str(f))
    chunks = ingest_file(f)
    assert chunks[0].metadata["page"] == 1 and "Acanthamoeba" in chunks[0].text
    assert chunks[1].metadata["page"] == 2


@pytest.mark.skipif(
    shutil.which("tesseract") is None and not Path("/opt/homebrew/bin/tesseract").exists(), reason="tesseract missing"
)
def test_image_ocr(tmp_path: Path) -> None:
    import pytesseract
    from PIL import Image, ImageDraw, ImageFont

    if not shutil.which("tesseract"):
        pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
    img = Image.new("RGB", (900, 160), "white")
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 64)
    except OSError:
        font = ImageFont.load_default(size=64)
    ImageDraw.Draw(img).text((20, 40), "AUTOCLAVE STEP", fill="black", font=font)
    f = tmp_path / "t.png"
    img.save(f)
    chunks = ingest_file(f)
    assert chunks and "autoclave" in chunks[0].text.lower()


def test_store_add_query_idempotent(store: VectorStore) -> None:
    chunks = [
        Chunk("Leptosol runoff raises turbidity", "a.md", {"type": "md"}),
        Chunk("Feozem soils are humus rich and fertile", "b.md", {"type": "md"}),
        Chunk("Autoclave all plates before disposal", "c.md", {"type": "md"}),
    ]
    assert store.add_chunks(chunks) == 3
    store.add_chunks(chunks)
    assert store.count() == 3
    res = store.query("turbidity from Leptosol runoff", k=2)
    assert len(res) == 2 and res[0].source == "a.md"
    assert res[0].score >= res[1].score and 0.0 < res[0].score <= 1.0
    assert res[0].metadata["type"] == "md"
    store.clear()
    assert store.count() == 0 and store.query("x") == []


def test_empty_and_persist(tmp_path: Path) -> None:
    s = VectorStore(tmp_path / "c", embedder=fake_embed)
    assert s.query("anything") == [] and s.add_chunks([]) == 0
    s.add_chunks([Chunk("persisted text", "p.txt", {})])
    s2 = VectorStore(tmp_path / "c", embedder=fake_embed)
    assert s2.count() == 1


def test_thread_safety(store: VectorStore) -> None:
    store.add_chunks([Chunk(f"doc {i} amoeba", "t.txt", {}) for i in range(20)])
    errors: list[Exception] = []

    def work() -> None:
        try:
            for _ in range(10):
                assert store.query("amoeba", k=3)
        except Exception as e:
            errors.append(e)

    ts = [threading.Thread(target=work) for _ in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors


def test_seed_knowledge_ingests_and_retrieves(store: VectorStore) -> None:
    total = 0
    for f in KNOWLEDGE.iterdir():
        if f.suffix.lower() in SUPPORTED_EXTENSIONS:
            total += store.add_chunks(ingest_file(f))
    assert total > 0
    res = store.query("Leptosol Feozem runoff turbidity", k=3)
    assert any("Leptosol" in r.text for r in res)
    res = store.query("non-nutrient agar heat-killed E. coli incubation", k=3)
    assert any("agar" in r.text.lower() for r in res)


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("SAR_SLOW") != "1", reason="set SAR_SLOW=1")
def test_real_embedder(tmp_path: Path) -> None:
    s = VectorStore(tmp_path / "c")
    s.add_chunks([Chunk("Amoebae eat bacteria", "a", {}), Chunk("Stock market rises", "b", {})])
    assert s.query("protozoa feeding on microbes", k=1)[0].source == "a"


def test_count_does_not_wait_for_model_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import time
    import types

    loading = threading.Event()

    class SlowModel:
        def __init__(self, name: str) -> None:
            loading.set()
            time.sleep(1.0)

        def encode(self, texts: list[str], normalize_embeddings: bool = True) -> object:
            import numpy as np

            return np.array(fake_embed(texts))

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=SlowModel))
    s = VectorStore(tmp_path / "c")
    worker = threading.Thread(target=s.add_chunks, args=([Chunk("alpha beta", "a.txt", {})],))
    worker.start()
    assert loading.wait(2)
    started = time.perf_counter()
    assert s.count() == 0
    assert time.perf_counter() - started < 0.1
    worker.join()
    assert s.count() == 1
    s.clear()
    assert s.count() == 0


def test_list_and_delete_sources(store: VectorStore) -> None:
    store.add_chunks([Chunk("one", "a.md", {"type": "md"}), Chunk("two", "a.md", {"type": "md"})])
    store.add_chunks([Chunk("three", "b.csv", {"type": "csv"})])
    assert store.list_sources() == [
        {"source": "a.md", "type": "md", "chunks": 2},
        {"source": "b.csv", "type": "csv", "chunks": 1},
    ]
    assert store.delete_source("a.md") == 2
    assert store.count() == 1
    assert store.delete_source("missing.md") == 0


def test_reupload_in_different_order_does_not_duplicate(store: VectorStore) -> None:
    chunks = [Chunk("first", "a.md", {}), Chunk("second", "a.md", {}), Chunk("first", "a.md", {})]
    store.add_chunks(chunks)
    store.add_chunks(list(reversed(chunks)))
    assert store.count() == 2


def _texts(store: VectorStore, source: str) -> set[str]:
    return {r.text for r in store.query("amoeba cyst agar", k=50) if r.source == source}


def test_replace_source_drops_stale_chunks_of_edited_file(store: VectorStore) -> None:
    other = Chunk("amoeba agar in another file", "b.md", {})
    store.add_chunks([other])
    v1 = [Chunk("amoeba cyst stage one", "a.md", {}), Chunk("amoeba agar old step", "a.md", {})]
    assert store.replace_source("a.md", v1) == (2, 0)
    v2 = [Chunk("amoeba cyst stage one", "a.md", {}), Chunk("amoeba agar new step", "a.md", {})]
    assert store.replace_source("a.md", v2) == (1, 1)
    assert _texts(store, "a.md") == {"amoeba cyst stage one", "amoeba agar new step"}
    assert store.source_count("a.md") == 2 and store.count() == 3
    assert _texts(store, "b.md") == {other.text}, "other sources untouched"


def test_replace_source_unchanged_reupload_is_a_no_op(store: VectorStore) -> None:
    chunks = [Chunk("amoeba one", "a.md", {}), Chunk("amoeba two", "a.md", {}), Chunk("amoeba one", "a.md", {})]
    assert store.replace_source("a.md", chunks) == (2, 0)
    assert store.replace_source("a.md", list(reversed(chunks))) == (0, 0)
    assert store.count() == 2 and store.source_count("a.md") == 2


def test_replace_source_with_no_chunks_removes_the_file(store: VectorStore) -> None:
    store.replace_source("a.md", [Chunk("amoeba one", "a.md", {}), Chunk("amoeba two", "a.md", {})])
    assert store.replace_source("a.md", [Chunk("  ", "a.md", {})]) == (0, 2)
    assert store.count() == 0 and store.source_count("a.md") == 0


def test_replace_source_rejects_chunks_of_another_source(store: VectorStore) -> None:
    with pytest.raises(ValueError, match=r"'a\.md'"):
        store.replace_source("a.md", [Chunk("x", "b.md", {})])
    assert store.count() == 0


# ---- per-file filtering (document selection) ----------------------------------------


class QuerySpy:
    """Wraps the chroma collection to record what query() asks it for."""

    def __init__(self, col: Any) -> None:
        self._col = col
        self.queries: list[dict[str, Any]] = []

    def query(self, **kwargs: Any) -> Any:
        self.queries.append(kwargs)
        return self._col.query(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._col, name)


@pytest.fixture
def three_files(store: VectorStore) -> VectorStore:
    store.add_chunks([Chunk(f"amoeba cyst note {i}", "a.md", {"type": "md"}) for i in range(3)])
    store.add_chunks([Chunk(f"amoeba agar plate {i}", "b.md", {"type": "md"}) for i in range(2)])
    store.add_chunks([Chunk("amoeba turbidity runoff", "c.csv", {"type": "csv"})])
    return store


def test_query_sources_none_searches_every_file(three_files: VectorStore) -> None:
    res = three_files.query("amoeba", k=10, sources=None)
    assert len(res) == 6 and {r.source for r in res} == {"a.md", "b.md", "c.csv"}


def test_query_single_source_only_returns_that_file(three_files: VectorStore) -> None:
    res = three_files.query("amoeba agar plate", k=4, sources=["b.md"])
    assert [r.source for r in res] == ["b.md", "b.md"]


def test_query_several_sources(three_files: VectorStore) -> None:
    res = three_files.query("amoeba", k=10, sources=["c.csv", "a.md", "a.md"])
    assert len(res) == 4 and {r.source for r in res} == {"a.md", "c.csv"}
    assert all(res[i].score >= res[i + 1].score for i in range(len(res) - 1))


def test_query_empty_source_list_returns_nothing_without_querying(tmp_path: Path) -> None:
    embedded: list[list[str]] = []

    def counting(texts: list[str]) -> list[list[float]]:
        embedded.append(texts)
        return fake_embed(texts)

    s = VectorStore(tmp_path / "c", embedder=counting)
    s.add_chunks([Chunk("amoeba", "a.md", {})])
    spy = QuerySpy(s._col)
    s._col = spy
    embedded.clear()
    assert s.query("amoeba", k=4, sources=[]) == []
    assert spy.queries == [] and embedded == [], "no embedding, no chroma query"


def test_query_n_results_clamped_to_selected_chunks(three_files: VectorStore) -> None:
    spy = QuerySpy(three_files._col)
    three_files._col = spy
    res = three_files.query("amoeba", k=10, sources=["b.md"])
    assert len(res) == 2
    assert spy.queries[-1]["n_results"] == 2, "never ask chroma for more than the selected files hold"
    assert spy.queries[-1]["where"] == {"source": "b.md"}
    three_files.query("amoeba", k=10, sources=["a.md", "b.md"])
    assert spy.queries[-1]["n_results"] == 5
    assert spy.queries[-1]["where"] == {"source": {"$in": ["a.md", "b.md"]}}
    three_files.query("amoeba", k=2, sources=["a.md", "b.md"])
    assert spy.queries[-1]["n_results"] == 2
    three_files.query("amoeba", k=10)
    assert spy.queries[-1]["n_results"] == 6 and spy.queries[-1]["where"] is None


def test_query_unknown_source_returns_nothing(three_files: VectorStore) -> None:
    spy = QuerySpy(three_files._col)
    three_files._col = spy
    assert three_files.query("amoeba", k=4, sources=["gone.md"]) == []
    assert spy.queries == [], "zero matching chunks: chroma is not asked"
    assert {r.source for r in three_files.query("amoeba", k=10, sources=["gone.md", "c.csv"])} == {"c.csv"}


def test_query_after_deleting_a_selected_source(three_files: VectorStore) -> None:
    three_files.delete_source("b.md")
    assert three_files.query("amoeba agar", k=4, sources=["b.md"]) == []
    assert [r.source for r in three_files.query("amoeba", k=4, sources=["b.md", "c.csv"])] == ["c.csv"]


def test_list_sources_tracks_every_write(three_files: VectorStore, tmp_path: Path) -> None:
    def summary(s: VectorStore) -> list[tuple[str, str, int]]:
        return [(d["source"], d["type"], d["chunks"]) for d in s.list_sources()]

    assert summary(three_files) == [("a.md", "md", 3), ("b.md", "md", 2), ("c.csv", "csv", 1)]
    three_files.replace_source("b.md", [Chunk("amoeba only one", "b.md", {"type": "md"})])
    assert summary(three_files) == [("a.md", "md", 3), ("b.md", "md", 1), ("c.csv", "csv", 1)]
    three_files.delete_source("a.md")
    assert summary(three_files) == [("b.md", "md", 1), ("c.csv", "csv", 1)]
    assert summary(VectorStore(tmp_path / "chroma", embedder=fake_embed)) == summary(three_files), "loaded at start"
    three_files.clear()
    assert three_files.list_sources() == []


def test_list_sources_returns_copies(three_files: VectorStore) -> None:
    three_files.list_sources()[0]["chunks"] = 99
    assert three_files.list_sources()[0]["chunks"] == 3


def test_list_sources_does_not_wait_for_the_store_lock(three_files: VectorStore) -> None:
    import time

    held, release = threading.Event(), threading.Event()

    def hold() -> None:
        with three_files._lock:
            held.set()
            release.wait(2)

    worker = threading.Thread(target=hold)
    worker.start()
    try:
        assert held.wait(2)
        started = time.perf_counter()
        assert len(three_files.list_sources()) == 3
        assert time.perf_counter() - started < 0.1, "status polls must not block behind an upload"
    finally:
        release.set()
        worker.join()
