from __future__ import annotations

import dataclasses
import socket
from pathlib import Path
from typing import Any

from biobench_sar.app import _port_free, seed_app_knowledge, seed_knowledge
from biobench_sar.config import PACKAGE_DIR, Settings
from biobench_sar.rag.quiz_bank import QuizBank

BIOLOGY = PACKAGE_DIR / "knowledge" / "biology"


class Store:
    def __init__(self, existing: int = 0) -> None:
        self.chunks: list[Any] = []
        self.existing = existing

    def count(self) -> int:
        return self.existing + len(self.chunks)

    def add_chunks(self, chunks: list[Any]) -> int:
        self.chunks.extend(chunks)
        return len(chunks)


def _math_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "math"
    folder.mkdir()
    (folder / "quiz_bank.csv").write_text(
        "problem,answer,topic,level\n3x + 5 = 20,x = 5,equations,2\n1/2 + 1/4,3/4,fractions,1\n"
    )
    (folder / "tutoring_guide.md").write_text("# Tutoring guide\n\n1. Ask what 3x = 12 means before solving.\n")
    (folder / ".gitkeep").write_text("")
    return folder


def test_seed_fills_an_empty_store_from_bundled_docs() -> None:
    store = Store()
    added = seed_knowledge(store, BIOLOGY)
    assert added > 0 and added == len(store.chunks)
    assert {c.source for c in store.chunks} >= {"acanthamoeba_isolation_protocol.md", "laguna_bustillos_site.md"}


def test_biology_docs_live_in_their_profile_folder() -> None:
    assert Settings(profile="biology").knowledge_dir == BIOLOGY
    assert {p.name for p in BIOLOGY.iterdir()} >= {
        "acanthamoeba_isolation_protocol.md",
        "laguna_bustillos_site.md",
        "sampling_sites.csv",
    }
    assert Settings(profile="math").knowledge_dir.is_dir()
    assert not any(p.suffix for p in (PACKAGE_DIR / "knowledge").iterdir() if p.is_file()), "no loose seed docs"


def test_seed_without_marker_leaves_a_populated_store_alone() -> None:
    store = Store(existing=3)
    assert seed_knowledge(store, BIOLOGY) == 0
    assert store.chunks == []


def test_seed_skips_unsupported_and_missing(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("Autoclave media before plating.")
    (tmp_path / "binary.exe").write_bytes(b"MZ")
    store = Store()
    assert seed_knowledge(store, tmp_path) == 1
    assert seed_knowledge(Store(), tmp_path / "missing") == 0


def test_seed_runs_once_per_profile_with_a_marker(tmp_path: Path) -> None:
    marker = tmp_path / "chroma" / ".seeded"
    math = _math_dir(tmp_path)
    store = Store()
    assert seed_knowledge(store, math, marker=marker) > 0
    assert marker.read_text().split() == ["math"]
    store.chunks.clear()  # the student removed every file
    assert seed_knowledge(store, math, marker=marker) == 0, "demo docs are not seeded again after a restart"
    assert store.chunks == []
    assert seed_knowledge(Store(existing=5), BIOLOGY, marker=marker) > 0, "another profile still gets its docs"
    assert sorted(marker.read_text().split()) == ["biology", "math"]


def test_seed_writes_no_marker_when_nothing_was_added(tmp_path: Path) -> None:
    marker = tmp_path / ".seeded"
    empty = tmp_path / "math"
    empty.mkdir()
    (empty / ".gitkeep").write_text("")
    assert seed_knowledge(Store(), empty, marker=marker) == 0
    assert not marker.exists(), "retried next start, e.g. once the docs arrive"


def test_seed_fills_the_quiz_bank_and_reports_names(tmp_path: Path) -> None:
    bank = QuizBank(tmp_path / "quiz_bank.json")
    seen: list[list[str]] = []
    seed_knowledge(Store(), _math_dir(tmp_path), quiz_bank=bank, on_done=seen.append)
    assert seen == [["quiz_bank.csv", "tutoring_guide.md"]]
    assert [p.problem for p in bank.problems()] == ["1/2 + 1/4", "3x + 5 = 20"], "level order"
    assert bank.sources() == ["quiz_bank.csv"], "the tutoring guide's numbered tip is not a quiz problem"


def test_seed_app_knowledge_pushes_documents_changed(
    settings: Settings, fakes: Any, templates_dir: Path, tmp_path: Path
) -> None:
    from biobench_sar import app as app_module

    flask_app, socketio = app_module.create_app(settings, fakes, template_folder=str(templates_dir))
    assistant = flask_app.config["SAR_ASSISTANT"]
    assistant.selection.select(["mine.pdf"])
    sio = socketio.test_client(flask_app)
    sio.get_received()
    added = seed_app_knowledge(flask_app, settings, _math_dir(tmp_path))
    assert added == 2
    got = sio.get_received()
    pushed = [m["args"][0] for m in got if m["name"] == "documents_changed"]
    assert len(pushed) == 1 and {d["source"] for d in pushed[0]["documents"]} == {
        "quiz_bank.csv",
        "tutoring_guide.md",
    }
    assert assistant.active_sources == {"mine.pdf", "quiz_bank.csv", "tutoring_guide.md"}, "seeded files are ticked"
    assert settings.seed_marker_file.read_text().split() == ["math"]
    assert seed_app_knowledge(flask_app, settings, tmp_path / "math") == 0


def test_seed_app_knowledge_uses_the_profile_folder(settings: Settings, fakes: Any, templates_dir: Path) -> None:
    from biobench_sar import app as app_module

    flask_app, _ = app_module.create_app(
        dataclasses.replace(settings, profile="biology"), fakes, template_folder=str(templates_dir)
    )
    assert seed_app_knowledge(flask_app, dataclasses.replace(settings, profile="biology")) > 0
    assert "laguna_bustillos_site.md" in {c.source for c in fakes.store.chunks}


def test_port_free_detects_a_bound_port() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        port = sock.getsockname()[1]
        assert _port_free("127.0.0.1", port) is False


def test_default_port_avoids_airplay(monkeypatch: Any) -> None:
    monkeypatch.delenv("SAR_PORT", raising=False)
    assert Settings().port == 5050
