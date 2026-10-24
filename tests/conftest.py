from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from biobench_sar import app as app_module
from biobench_sar.config import Settings
from biobench_sar.schemas import AvatarAction


class FakeVision:
    def __init__(self) -> None:
        self.frame: np.ndarray | None = None
        self.fps = 30.0
        self.is_running = True
        self.calibration: Any = None

    def latest_frame(self) -> np.ndarray | None:
        return None if self.frame is None else self.frame.copy()

    def latest_warped(self) -> np.ndarray | None:
        return self.latest_frame()

    def set_calibration(self, calibration: Any) -> None:
        self.calibration = calibration


class FakeStore:
    def __init__(self) -> None:
        self.chunks: list[Any] = []
        self.fail_query = False
        self.query_results: list[Any] = []
        self.query_calls: list[tuple[str, int]] = []
        self.query_sources: list[list[str] | None] = []  # the `sources` filter of each query call
        self.sources: list[dict[str, Any]] = []  # listed documents that have no fake chunks behind them

    def add_chunks(self, chunks: list[Any]) -> int:
        self.chunks.extend(chunks)
        return len(chunks)

    def replace_source(self, source: str, chunks: list[Any]) -> tuple[int, int]:
        old = {c.text for c in self.chunks if c.source == source}
        new = {c.text: c for c in chunks}
        self.chunks = [c for c in self.chunks if c.source != source] + list(new.values())
        return len(new.keys() - old), len(old - new.keys())

    def source_count(self, source: str) -> int:
        return sum(1 for c in self.chunks if c.source == source)

    def query(self, text: str, k: int = 4, sources: list[str] | None = None) -> list[Any]:
        if self.fail_query:
            raise RuntimeError("store down")
        self.query_calls.append((text, k))
        self.query_sources.append(None if sources is None else list(sources))
        if sources is None:
            return list(self.query_results)
        return [r for r in self.query_results if r.source in sources]

    def list_sources(self) -> list[dict[str, Any]]:
        by_source: dict[str, dict[str, Any]] = {}
        for c in self.chunks:
            entry = by_source.setdefault(
                c.source, {"source": c.source, "type": str(c.metadata.get("type", "")), "chunks": 0}
            )
            entry["chunks"] += 1
        return [dict(d) for d in self.sources] + sorted(by_source.values(), key=lambda d: (-d["chunks"], d["source"]))

    def delete_source(self, source: str) -> int:
        matched = [d for d in self.sources if d["source"] == source]
        self.sources = [d for d in self.sources if d["source"] != source]
        removed = sum(1 for c in self.chunks if c.source == source)
        self.chunks = [c for c in self.chunks if c.source != source]
        return sum(d["chunks"] for d in matched) + removed

    def count(self) -> int:
        return len(self.chunks)

    def clear(self) -> None:
        self.chunks.clear()


class FakeVLM:
    def __init__(self) -> None:
        self.error: Exception | None = None
        self.gate: Any = None  # threading.Event to hold the pipeline open
        self.calls: list[str] = []
        self.kwargs: list[dict[str, Any]] = []  # mode / task / recent_turns of each call
        self.action = AvatarAction("Look here", "pointing", 100, 200)

    def evaluate(
        self,
        query: str,
        image_jpeg: bytes | None,
        context: list[Any],
        ink_regions: list[Any],
        projector_size: tuple[int, int],
        **kwargs: Any,
    ) -> AvatarAction:
        self.calls.append(query)
        self.kwargs.append(kwargs)
        if self.gate is not None:
            self.gate.wait(5)
        if self.error is not None:
            raise self.error
        return self.action


class FakeAudio:
    available = True

    def __init__(self) -> None:
        self.on_query: Callable[[str], None] | None = None
        self.triggered = 0

    def start(self) -> None: ...
    def stop(self) -> None: ...

    def trigger(self) -> None:
        self.triggered += 1

    def submit_text(self, text: str) -> None:
        assert self.on_query is not None
        self.on_query(text)


class FakeTTS:
    def speak(self, text: str) -> str | None:
        return "fake.mp3"


@pytest.fixture(autouse=True)
def isolated_selection_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every Settings() in tests reads/writes a throwaway selection file, never the developer's data/."""
    path = tmp_path / "selection.json"
    monkeypatch.setenv("SAR_SELECTION_FILE", str(path))
    return path


@pytest.fixture(autouse=True)
def isolated_quiz_bank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Same for the quiz bank; and the default (math) profile, whatever the developer's shell exports."""
    path = tmp_path / "quiz_bank.json"
    monkeypatch.setenv("SAR_QUIZ_BANK_FILE", str(path))
    monkeypatch.delenv("SAR_PROFILE", raising=False)
    return path


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return dataclasses.replace(
        Settings(),
        upload_dir=tmp_path / "uploads",
        tts_dir=tmp_path / "tts",
        chroma_dir=tmp_path / "chroma",
        calibration_file=tmp_path / "calibration.json",
        selection_file=tmp_path / "selection.json",
        quiz_bank_file=tmp_path / "quiz_bank.json",
        projector_width=640,
        projector_height=360,
    )


@pytest.fixture
def templates_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "templates"
    folder.mkdir()
    (folder / "index.html").write_text("<html>index {{ projector_width }}x{{ projector_height }}</html>")
    (folder / "precharge.html").write_text("<html>precharge {{ projector_width }}</html>")
    return folder


@pytest.fixture
def fakes() -> app_module.Services:
    return app_module.Services(vision=FakeVision(), store=FakeStore(), vlm=FakeVLM(), audio=FakeAudio(), tts=FakeTTS())


@pytest.fixture
def built(settings: Settings, fakes: app_module.Services, templates_dir: Path) -> Any:
    flask_app, socketio = app_module.create_app(settings, fakes, template_folder=str(templates_dir))
    fakes.audio.on_query = flask_app.config["SAR_HANDLE_QUERY"]
    flask_app.testing = True
    return flask_app, socketio
