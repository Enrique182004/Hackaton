from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Callable

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

    def add_chunks(self, chunks: list[Any]) -> int:
        self.chunks.extend(chunks)
        return len(chunks)

    def query(self, text: str, k: int = 4) -> list[Any]:
        if self.fail_query:
            raise RuntimeError("store down")
        return []

    def count(self) -> int:
        return len(self.chunks)

    def clear(self) -> None:
        self.chunks.clear()


class FakeVLM:
    def __init__(self) -> None:
        self.error: Exception | None = None
        self.gate: Any = None  # threading.Event to hold the pipeline open
        self.calls: list[str] = []

    def evaluate(self, query: str, image_jpeg: bytes | None, context: list[Any],
                 ink_regions: list[Any], projector_size: tuple[int, int]) -> AvatarAction:
        self.calls.append(query)
        if self.gate is not None:
            self.gate.wait(5)
        if self.error is not None:
            raise self.error
        return AvatarAction("Look here", "pointing", 100, 200)


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


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return dataclasses.replace(
        Settings(),
        upload_dir=tmp_path / "uploads",
        tts_dir=tmp_path / "tts",
        chroma_dir=tmp_path / "chroma",
        calibration_file=tmp_path / "calibration.json",
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
    return app_module.Services(
        vision=FakeVision(), store=FakeStore(), vlm=FakeVLM(), audio=FakeAudio(), tts=FakeTTS()
    )


@pytest.fixture
def built(settings: Settings, fakes: app_module.Services, templates_dir: Path) -> Any:
    flask_app, socketio = app_module.create_app(settings, fakes, template_folder=str(templates_dir))
    fakes.audio.on_query = flask_app.config["SAR_HANDLE_QUERY"]
    flask_app.testing = True
    return flask_app, socketio
