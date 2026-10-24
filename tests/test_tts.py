from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

from biobench_sar.llm import tts_engine
from biobench_sar.llm.tts_engine import TTSEngine


class FakeComm:
    fail = False
    calls = 0
    partial = False

    def __init__(self, text: str, voice: str) -> None:
        self.text, self.voice = text, voice

    async def save(self, path: str) -> None:
        FakeComm.calls += 1
        if FakeComm.partial:
            Path(path).write_bytes(b"")
        if FakeComm.fail:
            raise RuntimeError("offline")
        Path(path).write_bytes(b"mp3")


@pytest.fixture(autouse=True)
def fake_edge(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeComm.fail = False
    FakeComm.calls = 0
    FakeComm.partial = False
    mod = types.ModuleType("edge_tts")
    mod.Communicate = FakeComm  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "edge_tts", mod)


class FakeProc:
    def __init__(self, args: list[str]) -> None:
        self.args = args
        self.terminated = False

    def poll(self) -> None:
        return None

    def terminate(self) -> None:
        self.terminated = True


def test_edge_writes_file(tmp_path: Path) -> None:
    eng = TTSEngine("edge", out_dir=tmp_path / "sub")
    name = eng.speak("hello")
    assert name and name.endswith(".mp3")
    assert (tmp_path / "sub" / name).read_bytes() == b"mp3"


def test_edge_fails_falls_back_to_say(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeComm.fail = True
    procs: list[FakeProc] = []
    monkeypatch.setattr(tts_engine.subprocess, "Popen", lambda a, *k, **kw: procs.append(FakeProc(a)) or procs[-1])
    eng = TTSEngine("edge", out_dir=tmp_path)
    assert eng.speak("hi there") is None
    assert procs[0].args == ["say", "hi there"]
    eng.stop()
    assert procs[0].terminated


def test_say_fails_falls_back_to_pyttsx3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeComm.fail = True
    spoken: list[str] = []
    import threading

    ev = threading.Event()

    class Engine:
        def say(self, t: str) -> None:
            spoken.append(t)

        def runAndWait(self) -> None:
            ev.set()

    mod = types.ModuleType("pyttsx3")
    mod.init = lambda: Engine()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyttsx3", mod)

    def no_say(*a: object, **k: object) -> None:
        raise FileNotFoundError("say")

    monkeypatch.setattr(tts_engine.subprocess, "Popen", no_say)
    eng = TTSEngine("edge", out_dir=tmp_path)
    assert eng.speak("yo") is None
    assert ev.wait(3)
    assert spoken == ["yo"]
    eng.stop()


def test_all_fail_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeComm.fail = True
    monkeypatch.setitem(sys.modules, "pyttsx3", None)  # import raises ImportError

    def no_say(*a: object, **k: object) -> None:
        raise FileNotFoundError("say")

    monkeypatch.setattr(tts_engine.subprocess, "Popen", no_say)
    assert TTSEngine("edge", out_dir=tmp_path).speak("x") is None


def test_none_backend_and_empty_text(tmp_path: Path) -> None:
    assert TTSEngine("none", out_dir=tmp_path).speak("x") is None
    assert TTSEngine("edge", out_dir=tmp_path).speak("  ") is None
    assert list(tmp_path.glob("*.mp3")) == []


def test_cleanup_keeps_newest(tmp_path: Path) -> None:
    for i in range(60):
        f = tmp_path / f"old{i:02d}.mp3"
        f.write_bytes(b"x")
        os.utime(f, (1000 + i, 1000 + i))
    name = TTSEngine("edge", out_dir=tmp_path).speak("new")
    files = {p.name for p in tmp_path.glob("*.mp3")}
    assert len(files) == tts_engine.MAX_FILES
    assert name in files
    assert "old00.mp3" not in files and "old59.mp3" in files


def _no_say(monkeypatch: pytest.MonkeyPatch) -> list[FakeProc]:
    procs: list[FakeProc] = []
    monkeypatch.setattr(tts_engine.subprocess, "Popen", lambda a, *k, **kw: procs.append(FakeProc(a)) or procs[-1])
    return procs


def test_edge_failure_is_remembered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeComm.fail = True
    procs = _no_say(monkeypatch)
    eng = TTSEngine("edge", out_dir=tmp_path)
    eng.speak("one")
    eng.speak("two")
    assert FakeComm.calls == 1
    assert [p.args for p in procs] == [["say", "one"], ["say", "two"]]


def test_edge_retried_after_cooldown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeComm.fail = True
    _no_say(monkeypatch)
    now = {"t": 1000.0}
    monkeypatch.setattr(tts_engine.time, "monotonic", lambda: now["t"])
    eng = TTSEngine("edge", out_dir=tmp_path)
    eng.speak("one")
    now["t"] += tts_engine.EDGE_RETRY_AFTER_S - 1
    eng.speak("two")
    assert FakeComm.calls == 1
    now["t"] += 2
    FakeComm.fail = False
    name = eng.speak("three")
    assert FakeComm.calls == 2 and name and name.endswith(".mp3")
    FakeComm.fail = True  # a success clears the cooldown state
    eng.speak("four")
    assert FakeComm.calls == 3


def test_edge_failure_removes_partial_mp3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeComm.fail = True
    FakeComm.partial = True
    _no_say(monkeypatch)
    TTSEngine("edge", out_dir=tmp_path).speak("x")
    assert FakeComm.calls == 1
    assert list(tmp_path.glob("*.mp3")) == []
