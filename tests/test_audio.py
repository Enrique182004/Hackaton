from __future__ import annotations

import threading

import numpy as np

from biobench_sar.core.audio_trigger import AudioTrigger

AUDIO = np.zeros(1600, dtype=np.float32)


def _wait(ev: threading.Event) -> None:
    assert ev.wait(3), "timed out"


def test_push_sequence() -> None:
    states: list[str] = []
    queries: list[str] = []
    done = threading.Event()
    t = AudioTrigger(
        lambda q: (queries.append(q), done.set()),
        on_state=states.append,
        recorder=lambda s: AUDIO,
        transcriber=lambda a: " what is this? ",
    )
    t.trigger()
    _wait(done)
    assert queries == ["what is this?"]
    assert states == ["listening", "transcribing", "idle"]


def test_empty_transcript_no_query() -> None:
    states: list[str] = []
    queries: list[str] = []
    t = AudioTrigger(queries.append, on_state=states.append, recorder=lambda s: AUDIO, transcriber=lambda a: "  ")
    t.trigger()
    for _ in range(300):
        if states[-1:] == ["idle"]:
            break
        threading.Event().wait(0.01)
    assert states == ["listening", "transcribing", "idle"]
    assert queries == []


def test_busy_ignored() -> None:
    gate = threading.Event()
    calls: list[float] = []
    done = threading.Event()

    def rec(s: float) -> np.ndarray:
        calls.append(s)
        gate.wait(3)
        return AUDIO

    t = AudioTrigger(lambda q: done.set(), recorder=rec, transcriber=lambda a: "hi", record_seconds=2.0)
    t.trigger()
    t.trigger()
    gate.set()
    _wait(done)
    assert calls == [2.0]


def test_exception_reports_idle() -> None:
    states: list[str] = []
    queries: list[str] = []

    def boom(a: np.ndarray) -> str:
        raise RuntimeError("x")

    t = AudioTrigger(queries.append, on_state=states.append, recorder=lambda s: AUDIO, transcriber=boom)
    t.trigger()
    for _ in range(300):
        if states[-1:] == ["idle"]:
            break
        threading.Event().wait(0.01)
    assert states[-1] == "idle" and queries == []
    t.trigger()  # busy flag released


def test_submit_text() -> None:
    q: list[str] = []
    t = AudioTrigger(q.append, recorder=lambda s: AUDIO, transcriber=lambda a: "")
    t.submit_text("  hello ")
    t.submit_text("   ")
    assert q == ["hello"]


def test_extract_after_wake() -> None:
    t = AudioTrigger(lambda q: None, wake_word="assistant", recorder=lambda s: AUDIO, transcriber=lambda a: "")
    assert t.extract_after_wake("Hey ASSISTANT, explain this.") == "explain this"
    assert t.extract_after_wake("assistants are great") is None
    assert t.extract_after_wake("assistant") == ""


def _wake(texts: list[str], record_calls: list[float]) -> tuple[AudioTrigger, list[str], threading.Event]:
    queries: list[str] = []
    done = threading.Event()
    it = iter(texts)

    def rec(s: float) -> np.ndarray:
        record_calls.append(s)
        threading.Event().wait(0.01)
        return AUDIO

    t = AudioTrigger(
        lambda q: (queries.append(q), done.set()),
        mode="wake",
        record_seconds=5.0,
        wake_chunk_seconds=3.0,
        recorder=rec,
        transcriber=lambda a: next(it, ""),
    )
    return t, queries, done


def test_wake_inline_query() -> None:
    calls: list[float] = []
    t, queries, done = _wake(["nothing", "Assistant what is mitosis"], calls)
    t.start()
    _wait(done)
    t.stop()
    assert queries == ["what is mitosis"]
    assert set(calls) == {3.0}


def test_wake_followup_recording() -> None:
    calls: list[float] = []
    t, queries, done = _wake(["assistant", "explain the diagram"], calls)
    t.start()
    _wait(done)
    t.stop()
    assert queries == ["explain the diagram"]
    assert calls[:2] == [3.0, 5.0]


def test_push_mode_start_is_noop() -> None:
    t = AudioTrigger(lambda q: None, mode="push", recorder=lambda s: AUDIO, transcriber=lambda a: "")
    t.start()
    assert t._wake_thread is None
    t.stop()
