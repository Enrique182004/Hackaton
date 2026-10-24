from __future__ import annotations

import threading
import time

import numpy as np

from biobench_sar.core.audio_trigger import AudioTrigger

AUDIO = (np.random.default_rng(0).standard_normal(1600) * 0.1).astype(np.float32)
SILENCE = np.zeros(1600, dtype=np.float32)


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


def test_wake_paused_while_not_listening() -> None:
    calls: list[float] = []
    queries: list[str] = []
    allowed = threading.Event()
    t = AudioTrigger(
        queries.append,
        mode="wake",
        wake_chunk_seconds=3.0,
        recorder=lambda s: (calls.append(s), AUDIO)[1],
        transcriber=lambda a: "assistant what is this",
        should_listen=allowed.is_set,
    )
    t.start()
    threading.Event().wait(0.5)
    assert calls == [] and queries == []
    allowed.set()
    deadline = time.monotonic() + 3
    while not queries and time.monotonic() < deadline:
        threading.Event().wait(0.05)
    t.stop()
    assert queries and queries[0] == "what is this"


def test_wake_word_heard_while_avatar_speaks_is_ignored() -> None:
    state = {"listening": True}
    queries: list[str] = []

    def rec(s: float) -> np.ndarray:
        state["listening"] = False  # avatar starts talking mid-chunk
        return AUDIO

    t = AudioTrigger(
        queries.append,
        mode="wake",
        wake_chunk_seconds=3.0,
        recorder=rec,
        transcriber=lambda a: "assistant explain",
        should_listen=lambda: state["listening"],
    )
    t.start()
    threading.Event().wait(0.6)
    t.stop()
    assert queries == []


def _until(cond, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


def test_wake_followup_empty_emits_idle() -> None:
    states: list[str] = []
    calls: list[float] = []
    it = iter(["assistant", ""])
    t = AudioTrigger(
        lambda q: None,
        mode="wake",
        on_state=states.append,
        recorder=lambda s: (calls.append(s), AUDIO)[1],
        transcriber=lambda a: next(it, ""),
    )
    t.start()
    assert _until(lambda: states[-1:] == ["idle"])
    t.stop()
    assert states[:2] == ["listening", "idle"]


def test_wake_followup_exception_emits_idle() -> None:
    states: list[str] = []
    n = {"i": 0}

    def tr(a: np.ndarray) -> str:
        n["i"] += 1
        if n["i"] == 1:
            return "assistant"
        raise RuntimeError("whisper died")

    t = AudioTrigger(lambda q: None, mode="wake", on_state=states.append, recorder=lambda s: AUDIO, transcriber=tr)
    t.start()
    assert _until(lambda: states[-1:] == ["idle"])
    t.stop()
    assert states[:2] == ["listening", "idle"]


def test_push_trigger_returns_bool() -> None:
    gate = threading.Event()
    done = threading.Event()

    def rec(s: float) -> np.ndarray:
        gate.wait(3)
        return AUDIO

    t = AudioTrigger(lambda q: done.set(), recorder=rec, transcriber=lambda a: "hi")
    assert t.trigger() is True
    assert _until(lambda: t._push_state == "recording")
    assert t.trigger() is False
    gate.set()
    _wait(done)
    assert _until(lambda: t._push_state == "idle")
    gate.clear()
    gate.set()
    assert t.trigger() is True


def test_wake_push_wins_over_background_chunk() -> None:
    chunk_started = threading.Event()
    release_chunk = threading.Event()
    calls: list[float] = []
    queries: list[str] = []
    done = threading.Event()

    def rec(s: float) -> np.ndarray:
        calls.append(s)
        if s == 3.0 and not chunk_started.is_set():
            chunk_started.set()
            release_chunk.wait(3)
        return AUDIO

    texts = {3.0: "assistant ignored background", 5.0: "push question"}
    last = {"s": 0.0}

    def rec2(s: float) -> np.ndarray:
        last["s"] = s
        return rec(s)

    t = AudioTrigger(
        lambda q: (queries.append(q), done.set()),
        mode="wake",
        record_seconds=5.0,
        wake_chunk_seconds=3.0,
        recorder=rec2,
        transcriber=lambda a: texts[last["s"]],
    )
    t.start()
    _wait(chunk_started)
    assert t.trigger() is True  # wake loop is mid-chunk
    release_chunk.set()
    _wait(done)
    t.stop()
    assert queries[0] == "push question"  # background chunk was dropped, push recorded right after
    assert 5.0 in calls


def test_wake_push_while_push_recording_returns_false() -> None:
    in_push = threading.Event()
    gate = threading.Event()

    def rec(s: float) -> np.ndarray:
        if s == 5.0:
            in_push.set()
            gate.wait(3)
        return AUDIO

    t = AudioTrigger(
        lambda q: None, mode="wake", record_seconds=5.0, recorder=rec, transcriber=lambda a: "nothing here"
    )
    t.start()
    assert t.trigger() is True
    _wait(in_push)
    assert t.trigger() is False
    gate.set()
    t.stop()


def test_silence_guard_skips_transcription() -> None:
    transcribed: list[int] = []
    chunks = {"n": 0}

    def rec(s: float) -> np.ndarray:
        chunks["n"] += 1
        return SILENCE if chunks["n"] <= 2 else AUDIO

    def tr(a: np.ndarray) -> str:
        transcribed.append(1)
        return "assistant what is this"

    queries: list[str] = []
    seen_at_query: list[int] = []
    t = AudioTrigger(
        lambda q: (queries.append(q), seen_at_query.append(len(transcribed))), mode="wake", recorder=rec, transcriber=tr
    )
    t.start()
    assert _until(lambda: bool(queries))
    t.stop()
    assert seen_at_query[0] == 1  # the two silent chunks never reached Whisper
    assert chunks["n"] >= 3


def test_silence_rms_is_configurable() -> None:
    quiet = np.full(1600, 0.005, dtype=np.float32)
    seen: list[int] = []
    t = AudioTrigger(lambda q: None, transcriber=lambda a: (seen.append(1), "x")[1], silence_rms=0.001)
    assert t._heard(quiet) == "x"
    t2 = AudioTrigger(lambda q: None, transcriber=lambda a: (seen.append(1), "x")[1])
    assert t2._heard(quiet) == "" and len(seen) == 1
