from __future__ import annotations

import sys
import threading
import time
import types

import numpy as np
import pytest

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


def test_wake_push_keeps_speech_from_the_interrupted_chunk() -> None:
    chunk = np.full(48000, 0.2, dtype=np.float32)
    push = np.full(16000, -0.3, dtype=np.float32)
    chunk_started = threading.Event()
    release_chunk = threading.Event()
    states: list[str] = []
    heard: list[np.ndarray] = []
    queries: list[str] = []
    done = threading.Event()

    def rec(s: float) -> np.ndarray:
        if s == 3.0 and not chunk_started.is_set():
            chunk_started.set()
            release_chunk.wait(3)
            return chunk
        return push if s == 5.0 else SILENCE

    t = AudioTrigger(
        lambda q: (queries.append(q), done.set()),
        mode="wake",
        record_seconds=5.0,
        wake_chunk_seconds=3.0,
        on_state=states.append,
        recorder=rec,
        transcriber=lambda a: (heard.append(a), "push question")[1],
    )
    t.start()
    _wait(chunk_started)
    assert t.trigger() is True  # wake loop is mid-chunk
    assert states == ["listening"]  # announced on accept, not after the chunk ends
    time.sleep(0.1)  # the user starts talking right after the click
    release_chunk.set()
    _wait(done)
    t.stop()
    assert queries == ["push question"]
    assert states == ["listening", "transcribing", "idle"]
    assert len(heard) == 1  # the interrupted chunk was not wake-word checked on its own
    lead = heard[0][: heard[0].size - push.size]
    assert 0 < lead.size < chunk.size and np.all(lead == 0.2)
    assert np.array_equal(heard[0][lead.size :], push)


def test_since_request_keeps_only_the_post_click_tail() -> None:
    t = AudioTrigger(lambda q: None, recorder=lambda s: AUDIO, transcriber=lambda a: "")
    chunk = np.arange(48000, dtype=np.float32)
    t._push_requested_at = time.monotonic() - 1.0
    tail = t._since_request(chunk)
    assert 24000 <= tail.size < 24000 + 3200  # 1 s since the click + 0.5 s pre-roll
    assert tail[-1] == chunk[-1]
    t._push_requested_at = time.monotonic() - 10.0
    assert t._since_request(chunk).size == chunk.size


def test_push_during_wake_followup_does_not_flash_idle() -> None:
    states: list[str] = []
    queries: list[str] = []
    texts = iter(["assistant", "explain", "push q"])
    followups = {"n": 0}

    def rec(s: float) -> np.ndarray:
        if s == 5.0:
            followups["n"] += 1
            if followups["n"] == 1:
                assert t.trigger() is True  # clicked Listen during the wake follow-up
        return AUDIO

    t = AudioTrigger(
        queries.append,
        mode="wake",
        on_state=states.append,
        recorder=rec,
        transcriber=lambda a: next(texts, ""),
    )
    t.start()
    assert _until(lambda: len(queries) == 2)
    t.stop()
    assert queries == ["explain", "push q"]
    assert states[:4] == ["listening", "listening", "transcribing", "idle"]


def test_wake_loop_exit_with_pending_push_reports_idle() -> None:
    states: list[str] = []

    def paused() -> bool:
        if not t._stop.is_set():
            assert t.trigger() is True
            t._stop.set()
        return False

    t = AudioTrigger(
        lambda q: None,
        mode="wake",
        on_state=states.append,
        recorder=lambda s: AUDIO,
        transcriber=lambda a: "",
        should_listen=paused,
    )
    t.start()
    assert _until(lambda: t._wake_thread is not None and not t._wake_thread.is_alive())
    assert states == ["listening", "idle"]
    assert t._push_state == "idle"


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


QUIET = np.full(1600, 0.005, dtype=np.float32)  # soft speech: under the wake gate, far above digital silence


def _wake_on_quiet(**kwargs: float) -> tuple[list[str], list[int]]:
    queries: list[str] = []
    seen: list[int] = []
    t = AudioTrigger(
        queries.append,
        mode="wake",
        recorder=lambda s: (time.sleep(0.005), QUIET)[1],
        transcriber=lambda a: (seen.append(1), "assistant what is this")[1],
        **kwargs,
    )
    t.start()
    _until(lambda: bool(queries), timeout=0.3)
    t.stop()
    return queries, seen


def test_wake_chunks_keep_the_default_silence_gate() -> None:
    queries, seen = _wake_on_quiet()
    assert queries == [] and seen == []


def test_silence_rms_is_configurable() -> None:
    queries, _ = _wake_on_quiet(silence_rms=0.001)
    assert queries and queries[0] == "what is this"


def _push_once(audio: np.ndarray) -> tuple[list[str], list[int]]:
    queries: list[str] = []
    seen: list[int] = []
    states: list[str] = []
    t = AudioTrigger(
        queries.append,
        on_state=states.append,
        recorder=lambda s: audio,
        transcriber=lambda a: (seen.append(1), "soft question")[1],
    )
    t.trigger()
    # Transcription happens before "idle" but delivery after it, so a transcribed clip waits for its query.
    assert _until(lambda: bool(queries) or (states[-1:] == ["idle"] and not seen))
    return queries, seen


def test_push_to_talk_hears_quiet_speech() -> None:
    queries, seen = _push_once(QUIET)
    assert queries == ["soft question"] and seen == [1]


def test_push_to_talk_skips_digital_silence() -> None:
    queries, seen = _push_once(np.full(1600, 1e-5, dtype=np.float32))
    assert queries == [] and seen == []


def test_wake_followup_hears_quiet_speech() -> None:
    queries: list[str] = []
    texts = iter(["assistant", "explain the diagram"])
    t = AudioTrigger(
        queries.append,
        mode="wake",
        recorder=lambda s: QUIET if s == 5.0 else AUDIO,
        transcriber=lambda a: next(texts, ""),
    )
    t.start()
    assert _until(lambda: bool(queries))
    t.stop()
    assert queries == ["explain the diagram"]


class _ReentryDetector:
    """Stands in for a Whisper model: records how many decodes overlap."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls = 0

    def __call__(self, audio: np.ndarray) -> str:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.03)
        with self._lock:
            self.active -= 1
            self.calls += 1
        return " ok "

    def transcribe(self, audio: np.ndarray, **kwargs: object) -> dict[str, str]:
        return {"text": self(audio)}


def _transcribe_concurrently(t: AudioTrigger, n: int = 6) -> list[str]:
    barrier = threading.Barrier(n)
    out: list[str] = []

    def work() -> None:
        barrier.wait(3)
        out.append(t.transcribe(AUDIO))

    threads = [threading.Thread(target=work) for _ in range(n)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(5)
    return out


def test_whisper_decodes_never_overlap() -> None:
    model = _ReentryDetector()
    t = AudioTrigger(lambda q: None, recorder=lambda s: AUDIO)
    t._model = model  # skip loading real Whisper
    assert _transcribe_concurrently(t) == ["ok"] * 6
    assert model.calls == 6 and model.max_active == 1


def test_injected_transcriber_decodes_never_overlap() -> None:
    fake = _ReentryDetector()
    t = AudioTrigger(lambda q: None, recorder=lambda s: AUDIO, transcriber=fake)
    assert _transcribe_concurrently(t) == ["ok"] * 6
    assert fake.calls == 6 and fake.max_active == 1


class _FakeWhisper:
    def __init__(self, text: str = " hola ", language: str | None = "es") -> None:
        self.text, self.language = text, language
        self.kwargs: list[dict[str, object]] = []

    def transcribe(self, audio: np.ndarray, **kwargs: object) -> dict[str, object]:
        self.kwargs.append(kwargs)
        out: dict[str, object] = {"text": self.text}
        if self.language is not None:
            out["language"] = self.language
        return out


def _with_model(model: _FakeWhisper, language: str = "auto") -> AudioTrigger:
    t = AudioTrigger(lambda q: None, recorder=lambda s: AUDIO, language=language)
    t._model = model
    return t


def _fake_lang(monkeypatch: pytest.MonkeyPatch, result: str = "es") -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []
    mod = types.ModuleType("biobench_sar.lang")

    def detect_language(text: str, default: str = "en") -> str:
        calls.append((text, default))
        return result

    mod.detect_language = detect_language  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "biobench_sar.lang", mod)
    return calls


def test_language_defaults_to_auto_and_english() -> None:
    t = AudioTrigger(lambda q: None, recorder=lambda s: AUDIO, transcriber=lambda a: "")
    assert t.language == "auto" and t.last_language == "en"


def test_auto_lets_whisper_detect_the_language() -> None:
    model = _FakeWhisper(" ¿y ahora? ", "es")
    t = _with_model(model)
    assert t.transcribe(AUDIO) == "¿y ahora?"
    assert model.kwargs == [{"fp16": False, "language": None}]
    assert t.last_language == "es"


def test_forced_language_is_passed_to_whisper() -> None:
    model = _FakeWhisper(" hello ", "en")
    t = _with_model(model, language="es")
    assert t.last_language == "es"
    t.transcribe(AUDIO)
    assert t.last_language == "es"
    assert t.set_language("en") == "en"
    t.transcribe(AUDIO)
    assert [k["language"] for k in model.kwargs] == ["es", "en"]
    assert all(k["fp16"] is False for k in model.kwargs)
    assert t.last_language == "en"


def test_set_language_normalizes_codes() -> None:
    t = AudioTrigger(lambda q: None, recorder=lambda s: AUDIO, transcriber=lambda a: "")
    assert t.set_language("ES_mx") == "es" and t.last_language == "es"
    assert t.set_language("English") == "en"
    assert t.set_language("Español") == "es"
    assert t.set_language("fr") == "auto"
    assert t.set_language(None) == "auto" and t.language == "auto"
    assert t.last_language == "es"  # auto keeps following the last language heard


def test_unexpected_whisper_language_falls_back_to_text_heuristic(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_lang(monkeypatch, "es")
    t = _with_model(_FakeWhisper(" ayuda ", "pt"))
    t.transcribe(AUDIO)
    assert t.last_language == "es"
    assert calls == [("ayuda", "en")]


def test_heuristic_without_lang_module_falls_back_to_english(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "biobench_sar.lang", None)  # import raises ImportError
    t = _with_model(_FakeWhisper(" ayuda ", "gl"))
    t.set_language("es")
    t.set_language("auto")
    t.transcribe(AUDIO)
    assert t.last_language == "en"


def test_heuristic_error_keeps_the_last_language(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = types.ModuleType("biobench_sar.lang")

    def boom(text: str, default: str = "en") -> str:
        raise RuntimeError("x")

    mod.detect_language = boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "biobench_sar.lang", mod)
    t = _with_model(_FakeWhisper(" x = 5 ", None))
    t.set_language("es")
    t.set_language("auto")
    t.transcribe(AUDIO)
    assert t.last_language == "es"


def test_empty_transcript_keeps_the_last_language() -> None:
    t = _with_model(_FakeWhisper("  ", "es"))
    t.transcribe(np.zeros(16000, dtype=np.float32))  # warm-up on silence
    assert t.last_language == "en"


def test_injected_transcriber_language_comes_from_text(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_lang(monkeypatch, "es")
    t = AudioTrigger(lambda q: None, recorder=lambda s: AUDIO, transcriber=lambda a: " no entiendo ")
    assert t.transcribe(AUDIO) == "no entiendo"
    assert t.last_language == "es" and calls == [("no entiendo", "en")]


def test_typed_text_updates_last_language(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_lang(monkeypatch, "es")
    q: list[str] = []
    t = AudioTrigger(q.append, recorder=lambda s: AUDIO, transcriber=lambda a: "")
    t.submit_text(" ¿cómo despejo x? ")
    assert q == ["¿cómo despejo x?"] and t.last_language == "es"
    t.set_language("en")
    t.submit_text("ayuda")
    assert t.last_language == "en"  # a forced language wins over the heuristic


def test_push_query_reports_its_language(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_lang(monkeypatch, "es")
    seen: list[tuple[str, str]] = []
    done = threading.Event()
    t = AudioTrigger(
        lambda q: (seen.append((q, t.last_language)), done.set()),
        recorder=lambda s: AUDIO,
        transcriber=lambda a: "el siguiente paso",
    )
    t.trigger()
    _wait(done)
    assert seen == [("el siguiente paso", "es")]
