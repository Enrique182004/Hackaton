from __future__ import annotations

import sys
import types
from typing import Any, Callable

import numpy as np
import pytest

from biobench_sar import assistant as assistant_module
from biobench_sar.assistant import HISTORY_SIZE, Assistant, estimate_speech_seconds
from biobench_sar.config import Settings
from biobench_sar.schemas import AvatarAction, RetrievedChunk


@pytest.fixture(autouse=True)
def fake_cv_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    vision = types.ModuleType("biobench_sar.core.vision_loop")
    vision.encode_jpeg = lambda frame, quality=80: b"jpeg"  # type: ignore[attr-defined]
    vision.find_ink_regions = lambda frame, **kw: [{"x": 0}, {"x": 1}]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "biobench_sar.core.vision_loop", vision)


def _chunk(source: str) -> RetrievedChunk:
    return RetrievedChunk(text="t", source=source, score=0.5, metadata={})


class Vision:
    def __init__(self, frame: np.ndarray | None = None) -> None:
        self.frame = frame

    def latest_warped(self) -> np.ndarray | None:
        return self.frame


class Store:
    def __init__(self) -> None:
        self.results: list[RetrievedChunk] = []
        self.error: Exception | None = None

    def query(self, text: str, k: int) -> list[RetrievedChunk]:
        if self.error:
            raise self.error
        return list(self.results)


class VLM:
    def __init__(self) -> None:
        self.error: Exception | None = None
        self.action = AvatarAction("one two three four", "pointing", 10, 20)
        self.on_call: Callable[[], None] | None = None
        self.seen: list[tuple[str, bytes | None, list[Any]]] = []

    def evaluate(self, query: str, image: bytes | None, context: list[Any], ink: list[Any],
                 size: tuple[int, int]) -> AvatarAction:
        self.seen.append((query, image, ink))
        if self.on_call:
            self.on_call()
        if self.error:
            raise self.error
        return self.action


class TTS:
    def __init__(self) -> None:
        self.name: str | None = "a.mp3"
        self.error: Exception | None = None

    def speak(self, text: str) -> str | None:
        if self.error:
            raise self.error
        return self.name


class Harness:
    def __init__(self, deferred: bool = False) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.pending: list[tuple[Callable[..., Any], tuple[Any, ...]]] = []
        self.vision, self.store, self.vlm, self.tts = Vision(), Store(), VLM(), TTS()

        def start_task(fn: Callable[..., Any], *args: Any) -> None:
            if deferred:
                self.pending.append((fn, args))
            else:
                fn(*args)

        self.assistant = Assistant(Settings(), lambda name, data: self.events.append((name, data)), start_task)
        self.assistant.services = types.SimpleNamespace(
            vision=self.vision, store=self.store, vlm=self.vlm, tts=self.tts
        )

    def run_pending(self) -> None:
        fn, args = self.pending.pop(0)
        fn(*args)

    def names(self) -> list[str]:
        return [n for n, _ in self.events]

    def payloads(self, name: str) -> list[dict[str, Any]]:
        return [d for n, d in self.events if n == name]

    def states(self) -> list[str]:
        return [d["state"] for d in self.payloads("status")]


@pytest.fixture
def h() -> Harness:
    return Harness()


@pytest.fixture
def deferred() -> Harness:
    return Harness(deferred=True)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    c = FakeClock()
    monkeypatch.setattr(assistant_module, "time", c)
    return c


# ---- handle_query ----------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", "\n\t", None])
def test_empty_query_rejected_without_side_effects(h: Harness, text: str | None) -> None:
    assert h.assistant.handle_query(text) is False  # type: ignore[arg-type]
    assert h.events == [] and h.vlm.seen == [] and h.assistant.busy is False


def test_accepted_query_returns_true_and_runs(h: Harness) -> None:
    assert h.assistant.handle_query("  hello  ") is True
    assert h.vlm.seen[0][0] == "hello", "query is stripped"
    assert h.payloads("transcript") == [{"text": "hello"}]


def test_busy_rejects_second_query_and_emits_busy_status(deferred: Harness) -> None:
    a = deferred.assistant
    assert a.handle_query("first") is True
    assert a.busy is True
    assert a.handle_query("second") is False
    assert deferred.events == [("status", {"state": "idle", "detail": "busy"})]
    assert len(deferred.pending) == 1, "second query never scheduled"


def test_busy_status_keeps_current_state(deferred: Harness) -> None:
    a = deferred.assistant
    a.handle_query("first")
    a.emit_status("thinking")
    a.handle_query("second")
    assert deferred.events[-1] == ("status", {"state": "thinking", "detail": "busy"})


def test_lock_released_after_run_so_next_query_accepted(deferred: Harness) -> None:
    a = deferred.assistant
    a.handle_query("first")
    deferred.run_pending()
    assert a.busy is False
    assert a.handle_query("second") is True


def test_start_task_failure_releases_lock_and_propagates() -> None:
    def boom(fn: Callable[..., Any], *args: Any) -> None:
        raise RuntimeError("cannot spawn")

    a = Assistant(Settings(), lambda n, d: None, boom)
    with pytest.raises(RuntimeError, match="cannot spawn"):
        a.handle_query("x")
    assert a.busy is False


# ---- pipeline / events -----------------------------------------------------------


def test_event_order_and_final_idle(h: Harness) -> None:
    h.assistant.handle_query("q")
    assert h.names() == ["transcript", "status", "avatar_update", "status", "speak", "interaction", "status"]
    assert h.states() == ["thinking", "speaking", "idle"]
    assert h.payloads("speak") == [{"url": "/tts/a.mp3", "text": "one two three four"}]
    assert h.assistant.state == "idle"


def test_speak_url_none_when_tts_returns_nothing(h: Harness) -> None:
    h.tts.name = None
    h.assistant.handle_query("q")
    assert h.payloads("speak")[0]["url"] is None


def test_frame_present_passes_image_and_ink_to_vlm(h: Harness) -> None:
    h.vision.frame = np.zeros((4, 4, 3), np.uint8)
    h.assistant.handle_query("q")
    _, image, ink = h.vlm.seen[0]
    assert image == b"jpeg" and len(ink) == 2


def test_no_frame_passes_no_image_and_no_ink(h: Harness) -> None:
    h.assistant.handle_query("q")
    _, image, ink = h.vlm.seen[0]
    assert image is None and ink == []


def test_state_is_busy_during_pipeline(h: Harness) -> None:
    seen: list[tuple[bool, str]] = []
    h.vlm.on_call = lambda: seen.append((h.assistant.busy, h.assistant.state))
    h.assistant.handle_query("q")
    assert seen == [(True, "thinking")]


# ---- history ---------------------------------------------------------------------


def test_history_empty_initially(h: Harness) -> None:
    assert h.assistant.history() == []
    assert h.assistant.last_latency_s is None


def test_history_item_fields(h: Harness, clock: FakeClock) -> None:
    h.vision.frame = np.zeros((4, 4, 3), np.uint8)
    h.store.results = [_chunk("a.pdf"), _chunk("b.pdf"), _chunk("a.pdf")]
    h.vlm.on_call = lambda: setattr(clock, "now", clock.now + 1.234)
    h.assistant.handle_query("what?")
    [item] = h.assistant.history()
    assert item["id"] == 1
    assert item["query"] == "what?"
    assert item["spoken_text"] == "one two three four"
    assert item["animation_state"] == "pointing"
    assert (item["target_x"], item["target_y"]) == (10, 20)
    assert item["sources"] == ["a.pdf", "b.pdf"], "de-duplicated, first-seen order"
    assert item["ink_regions"] == 2
    assert item["had_frame"] is True
    assert item["latency_s"] == 1.23
    assert item["error"] is False
    assert item["at"].endswith("+00:00")
    assert h.payloads("interaction") == [item]
    assert h.assistant.last_latency_s == 1.23


def test_history_had_frame_false_without_frame(h: Harness) -> None:
    h.assistant.handle_query("q")
    item = h.assistant.history()[0]
    assert item["had_frame"] is False and item["ink_regions"] == 0 and item["sources"] == []


def test_history_newest_first_with_increasing_ids(h: Harness) -> None:
    for text in ("one", "two", "three"):
        h.assistant.handle_query(text)
    hist = h.assistant.history()
    assert [i["query"] for i in hist] == ["three", "two", "one"]
    assert [i["id"] for i in hist] == [3, 2, 1]


def test_history_capped_at_history_size(h: Harness) -> None:
    total = HISTORY_SIZE + 5
    for n in range(total):
        h.assistant.handle_query(f"q{n}")
    hist = h.assistant.history()
    assert len(hist) == HISTORY_SIZE
    assert hist[0]["query"] == f"q{total - 1}"
    assert hist[-1]["query"] == "q5", "oldest entries dropped"
    assert hist[0]["id"] == total


def test_history_returns_independent_copies(h: Harness) -> None:
    h.assistant.handle_query("q")
    h.assistant.history()[0]["sources"].append("x")
    h.assistant.history().clear()
    assert len(h.assistant.history()) == 1


def test_vlm_error_action_recorded_as_error(h: Harness) -> None:
    h.vlm.action = AvatarAction("bad", "error", 0, 0)
    h.assistant.handle_query("q")
    assert h.assistant.history()[0]["error"] is True


def test_rag_failure_tolerated_with_empty_sources(h: Harness) -> None:
    h.store.error = RuntimeError("down")
    h.assistant.handle_query("q")
    item = h.assistant.history()[0]
    assert item["sources"] == [] and item["error"] is False
    assert "speak" in h.names()


# ---- failures --------------------------------------------------------------------


def test_pipeline_exception_emits_error_and_recovers(h: Harness) -> None:
    h.vlm.error = RuntimeError("boom")
    assert h.assistant.handle_query("q") is True
    [avatar] = h.payloads("avatar_update")
    assert avatar["animation_state"] == "error"
    assert avatar["spoken_text"] == "Sorry, something went wrong. Please try again."
    width, height = Settings().projector_size
    assert (avatar["target_x"], avatar["target_y"]) == (width // 2, height // 2)
    [item] = h.assistant.history()
    assert item["error"] is True and item["query"] == "q"
    assert h.states()[-1] == "idle" and h.assistant.state == "idle"
    assert h.assistant.busy is False
    assert "speak" not in h.names()
    h.vlm.error = None
    assert h.assistant.handle_query("again") is True


def test_tts_failure_is_recorded_as_error_and_lock_released(h: Harness) -> None:
    h.tts.error = RuntimeError("no audio")
    h.assistant.handle_query("q")
    assert h.assistant.history()[0]["error"] is True
    assert h.assistant.busy is False and h.states()[-1] == "idle"


def test_failure_keeps_sources_already_gathered(h: Harness) -> None:
    h.store.results = [_chunk("a.pdf")]
    h.vlm.error = RuntimeError("boom")
    h.assistant.handle_query("q")
    assert h.assistant.history()[0]["sources"] == ["a.pdf"]


def test_failure_does_not_set_speaking_window(h: Harness, clock: FakeClock) -> None:
    h.vlm.error = RuntimeError("boom")
    h.assistant.handle_query("q")
    assert h.assistant.should_listen() is True


# ---- should_listen / on_state ----------------------------------------------------


def test_should_listen_true_when_idle(h: Harness) -> None:
    assert h.assistant.should_listen() is True


def test_should_listen_false_while_busy(deferred: Harness) -> None:
    deferred.assistant.handle_query("q")
    assert deferred.assistant.should_listen() is False


def test_should_listen_false_during_estimated_speaking_window(h: Harness, clock: FakeClock) -> None:
    h.assistant.handle_query("q")  # 4 words -> 4 / 2.5 + 1 = 2.6 s
    window = estimate_speech_seconds("one two three four")
    assert window == pytest.approx(2.6)
    assert h.assistant.busy is False
    assert h.assistant.should_listen() is False
    clock.now += window - 0.01
    assert h.assistant.should_listen() is False
    clock.now += 0.02
    assert h.assistant.should_listen() is True


def test_speaking_window_scales_with_text_length(h: Harness, clock: FakeClock) -> None:
    h.vlm.action = AvatarAction(" ".join(["w"] * 25), "idle", 0, 0)
    h.assistant.handle_query("q")
    clock.now += 10.5
    assert h.assistant.should_listen() is False
    clock.now += 1
    assert h.assistant.should_listen() is True


def test_estimate_speech_seconds_empty_text() -> None:
    assert estimate_speech_seconds("") == 1.0


def test_on_state_idle_ignored_while_busy(deferred: Harness) -> None:
    a = deferred.assistant
    a.handle_query("q")
    a.emit_status("thinking")
    deferred.events.clear()
    a.on_state("idle")
    assert deferred.events == [] and a.state == "thinking"


def test_on_state_non_idle_passes_through_while_busy(deferred: Harness) -> None:
    a = deferred.assistant
    a.handle_query("q")
    a.on_state("listening")
    assert a.state == "listening" and deferred.states() == ["listening"]


def test_on_state_idle_applied_when_not_busy(h: Harness) -> None:
    h.assistant.on_state("listening")
    h.assistant.on_state("idle")
    assert h.states() == ["listening", "idle"] and h.assistant.state == "idle"


def test_emit_status_detail_only_when_given(h: Harness) -> None:
    h.assistant.emit_status("idle")
    h.assistant.emit_status("thinking", "why")
    assert h.payloads("status") == [{"state": "idle"}, {"state": "thinking", "detail": "why"}]
