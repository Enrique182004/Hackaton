from __future__ import annotations

import json
import os
import sys
import threading
import time
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from biobench_sar import assistant as assistant_module
from biobench_sar.assistant import HISTORY_SIZE, Assistant, DocumentSelection, estimate_speech_seconds
from biobench_sar.config import Settings
from biobench_sar.rag.quiz_bank import QuizProblem
from biobench_sar.schemas import AvatarAction, RetrievedChunk
from biobench_sar.tutor import DEFAULT_QUIZ_LENGTH


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
        self.calls: list[tuple[str, int, list[str] | None]] = []

    def query(self, text: str, k: int, sources: list[str] | None = None) -> list[RetrievedChunk]:
        self.calls.append((text, k, sources))
        if self.error:
            raise self.error
        return [r for r in self.results if sources is None or r.source in sources]


class VLM:
    def __init__(self) -> None:
        self.error: Exception | None = None
        self.action = AvatarAction("one two three four", "pointing", 10, 20)
        self.on_call: Callable[[], None] | None = None
        self.seen: list[tuple[str, bytes | None, list[Any]]] = []
        self.contexts: list[list[Any]] = []
        self.kwargs: list[dict[str, Any]] = []

    def evaluate(
        self, query: str, image: bytes | None, context: list[Any], ink: list[Any], size: tuple[int, int], **kwargs: Any
    ) -> AvatarAction:
        self.seen.append((query, image, ink))
        self.contexts.append(list(context))
        self.kwargs.append(kwargs)
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


def test_history_readable_while_pipeline_appends(h: Harness) -> None:
    for n in range(HISTORY_SIZE):
        h.assistant.handle_query(f"q{n}")
    stop = threading.Event()

    def pipeline() -> None:
        while not stop.is_set():
            h.assistant.handle_query("more")

    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # switch threads mid-iteration as often as possible
    writer = threading.Thread(target=pipeline)
    writer.start()
    try:
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            assert len(h.assistant.history()) == HISTORY_SIZE
            assert h.assistant.last_latency_s is not None
    finally:
        stop.set()
        writer.join()
        sys.setswitchinterval(interval)


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


def test_pointing_kept_when_there_is_ink(h: Harness) -> None:
    h.vision.frame = np.zeros((4, 4, 3), np.uint8)
    h.assistant.handle_query("q")
    assert h.payloads("avatar_update")[0]["animation_state"] == "pointing"


def test_pointing_without_ink_becomes_idle(h: Harness) -> None:
    h.assistant.handle_query("q")
    assert h.payloads("avatar_update")[0]["animation_state"] == "idle"
    assert h.assistant.history()[0]["animation_state"] == "idle"


def test_ink_thresholds_come_from_settings(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    vision = sys.modules["biobench_sar.core.vision_loop"]
    monkeypatch.setattr(vision, "find_ink_regions", lambda frame, **kw: seen.update(kw) or [])
    h.vision.frame = np.zeros((4, 4, 3), np.uint8)
    h.assistant.handle_query("q")
    assert seen == Settings().ink_kwargs


# ---- document selection ----------------------------------------------------------


@pytest.fixture
def sel_path(tmp_path: Path) -> Path:
    return tmp_path / "state" / "selection.json"


def _tmp_leftovers(path: Path) -> list[str]:
    return [p.name for p in path.parent.iterdir() if p.name != path.name] if path.parent.exists() else []


def test_selection_missing_file_means_all_active(sel_path: Path) -> None:
    sel = DocumentSelection(sel_path)
    assert sel.active is None and sel.query_sources() is None
    assert not sel_path.exists(), "loading never writes"


@pytest.mark.parametrize(
    "content",
    ["{not json", "", "[]", '"a.md"', "null", '{"sources": "a.md"}', '{"all": false}', '{"other": 1}'],
)
def test_selection_corrupt_file_means_all_active(sel_path: Path, content: str) -> None:
    sel_path.parent.mkdir(parents=True)
    sel_path.write_text(content)
    assert DocumentSelection(sel_path).active is None


def test_selection_loads_all_and_sources(sel_path: Path) -> None:
    sel_path.parent.mkdir(parents=True)
    sel_path.write_text('{"all": true}')
    assert DocumentSelection(sel_path).active is None
    sel_path.write_text('{"sources": ["b.md", "a.md", 3, null]}')
    sel = DocumentSelection(sel_path)
    assert sel.active == {"a.md", "b.md"}, "non-string names dropped"
    assert sel.query_sources() == ["a.md", "b.md"]
    sel_path.write_text('{"sources": []}')
    assert DocumentSelection(sel_path).active == set()


def test_selection_select_and_select_all_persist(sel_path: Path) -> None:
    sel = DocumentSelection(sel_path)
    sel.select(["b.md", "a.md", "a.md"])
    assert json.loads(sel_path.read_text()) == {"sources": ["a.md", "b.md"]}
    assert DocumentSelection(sel_path).active == {"a.md", "b.md"}, "survives a restart"
    sel.select([])
    assert json.loads(sel_path.read_text()) == {"sources": []}
    assert DocumentSelection(sel_path).active == set()
    sel.select_all()
    assert json.loads(sel_path.read_text()) == {"all": True}
    assert DocumentSelection(sel_path).active is None
    assert _tmp_leftovers(sel_path) == [], "temp files are renamed into place"


def test_selection_active_is_a_copy(sel_path: Path) -> None:
    sel = DocumentSelection(sel_path)
    sel.select(["a.md"])
    sel.active.add("b.md")  # type: ignore[union-attr]
    assert sel.active == {"a.md"}


def test_selection_activate_only_extends_an_explicit_selection(sel_path: Path) -> None:
    sel = DocumentSelection(sel_path)
    sel.activate(["new.md"])
    assert sel.active is None and not sel_path.exists(), "all active already includes new files"
    sel.select(["a.md"])
    sel.activate(["new.md", "a.md"])
    assert sel.active == {"a.md", "new.md"}
    assert json.loads(sel_path.read_text()) == {"sources": ["a.md", "new.md"]}


def test_selection_discard(sel_path: Path) -> None:
    sel = DocumentSelection(sel_path)
    sel.discard("a.md")
    assert sel.active is None and not sel_path.exists()
    sel.select(["a.md", "b.md"])
    sel.discard("a.md")
    sel.discard("missing.md")
    assert sel.active == {"b.md"}
    assert json.loads(sel_path.read_text()) == {"sources": ["b.md"]}


def test_selection_write_is_atomic(sel_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sel = DocumentSelection(sel_path)
    sel.select(["a.md"])
    before = sel_path.read_text()

    def fail(src: Any, dst: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", fail)
    sel.select(["b.md"])
    assert sel_path.read_text() == before, "a failed write never leaves a half-written file"
    assert _tmp_leftovers(sel_path) == [], "the temp file is cleaned up"
    assert sel.active == {"b.md"}, "the new selection still applies until restart"


def test_selection_concurrent_updates_stay_consistent(sel_path: Path) -> None:
    sel = DocumentSelection(sel_path)
    sel.select([])
    names = [f"f{i}.md" for i in range(40)]

    def worker(chunk: list[str]) -> None:
        for name in chunk:
            sel.activate([name])
            sel.discard(name)
            sel.activate([name])

    threads = [threading.Thread(target=worker, args=(names[i::4],)) for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sel.active == set(names)
    assert json.loads(sel_path.read_text()) == {"sources": sorted(names)}
    assert DocumentSelection(sel_path).active == set(names)


def test_assistant_loads_selection_from_settings_at_startup(isolated_selection_file: Path) -> None:
    isolated_selection_file.write_text('{"sources": ["a.md"]}')
    assert Settings().selection_file == isolated_selection_file
    assert Harness().assistant.active_sources == {"a.md"}


def test_assistant_defaults_to_all_active(h: Harness) -> None:
    assert h.assistant.active_sources is None


def test_pipeline_searches_everything_when_all_active(h: Harness) -> None:
    h.store.results = [_chunk("a.md"), _chunk("b.md")]
    h.assistant.handle_query("q")
    assert h.store.calls == [("q", Settings().rag_top_k, None)]
    assert h.assistant.history()[0]["sources"] == ["a.md", "b.md"]


def test_pipeline_passes_sorted_selection_to_store(h: Harness) -> None:
    h.store.results = [_chunk("a.md"), _chunk("b.md"), _chunk("c.md")]
    h.assistant.selection.select(["c.md", "a.md"])
    h.assistant.handle_query("q")
    assert h.store.calls == [("q", Settings().rag_top_k, ["a.md", "c.md"])]
    assert [c.source for c in h.vlm.contexts[0]] == ["a.md", "c.md"]
    assert h.assistant.history()[0]["sources"] == ["a.md", "c.md"]


def test_pipeline_skips_rag_when_selection_is_empty(h: Harness) -> None:
    h.store.results = [_chunk("a.md")]
    h.assistant.selection.select([])
    h.assistant.handle_query("q")
    assert h.store.calls == [], "store never queried"
    assert h.vlm.contexts == [[]]
    item = h.assistant.history()[0]
    assert item["sources"] == [] and item["error"] is False
    assert "speak" in h.names()


def test_pipeline_follows_selection_changes_between_queries(h: Harness) -> None:
    h.assistant.selection.select(["a.md"])
    h.assistant.handle_query("one")
    h.assistant.selection.select_all()
    h.assistant.handle_query("two")
    assert [c[2] for c in h.store.calls] == [["a.md"], None]


def test_celebrating_without_ink_becomes_idle(h: Harness) -> None:
    h.vlm.action = AvatarAction("All correct!", "celebrating", 10, 20)
    h.assistant.handle_query("is this right?")
    assert h.payloads("avatar_update")[0]["animation_state"] == "idle"


def test_error_is_kept_without_ink(h: Harness) -> None:
    h.vlm.action = AvatarAction("That step is unsafe.", "error", 10, 20)
    h.assistant.handle_query("can I skip the autoclave?")
    assert h.payloads("avatar_update")[0]["animation_state"] == "error"


# ---- tutor: conversation memory, lessons, practice, quiz ---------------------------


def _ask(h: Harness, text: str, state: str = "idle", spoken: str = "ok", ink: bool = False) -> dict[str, Any]:
    """Run one question; returns the kwargs the model got (mode / task / recent_turns)."""
    h.vlm.action = AvatarAction(spoken, state, 10, 20)  # type: ignore[arg-type]
    h.vision.frame = np.zeros((4, 4, 3), np.uint8) if ink else None
    h.assistant.handle_query(text)
    return h.vlm.kwargs[-1]


def _tutor_events(h: Harness) -> list[dict[str, Any]]:
    return h.payloads("tutor")


def _bank(h: Harness, source: str, *problems: tuple[str, str, int]) -> None:
    h.assistant.quiz_bank.replace_source(
        source, [QuizProblem(p, a, "equations", level, source) for p, a, level in problems]
    )


def test_model_gets_help_mode_and_conversation_so_far(h: Harness) -> None:
    first = _ask(h, "what is this?", spoken="A fraction.")
    assert first == {"mode": "help", "task": None, "recent_turns": []}
    second = _ask(h, "and this?")
    assert second["recent_turns"] == [{"q": "what is this?", "a": "A fraction."}]
    for n in range(4):
        _ask(h, f"q{n}", spoken=f"a{n}")
    assert [t["q"] for t in h.vlm.kwargs[-1]["recent_turns"]] == ["q0", "q1", "q2"], "the last three, oldest first"


def test_failed_query_is_not_remembered(h: Harness) -> None:
    h.vlm.error = RuntimeError("boom")
    h.assistant.handle_query("q")
    assert h.assistant.tutor.turns() == []


def test_plain_question_pushes_no_tutor_event(h: Harness) -> None:
    _ask(h, "what is this?")
    assert _tutor_events(h) == []


def test_learn_request_on_an_empty_board_offers_an_exercise(h: Harness) -> None:
    kwargs = _ask(h, "I want to learn how to solve fractions", state="pointing")
    assert kwargs["task"] == {"kind": "learn", "topic": "solving fractions", "stage": "offer"}
    assert h.assistant.tutor.offer_pending
    assert h.payloads("avatar_update")[-1]["animation_state"] == "pointing", "lessons are never downgraded"
    assert _tutor_events(h)[-1] == {"mode": "help", "learn_topic": "solving fractions", "quiz": None}
    assert h.assistant.history()[0]["task_kind"] == "learn"


def test_offer_reply_make_one_up_dictates_an_example(h: Harness) -> None:
    _ask(h, "teach me equations", spoken="An equation says two sides are equal. Should I make one up?")
    kwargs = _ask(h, "make one up")
    assert kwargs["task"] == {"kind": "learn", "topic": "equations", "stage": "example"}
    assert kwargs["recent_turns"][-1]["q"] == "teach me equations"
    assert not h.assistant.tutor.offer_pending
    assert _ask(h, "make one up")["task"] is None, "the offer was already answered"


def test_offer_reply_own_exercise(h: Harness) -> None:
    _ask(h, "teach me fractions")
    assert _ask(h, "I have one")["task"] == {"kind": "learn", "topic": "fractions", "stage": "own"}


def test_offer_reply_only_counts_while_an_offer_is_pending(h: Harness) -> None:
    assert _ask(h, "I have one")["task"] is None


def test_unrelated_reply_keeps_the_offer_open(h: Harness) -> None:
    _ask(h, "teach me fractions")
    assert _ask(h, "what is this?")["task"] is None
    assert h.assistant.tutor.offer_pending


def test_practice_mode_sticks_until_help(h: Harness) -> None:
    assert _ask(h, "give me a similar problem", ink=True)["task"] == {"kind": "practice"}
    assert h.assistant.tutor.mode == "practice"
    assert _tutor_events(h)[-1]["mode"] == "practice"
    kwargs = _ask(h, "and now?")
    assert kwargs["task"] == {"kind": "practice"} and kwargs["mode"] == "practice"
    assert h.assistant.history()[0]["mode"] == "practice"
    assert _ask(h, "I'm stuck")["task"] is None
    assert h.assistant.tutor.mode == "help"


def test_quiz_dictates_then_checks_problems_from_ticked_files(h: Harness) -> None:
    _bank(h, "a.md", ("3x + 5 = 20", "x = 5", 2), ("x + 1 = 3", "x = 2", 1))
    _bank(h, "b.md", ("9 - 4", "5", 1))
    h.assistant.selection.select(["a.md"])
    kwargs = _ask(h, "quiz me")
    assert kwargs["mode"] == "quiz"
    assert kwargs["task"] == {
        "kind": "quiz",
        "stage": "dictate",
        "number": 1,
        "total": 2,
        "problem": "x + 1 = 3",
        "answer": "x = 2",
        "next_problem": None,
    }
    assert h.assistant.tutor.dictated
    assert _tutor_events(h)[0]["quiz"] == {"active": True, "number": 1, "total": 2, "score": 0, "problem": "x + 1 = 3"}

    check = _ask(h, "is this right?", state="celebrating", spoken="Yes! Next: 3x + 5 = 20.", ink=True)["task"]
    assert check["stage"] == "check" and check["next_problem"] == "3x + 5 = 20"
    assert h.payloads("avatar_update")[-1]["animation_state"] == "celebrating", "a correct ink answer is kept"
    tutor = h.assistant.tutor
    assert (tutor.quiz_number, tutor.score, tutor.dictated) == (2, 1, True), "next problem was read out with it"

    wrong = _ask(h, "is this right?", state="error", spoken="Not quite. Subtract 5 first.", ink=True)["task"]
    assert wrong["stage"] == "check" and wrong["number"] == 2 and wrong["next_problem"] is None
    assert (tutor.quiz_number, tutor.score) == (2, 1)

    _ask(h, "now?", state="celebrating", spoken="Correct!", ink=True)
    assert not tutor.quiz_active and tutor.mode == "help"
    assert h.payloads("speak")[-1]["text"] == "Correct! That was the last problem! You solved 1 of 2 on the first try."
    assert _tutor_events(h)[-1] == {
        "mode": "help",
        "learn_topic": None,
        "quiz": {"active": False, "number": 2, "total": 2, "score": 1, "problem": None},
    }
    assert _ask(h, "thanks")["task"] is None, "back to help after the quiz"


def test_quiz_summary_does_not_repeat_an_announced_end(h: Harness) -> None:
    _bank(h, "a.md", ("1 + 1", "2", 1))
    _ask(h, "quiz me")
    _ask(h, "done", state="celebrating", spoken="Correct! You finished the quiz!", ink=True)
    assert h.payloads("speak")[-1]["text"] == "Correct! You finished the quiz! You solved 1 of 1 on the first try."


def test_quiz_without_worksheet_problems_lets_the_model_invent_them(h: Harness) -> None:
    task = _ask(h, "start a quiz")["task"]
    assert task["problem"] is None and task["answer"] is None and task["total"] == DEFAULT_QUIZ_LENGTH
    check = _ask(h, "done", state="celebrating", ink=True)["task"]
    assert check["stage"] == "check" and check["next_problem"] is None
    assert h.assistant.tutor.quiz_number == 2 and not h.assistant.tutor.dictated, "next turn dictates a new one"
    assert _ask(h, "ready")["task"]["stage"] == "dictate"


def test_quiz_with_every_file_unticked_has_no_worksheet_problems(h: Harness) -> None:
    _bank(h, "a.md", ("1 + 1", "2", 1))
    h.assistant.selection.select([])
    assert _ask(h, "quiz me")["task"]["problem"] is None


def test_quiz_mode_starts_a_quiz_on_the_next_question(h: Harness) -> None:
    _bank(h, "a.md", ("1 + 1", "2", 1))
    h.assistant.tutor.set_mode("quiz")
    task = _ask(h, "ok")["task"]
    assert task["stage"] == "dictate" and task["problem"] == "1 + 1"


def test_quiz_check_without_ink_is_not_approved(h: Harness) -> None:
    _ask(h, "quiz me")
    _ask(h, "it's five", state="celebrating")
    assert h.payloads("avatar_update")[-1]["animation_state"] == "idle"
    assert h.assistant.tutor.quiz_number == 1 and h.assistant.tutor.score == 0


def test_stuck_mid_quiz_asks_for_a_hint_not_to_quit(h: Harness) -> None:
    _ask(h, "quiz me")
    task = _ask(h, "I'm stuck", state="error", ink=True)["task"]
    assert task["kind"] == "quiz" and task["stage"] == "check"
    assert h.assistant.tutor.quiz_active


def test_stop_the_quiz(h: Harness) -> None:
    _ask(h, "quiz me")
    assert _ask(h, "stop the quiz")["task"] is None
    assert h.assistant.tutor.to_dict() == {"mode": "help", "learn_topic": None, "quiz": None}
    assert _tutor_events(h)[-1]["quiz"] is None


def test_quiz_me_again_repeats_the_current_problem(h: Harness) -> None:
    _bank(h, "a.md", ("1 + 1", "2", 1), ("2 + 2", "4", 1))
    _ask(h, "quiz me")
    _ask(h, "yes", state="celebrating", ink=True)
    task = _ask(h, "quiz me")["task"]
    assert task["stage"] == "dictate" and task["number"] == 2, "the running quiz is kept"


def test_failed_dictation_is_read_out_again(h: Harness) -> None:
    _ask(h, "quiz me", state="error", spoken="The AI model took too long to answer.")
    assert not h.assistant.tutor.dictated
    assert _ask(h, "hello?")["task"]["stage"] == "dictate"


def test_quiz_stopped_while_thinking_is_not_scored(h: Harness) -> None:
    _bank(h, "a.md", ("1 + 1", "2", 1), ("2 + 2", "4", 1))
    _ask(h, "quiz me")
    h.vlm.on_call = h.assistant.tutor.stop_quiz
    _ask(h, "done", state="celebrating", ink=True)
    assert not h.assistant.tutor.quiz_active and h.assistant.tutor.score == 0


def test_learn_request_mid_quiz_keeps_the_quiz(h: Harness) -> None:
    _ask(h, "quiz me")
    kwargs = _ask(h, "teach me fractions")
    assert kwargs["task"]["kind"] == "learn" and kwargs["mode"] == "quiz"
    assert h.assistant.tutor.quiz_active


def test_history_records_mode_and_task_kind(h: Harness) -> None:
    _ask(h, "q")
    _ask(h, "quiz me")
    newest, oldest = h.assistant.history()[:2]
    assert (oldest["mode"], oldest["task_kind"]) == ("help", None)
    assert (newest["mode"], newest["task_kind"]) == ("quiz", "quiz")


def test_pipeline_failure_records_the_task(h: Harness) -> None:
    h.vlm.error = RuntimeError("boom")
    h.assistant.handle_query("quiz me")
    item = h.assistant.history()[0]
    assert item["error"] is True and item["task_kind"] == "quiz" and item["mode"] == "quiz"


def test_biology_profile_has_no_tutor_tasks_but_remembers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAR_PROFILE", "biology")
    bio = Harness()
    assert not bio.assistant.tutoring
    first = _ask(bio, "quiz me", spoken="Sure.")
    assert first == {"mode": "help", "task": None, "recent_turns": []}
    second = _ask(bio, "I want to learn how to solve fractions")
    assert second["task"] is None and second["recent_turns"] == [{"q": "quiz me", "a": "Sure."}]
    assert bio.assistant.tutor.mode == "help" and _tutor_events(bio) == []
    assert bio.assistant.history()[0]["task_kind"] is None


def test_start_quiz_uses_ticked_files_and_pushes(h: Harness) -> None:
    _bank(h, "a.md", ("1 + 1", "2", 1))
    _bank(h, "b.md", ("5 - 3", "2", 1))
    h.assistant.selection.select(["b.md"])
    h.assistant.start_quiz()
    assert h.assistant.tutor.current().problem == "5 - 3"  # type: ignore[union-attr]
    assert _tutor_events(h) == [h.assistant.tutor.to_dict()]
    h.assistant.push_tutor()
    assert len(_tutor_events(h)) == 1, "unchanged state is not pushed again"
