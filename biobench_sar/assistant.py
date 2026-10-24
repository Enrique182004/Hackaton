"""The Sense -> Think -> Act pipeline: one query at a time, with a status state machine and history."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from biobench_sar.config import Settings
from biobench_sar.rag.quiz_bank import QuizBank, QuizProblem, pick_progressive
from biobench_sar.schemas import AvatarAction, error_action
from biobench_sar.tutor import (
    TutorSession,
    detect_learn_request,
    detect_mode,
    detect_offer_reply,
    is_quiz_stop,
    is_repeat_request,
)

log = logging.getLogger(__name__)

WORDS_PER_SECOND = 2.5
DRAWING_STATES = frozenset({"pointing", "celebrating"})
HISTORY_SIZE = 50
NO_ANSWER_TEXT = "I can't see an answer on the board yet. Write it, then ask me to check it."
# The model or the VLM client often announces the end itself; then the summary only adds the score.
_QUIZ_END_SAID = re.compile(r"\b(?:last problem|finished|all done|end of the quiz)\b", re.I)
_UNSET: Any = object()


def _speakable(text: str) -> str:
    """Text the assistant adds itself goes through mathspeak too ("3 of 10" -> "three of ten")."""
    try:
        from biobench_sar.llm.mathspeak import to_speech

        return str(to_speech(text)).strip() or text
    except Exception:
        return text


Emit = Callable[[str, dict[str, Any]], None]
StartTask = Callable[..., Any]
Task = dict[str, Any]


def estimate_speech_seconds(text: str) -> float:
    return len(text.split()) / WORDS_PER_SECOND + 1.0


@dataclass(frozen=True)
class Interaction:
    id: int
    at: str
    query: str
    spoken_text: str
    animation_state: str
    target_x: int
    target_y: int
    sources: list[str]
    ink_regions: int
    had_frame: bool
    latency_s: float
    error: bool
    mode: str = "help"
    task_kind: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DocumentSelection:
    """Which uploaded files answers may draw on; None means every file. Persisted as JSON, thread-safe.

    File format: {"all": true} or {"sources": [...]}. A missing or unreadable file means every file is active.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._sources: set[str] | None = self._load()

    def _load(self) -> set[str] | None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or data.get("all") is True or not isinstance(data.get("sources"), list):
            return None
        return {s for s in data["sources"] if isinstance(s, str)}

    @property
    def active(self) -> set[str] | None:
        with self._lock:
            return None if self._sources is None else set(self._sources)

    def query_sources(self) -> list[str] | None:
        """What the store should search: None for every file, else the selected names in sorted order."""
        active = self.active
        return None if active is None else sorted(active)

    def select_all(self) -> None:
        with self._lock:
            self._sources = None
            self._save()

    def select(self, sources: Iterable[str]) -> None:
        with self._lock:
            self._sources = set(sources)
            self._save()

    def activate(self, sources: Iterable[str]) -> None:
        """Add files to an explicit selection, so a fresh upload is never silently left out of answers."""
        with self._lock:
            if self._sources is None:
                return
            new = set(sources) - self._sources
            if new:
                self._sources |= new
                self._save()

    def discard(self, source: str) -> None:
        with self._lock:
            if self._sources is not None and source in self._sources:
                self._sources.discard(source)
                self._save()

    def _save(self) -> None:
        payload = {"all": True} if self._sources is None else {"sources": sorted(self._sources)}
        tmp = ""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=f".{self._path.name}.", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp, self._path)
        except OSError as exc:
            # The in-memory selection still applies; it just won't survive a restart.
            log.warning("could not save document selection to %s: %s", self._path, exc)
            if tmp:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)


class Assistant:
    """Owns the busy lock, the status shown to clients and the interaction history.

    `services` is attached after construction because AudioTrigger needs `handle_query` as its callback.
    """

    def __init__(self, settings: Settings, emit: Emit, start_task: StartTask) -> None:
        self.settings = settings
        self.services: Any = None
        self.calibrated = False
        self.started_at = time.monotonic()
        self._emit = emit
        self._start_task = start_task
        self._busy = threading.Lock()
        self._state = "idle"
        self._speaking_until = 0.0
        self._history: deque[Interaction] = deque(maxlen=HISTORY_SIZE)
        self._next_id = 1
        self.selection = DocumentSelection(settings.selection_file)
        self.quiz_bank = QuizBank(settings.quiz_bank_file)
        self.tutor = TutorSession()
        self._tutor_lock = threading.Lock()
        self._tutor_pushed = self.tutor.to_dict()

    @property
    def active_sources(self) -> set[str] | None:
        """Files answers may use; None = every file."""
        return self.selection.active

    @property
    def state(self) -> str:
        return self._state

    @property
    def busy(self) -> bool:
        return self._busy.locked()

    def history(self) -> list[dict[str, Any]]:
        # Snapshot first: the pipeline thread appends while to_dict() runs, and a live deque iterator raises.
        return [item.to_dict() for item in reversed(list(self._history))]

    @property
    def last_latency_s(self) -> float | None:
        return self._history[-1].latency_s if self._history else None

    def emit_status(self, value: str, detail: str | None = None) -> None:
        self._state = value
        payload: dict[str, Any] = {"state": value}
        if detail:
            payload["detail"] = detail
        self._emit("status", payload)

    def on_state(self, value: str) -> None:
        # While a query runs, the pipeline owns the state machine; the mic thread may not reset it.
        if value == "idle" and self.busy:
            return
        self.emit_status(value)

    @property
    def tutoring(self) -> bool:
        """Tutor modes, quizzes and lessons are math-only; other subjects just keep the conversation memory."""
        return self.settings.profile == "math"

    def push_tutor(self) -> None:
        """Send the tutor state to every client when it changed since the last push."""
        with self._tutor_lock:
            state = self.tutor.to_dict()
            if state == self._tutor_pushed:
                return
            self._tutor_pushed = state
        self._emit("tutor", state)

    def quiz_problems(self) -> list[QuizProblem]:
        """Worksheet problems from the ticked files only, at most Settings.quiz_length spread over every level."""
        return pick_progressive(self.quiz_bank.problems(self.selection.query_sources()), self.settings.quiz_length)

    def _start_tutor_quiz(self) -> None:
        self.tutor.start_quiz(self.quiz_problems(), length=self.settings.quiz_length)

    def start_quiz(self) -> None:
        self._start_tutor_quiz()
        self.push_tutor()

    def should_listen(self) -> bool:
        # The browser keeps playing the answer after the pipeline finishes, so busy alone isn't enough.
        return not self.busy and time.monotonic() >= self._speaking_until

    def handle_query(self, text: str) -> bool:
        """Start a query in the background. False if it's empty or another query is still running."""
        text = (text or "").strip()
        if not text:
            return False
        if not self._busy.acquire(blocking=False):
            self.emit_status(self._state, "busy")
            return False
        try:
            self._start_task(self._run, text)
        except Exception:
            self._busy.release()
            raise
        return True

    def _run(self, text: str) -> None:
        from biobench_sar.core.vision_loop import encode_jpeg, find_ink_regions

        svc = self.services
        started = time.monotonic()
        sources: list[str] = []
        ink: list[Any] = []
        frame = None
        task: Task | None = None
        mode = "help"
        try:
            self._emit("transcript", {"text": text})
            self.emit_status("thinking")
            frame = svc.vision.latest_warped()
            ink = find_ink_regions(frame, **self.settings.ink_kwargs) if frame is not None else []
            # Planned after looking at the board: "how do I add fractions?" next to ink is about that ink.
            task = self._plan(text, has_ink=bool(ink)) if self.tutoring else None
            mode = self.tutor.mode if self.tutoring else "help"
            self.push_tutor()
            context = self._retrieve(text)
            sources = list(dict.fromkeys(chunk.source for chunk in context))
            image = encode_jpeg(frame) if frame is not None else None
            action = svc.vlm.evaluate(
                text,
                image,
                context,
                ink,
                self.settings.projector_size,
                mode=mode,
                task=task,
                recent_turns=self.tutor.turns(),
            )
            # A failed call (timeout, model down) says nothing about the student: no grading, nothing remembered.
            failed = bool(getattr(action, "failed", False))
            checking = task is not None and task["kind"] == "quiz" and task["stage"] == "check"
            if checking and not ink and not failed:
                # Nothing on the board to grade. Saying "correct, next problem" here would leave the quiz behind.
                action = replace(action, spoken_text=NO_ANSWER_TEXT, animation_state="idle")
            elif action.animation_state in DRAWING_STATES and not ink and not _is_learn(task):
                # Nothing drawn to point at or approve; small VLMs still pick these for plain questions.
                action = replace(action, animation_state="idle")
            if task is not None and task["kind"] == "quiz" and not failed:
                action = self._advance_quiz(task, action, graded=bool(ink))
            if not failed:
                self.tutor.add_turn(text, action.spoken_text)
            self.push_tutor()
            self._emit("avatar_update", action.to_dict())
            self.emit_status("speaking")
            audio_name = svc.tts.speak(action.spoken_text)
            self._emit("speak", {"url": f"/tts/{audio_name}" if audio_name else None, "text": action.spoken_text})
            self._speaking_until = time.monotonic() + estimate_speech_seconds(action.spoken_text)
            self._record(
                text,
                action,
                sources,
                ink,
                frame is not None,
                started,
                error=action.animation_state == "error",
                mode=mode,
                task=task,
            )
        except Exception:
            log.exception("query pipeline failed")
            action = error_action("Sorry, something went wrong. Please try again.", self.settings.projector_size)
            self._emit("avatar_update", action.to_dict())
            self._record(text, action, sources, ink, frame is not None, started, error=True, mode=mode, task=task)
        finally:
            self.emit_status("idle")
            self._busy.release()

    def _plan(self, text: str, *, has_ink: bool = False) -> Task | None:
        """What this question asks the model to do: a lesson step, a practice problem, a quiz step or plain help."""
        tutor = self.tutor
        if tutor.offer_pending:
            reply = detect_offer_reply(text)
            if reply is not None:
                tutor.resolve_offer()
                stage = "example" if reply == "generate" else "own"
                return {"kind": "learn", "topic": tutor.learn_topic, "stage": stage}
            # Only the very next turn answers the offer: a "no" or "I'll write it" much later means something else.
            tutor.expire_offer()
        topic = detect_learn_request(text, board_in_use=has_ink or tutor.quiz_active)
        if topic is not None:
            tutor.start_learning(topic)
            return {"kind": "learn", "topic": topic, "stage": "offer"}
        asked = detect_mode(text)
        stuck = asked == "help" and tutor.quiz_active and not is_quiz_stop(text)
        if stuck:
            asked = None  # "I'm stuck" mid-quiz wants a hint on this problem, not to quit the quiz
        if asked is not None:
            tutor.set_mode(asked)
        if tutor.mode == "practice":
            return {"kind": "practice"}
        if tutor.mode != "quiz":
            return None
        if asked == "quiz" or not tutor.quiz_active:
            if not tutor.quiz_active:
                self._start_tutor_quiz()
            stage = "dictate"
        elif not tutor.dictated or is_repeat_request(text):
            stage = "dictate"
        else:
            stage = "hint" if stuck else "check"
        current, upcoming = tutor.current(), tutor.next_problem()
        return {
            "kind": "quiz",
            "stage": stage,
            "number": tutor.quiz_number,
            "total": tutor.quiz_total,
            "problem": current.problem if current else None,
            "answer": current.answer if current else None,
            "next_problem": upcoming.problem if stage == "check" and upcoming else None,
        }

    def _advance_quiz(self, task: Task, action: AvatarAction, *, graded: bool = True) -> AvatarAction:
        """Quiz bookkeeping for a finished turn. The VLM client says what it graded and read out (TutorAction
        verdict / dictated); for a plain AvatarAction the state stands in: celebrating = correct, error = wrong."""
        tutor = self.tutor
        if not tutor.quiz_active or tutor.quiz_number != task["number"]:
            return action  # the quiz was stopped or restarted while the model was thinking
        if task["stage"] == "dictate":
            if action.animation_state != "error":
                tutor.mark_dictated()
            return action
        if task["stage"] != "check" or not graded:
            return action  # a hint, or nothing on the board to grade: the attempt does not count
        verdict = getattr(action, "verdict", _UNSET)
        if verdict is _UNSET:
            verdict = {"celebrating": "correct", "error": "wrong"}.get(action.animation_state)
        if verdict == "wrong":
            tutor.record(False)
        elif verdict == "correct":
            tutor.record(True)
            if not tutor.quiz_active:
                ending = "" if _QUIZ_END_SAID.search(action.spoken_text) else "That was the last problem! "
                summary = _speakable(f"{ending}You solved {tutor.score} of {tutor.quiz_total} on the first try.")
                return replace(action, spoken_text=f"{action.spoken_text} {summary}")
            read = getattr(action, "dictated", _UNSET)
            if read is _UNSET:
                read = task["next_problem"]
            if read:
                tutor.mark_dictated()  # the next problem was read out with the praise: the next turn checks it
        return action

    def _retrieve(self, text: str) -> list[Any]:
        sources = self.selection.query_sources()
        if sources is not None and not sources:
            return []  # the student switched every file off: answer without notes
        try:
            return list(self.services.store.query(text, self.settings.rag_top_k, sources=sources))
        except Exception:
            log.exception("RAG query failed; continuing without context")
            return []

    def _record(
        self,
        query: str,
        action: AvatarAction,
        sources: list[str],
        ink: list[Any],
        had_frame: bool,
        started: float,
        *,
        error: bool,
        mode: str = "help",
        task: Task | None = None,
    ) -> None:
        item = Interaction(
            id=self._next_id,
            at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            query=query,
            spoken_text=action.spoken_text,
            animation_state=action.animation_state,
            target_x=action.target_x,
            target_y=action.target_y,
            sources=sources,
            ink_regions=len(ink),
            had_frame=had_frame,
            latency_s=round(time.monotonic() - started, 2),
            error=error,
            mode=mode,
            task_kind=None if task is None else str(task["kind"]),
        )
        self._next_id += 1
        self._history.append(item)
        self._emit("interaction", item.to_dict())


def _is_learn(task: Task | None) -> bool:
    return task is not None and task["kind"] == "learn"
