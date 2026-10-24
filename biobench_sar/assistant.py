"""The Sense -> Think -> Act pipeline: one query at a time, with a status state machine and history."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from biobench_sar.accessibility import AccessibilityStore, parse_changes
from biobench_sar.config import Settings
from biobench_sar.jsonfile import save_json
from biobench_sar.rag.quiz_bank import QuizBank, QuizProblem, pick_progressive
from biobench_sar.schemas import AvatarAction, error_action
from biobench_sar.tutor import (
    TutorSession,
    detect_learn_request_any,
    detect_mode_any,
    detect_offer_reply_any,
    is_quiz_stop_any,
    is_repeat_request_any,
)

log = logging.getLogger(__name__)

WORDS_PER_SECOND = 2.5
DRAWING_STATES = frozenset({"pointing", "celebrating"})
HISTORY_SIZE = 50
LANGUAGES = ("en", "es")
LANGUAGE_SETTINGS = ("auto", *LANGUAGES)
GLARE_INTERVAL_S = 2.0
# A bulb reflection is a patch of the board; a "hotspot" over half of it or more is an overexposed camera.
MAX_GLARE_SHARE = 0.5
NO_ANSWER_TEXT = "I can't see an answer on the board yet. Write it, then ask me to check it."
TEXTS = {
    "en": {
        "no_answer": NO_ANSWER_TEXT,
        "error": "Sorry, something went wrong. Please try again.",
        "last": "That was the last problem! ",
        "summary": "{ending}You solved {score} of {total} on the first try.",
    },
    "es": {
        "no_answer": "Todavía no veo una respuesta en el pizarrón. Escríbela y luego pídeme que la revise.",
        "error": "Perdón, algo salió mal. Inténtalo de nuevo.",
        "last": "¡Ese fue el último problema! ",
        "summary": "{ending}Resolviste {score} de {total} al primer intento.",
    },
}
# The model or the VLM client often announces the end itself; then the summary only adds the score.
_QUIZ_END_SAID = re.compile(
    r"\b(?:last problem|finished|all done|end of the quiz|[uú]ltimo problema|terminaste|terminamos|fin del quiz)\b",
    re.I,
)
_UNSET: Any = object()


def _language(code: Any) -> str:
    return "es" if code == "es" else "en"


def _speakable(text: str, language: str = "en") -> str:
    """Text the assistant adds itself goes through mathspeak too ("3 of 10" -> "three of ten" / "tres de diez")."""
    try:
        if _language(language) == "es":
            from biobench_sar.llm.mathspeak_es import to_speech_es as to_speech
        else:
            from biobench_sar.llm.mathspeak import to_speech

        return str(to_speech(text)).strip() or text
    except Exception:
        return text


def _detect_language(text: str, default: str) -> str:
    try:
        from biobench_sar.lang import detect_language

        return _language(detect_language(text, default=default))
    except Exception:
        log.exception("language detection failed")
        return _language(default)


class GlareMonitor:
    """glare.measure_glare on the latest raw camera frame, at most every `interval_s`; thread-safe.

    The report's hotspots are in that frame's camera pixels; hotspots_on_board maps them into projector space.
    """

    def __init__(self, interval_s: float = GLARE_INTERVAL_S, clock: Callable[[], float] = time.monotonic) -> None:
        self._interval = interval_s
        self._clock = clock
        self._lock = threading.Lock()
        self._at: float | None = None
        self._report: Any = None
        self._size: tuple[int, int] | None = None

    def report(self, frame_source: Callable[[], Any]) -> Any:
        """The GlareReport of a frame from `frame_source` (cached), or None without a frame."""
        with self._lock:
            now = self._clock()
            if self._at is not None and now - self._at < self._interval:
                return self._report
            self._at = now
            self._report, self._size = None, None
            try:
                frame = frame_source()
                if frame is not None:
                    from biobench_sar.core.glare import measure_glare

                    self._report = measure_glare(frame)
                    self._size = (int(frame.shape[1]), int(frame.shape[0]))
            except Exception:
                log.exception("glare check failed")
                self._report, self._size = None, None
            return self._report

    def status(self, frame_source: Callable[[], Any]) -> dict[str, Any] | None:
        """{"fraction", "severe", "hotspots": [[x, y, w, h], ...], "size": [w, h]} or None without a camera."""
        report = self.report(frame_source)
        if report is None:
            return None
        with self._lock:
            size = self._size
        body = report.to_dict()
        return {**body, "size": list(size) if size else None}


def hotspots_on_board(hotspots: Iterable[Any], calibration: Any) -> list[tuple[float, float, float, float]]:
    """Camera-pixel hotspot boxes as projector-space (x, y, w, h) boxes: the bounding box of the mapped corners."""
    boxes: list[tuple[float, float, float, float]] = []
    for x, y, w, h in hotspots:
        if calibration is None:
            boxes.append((float(x), float(y), float(w), float(h)))
            continue
        import numpy as np

        corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
        mapped = calibration.camera_to_projector(corners)
        x0, y0 = (float(v) for v in mapped.min(axis=0))
        x1, y1 = (float(v) for v in mapped.max(axis=0))
        boxes.append((x0, y0, x1 - x0, y1 - y0))
    return boxes


def drop_glare_ink(
    ink: list[Any], boxes: list[tuple[float, float, float, float]], board_size: tuple[int, int] | None = None
) -> list[Any]:
    """Ink regions whose center is not inside a glare hotspot: the bulb's reflection washes ink out and its rim
    reads as a stroke, so what is found there is not the student's writing. A "hotspot" over most of the board is
    an overexposed camera, not a reflection: the ink in it is still the student's, so it stays."""
    if board_size is not None:
        area = max(1, board_size[0] * board_size[1])
        boxes = [box for box in boxes if box[2] * box[3] < MAX_GLARE_SHARE * area]
    if not boxes:
        return ink

    def in_glare(region: Any) -> bool:
        try:
            cx, cy = float(region["cx"]), float(region["cy"])
        except (KeyError, TypeError, ValueError):
            return False
        return any(x <= cx <= x + w and y <= cy <= y + h for x, y, w, h in boxes)

    return [region for region in ink if not in_glare(region)]


Emit = Callable[[str, dict[str, Any]], None]
StartTask = Callable[..., Any]
Task = dict[str, Any]


def estimate_speech_seconds(text: str, rate: float = 1.0) -> float:
    """Seconds the voice needs for `text` at `rate` x its normal speed, plus one; a rate that isn't positive is 1."""
    rate = rate if rate > 0 else 1.0
    return len(text.split()) / (WORDS_PER_SECOND * rate) + 1.0


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
    # The answer as shown (math notation kept) and the reply's language; spoken_text is what the voice said.
    display_text: str = ""
    language: str = "en"

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
        # A failed save is logged; the in-memory selection still applies, it just won't survive a restart.
        save_json(self._path, payload, "document selection")


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
        # Captions, text size, voice speed, calm mode, high contrast: one set for the whole room, kept as JSON.
        self.accessibility = AccessibilityStore(settings.accessibility_file)
        # One change at a time from save to push: two devices changing it at once can't leave the voice or the
        # pages on the older settings.
        self._access_lock = threading.Lock()
        self.tutor = TutorSession()
        self._tutor_lock = threading.Lock()
        self._tutor_pushed = self.tutor.to_dict()
        self.glare = GlareMonitor()
        self._language_lock = threading.Lock()
        self._language = settings.language if settings.language in LANGUAGE_SETTINGS else "auto"
        # The language of the last reply: what "auto" falls back on when a question leans neither way.
        self._reply_language = self._language if self._language in LANGUAGES else "en"
        self._typed = threading.local()

    # ---- accessibility -----------------------------------------------------
    def accessibility_body(self) -> dict[str, bool | str]:
        """The room's five accessibility settings, every key."""
        return self.accessibility.current.to_dict()

    def set_accessibility(self, body: object) -> dict[str, bool | str]:
        """Apply a partial change ({"speech_rate": "slower"}, ...); ValueError, with nothing saved or pushed, for
        anything else. The voice takes the new speed and every open page gets every key."""
        changes = parse_changes(body)
        with self._access_lock:
            saved = self.accessibility.update(changes).to_dict()
            self.push_speech_rate()
            self._emit("accessibility", saved)
        return saved

    def push_speech_rate(self) -> None:
        """Hand the saved voice speed to the TTS service; one without set_rate (older fakes) keeps its own."""
        tts = getattr(self.services, "tts", None)
        set_rate = getattr(tts, "set_rate", None)
        if callable(set_rate):
            try:
                set_rate(self.accessibility.current.rate_factor)
            except Exception:
                log.exception("could not set the voice speed")

    # ---- language ----------------------------------------------------------
    @property
    def language_setting(self) -> str:
        """ "auto" (reply in the student's language) | "en" | "es"."""
        with self._language_lock:
            return self._language

    @property
    def current_language(self) -> str:
        """The language replies are in now: the setting when it is forced, else the last reply's."""
        with self._language_lock:
            return self._language if self._language in LANGUAGES else self._reply_language

    def language_body(self) -> dict[str, str]:
        with self._language_lock:
            current = self._language if self._language in LANGUAGES else self._reply_language
            return {"setting": self._language, "current": current}

    def set_language(self, setting: str) -> dict[str, str]:
        """Switch the setting ("auto" | "en" | "es"); ValueError otherwise. Whisper decodes in a forced language."""
        if setting not in LANGUAGE_SETTINGS:
            raise ValueError(f"language must be one of {', '.join(LANGUAGE_SETTINGS)}")
        with self._language_lock:
            self._language = setting
        audio = getattr(self.services, "audio", None)
        set_audio = getattr(audio, "set_language", None)
        if callable(set_audio):
            try:
                set_audio(setting)
            except Exception:
                log.exception("could not set the microphone language")
        body = self.language_body()
        self._emit("language", body)
        return body

    def reply_language(self, text: str, *, spoken: bool) -> str:
        """The language to answer `text` in: the setting when forced; with "auto", the language Whisper heard a
        spoken question in, else the text's own (the last reply's language when it leans neither way)."""
        with self._language_lock:
            setting, previous = self._language, self._reply_language
        if setting in LANGUAGES:
            return setting
        if spoken:
            heard = getattr(getattr(self.services, "audio", None), "last_language", None)
            if heard in LANGUAGES:
                return str(heard)
        return _detect_language(text, previous)

    def _replied_in(self, language: str) -> None:
        with self._language_lock:
            changed = self._reply_language != language and self._language == "auto"
            self._reply_language = language
        if changed:
            self._emit("language", self.language_body())

    def submit_text(self, text: str) -> None:
        """A typed question (dashboard, socket): delivered through the audio service like a spoken one, but its
        language comes from the text, not from what the microphone last heard."""
        self._typed.active = True
        try:
            self.services.audio.submit_text(text)
        finally:
            self._typed.active = False

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
            # Decided here, on the delivering thread: AudioTrigger.last_language belongs to this question only now.
            language = self.reply_language(text, spoken=not getattr(self._typed, "active", False))
            self._start_task(self._run, text, language)
        except Exception:
            self._busy.release()
            raise
        return True

    def _board(self) -> tuple[Any, Any, Any]:
        """(raw camera frame, its calibration, projector-space frame), read together; raw is None for a vision
        service without latest_frame_and_calibration."""
        vision = self.services.vision
        both = getattr(vision, "latest_frame_and_calibration", None)
        if not callable(both):
            return None, None, vision.latest_warped()
        raw, calibration = both()
        if raw is None:
            return None, None, None
        return raw, calibration, calibration.warp_frame(raw) if calibration is not None else raw

    def _board_image(self, raw: Any, calibration: Any, warped: Any, ink: list[Any]) -> tuple[bytes | None, Any]:
        """(JPEG for the model, its InkCrop or None): the high-res crop of the ink, else the whole board."""
        from biobench_sar.core.vision_loop import encode_jpeg

        crop = None
        if ink:
            try:
                from biobench_sar.core.ink_crop import crop_from_warped, crop_ink_view

                if raw is not None and calibration is not None:
                    crop = crop_ink_view(raw, calibration, ink)
                if crop is None and warped is not None:
                    crop = crop_from_warped(warped, ink)
            except Exception as exc:
                log.warning("ink crop failed, sending the whole board: %s", exc)
                crop = None
        if crop is not None:
            return crop.jpeg, crop
        return (encode_jpeg(warped) if warped is not None else None), None

    def _ink(self, raw: Any, calibration: Any, warped: Any) -> list[Any]:
        from biobench_sar.core.vision_loop import find_ink_regions

        if warped is None:
            return []
        ink = find_ink_regions(warped, **self.settings.ink_kwargs)
        if not ink or raw is None:
            return ink  # without the raw frame and its calibration a hotspot can't be placed on the board
        hotspots = list(getattr(self.glare.report(lambda: raw), "hotspots", None) or [])
        if not hotspots:
            return ink
        try:
            board = (int(warped.shape[1]), int(warped.shape[0]))
            return drop_glare_ink(ink, hotspots_on_board(hotspots, calibration), board)
        except Exception:
            log.exception("glare filter failed; keeping every ink region")
            return ink

    def _run(self, text: str, language: str = "en") -> None:
        svc = self.services
        language = _language(language)
        started = time.monotonic()
        sources: list[str] = []
        ink: list[Any] = []
        frame = None
        task: Task | None = None
        mode = "help"
        try:
            self._emit("transcript", {"text": text})
            self.emit_status("thinking")
            raw, calibration, frame = self._board()
            ink = self._ink(raw, calibration, frame)
            # Planned after looking at the board: "how do I add fractions?" next to ink is about that ink.
            task = self._plan(text, has_ink=bool(ink), language=language) if self.tutoring else None
            mode = self.tutor.mode if self.tutoring else "help"
            self.push_tutor()
            context = self._retrieve(text)
            sources = list(dict.fromkeys(chunk.source for chunk in context))
            image, crop = self._board_image(raw, calibration, frame, ink)
            extra: dict[str, Any] = {"language": language}
            if crop is not None:
                extra["crop"] = crop
            action = svc.vlm.evaluate(
                text,
                image,
                context,
                ink,
                self.settings.projector_size,
                mode=mode,
                task=task,
                recent_turns=self.tutor.turns(),
                **extra,
            )
            display = str(getattr(action, "display_text", None) or action.spoken_text)
            # A failed call (timeout, model down) says nothing about the student: no grading, nothing remembered.
            failed = bool(getattr(action, "failed", False))
            checking = task is not None and task["kind"] == "quiz" and task["stage"] == "check"
            if checking and not ink and not failed:
                # Nothing on the board to grade. Saying "correct, next problem" here would leave the quiz behind.
                display = TEXTS[language]["no_answer"]
                action = replace(action, spoken_text=_speakable(display, language), animation_state="idle")
            elif action.animation_state in DRAWING_STATES and not ink and not _is_learn(task):
                # Nothing drawn to point at or approve; small VLMs still pick these for plain questions.
                action = replace(action, animation_state="idle")
            if task is not None and task["kind"] == "quiz" and not failed:
                action, display = self._advance_quiz(task, action, display, graded=bool(ink), language=language)
            if not failed:
                self.tutor.add_turn(text, action.spoken_text)
            self.push_tutor()
            said = {"display_text": display, "spoken_text": action.spoken_text, "language": language}
            self._emit("avatar_update", {**action.to_dict(), **said})
            self.emit_status("speaking")
            audio_name = svc.tts.speak(action.spoken_text, lang=language)
            url = f"/tts/{audio_name}" if audio_name else None
            self._emit("speak", {"url": url, "text": action.spoken_text, **said})
            # A slower voice talks for longer: the wake-word loop stays deaf until it is done.
            rate = self.accessibility.current.rate_factor
            self._speaking_until = time.monotonic() + estimate_speech_seconds(action.spoken_text, rate)
            self._replied_in(language)
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
                display=display,
                language=language,
            )
        except Exception:
            log.exception("query pipeline failed")
            action = error_action(TEXTS[language]["error"], self.settings.projector_size)
            said = {"display_text": action.spoken_text, "spoken_text": action.spoken_text, "language": language}
            self._emit("avatar_update", {**action.to_dict(), **said})
            self._record(
                text,
                action,
                sources,
                ink,
                frame is not None,
                started,
                error=True,
                mode=mode,
                task=task,
                language=language,
            )
        finally:
            self.emit_status("idle")
            self._busy.release()

    def _plan(self, text: str, *, has_ink: bool = False, language: str = "en") -> Task | None:
        """What this question asks the model to do: a lesson step, a practice problem, a quiz step or plain help.
        English and Spanish requests both count, the reply language's first."""
        tutor = self.tutor
        if tutor.offer_pending:
            reply = detect_offer_reply_any(text, language)
            if reply is not None:
                tutor.resolve_offer()
                stage = "example" if reply == "generate" else "own"
                return {"kind": "learn", "topic": tutor.learn_topic, "stage": stage}
            # Only the very next turn answers the offer: a "no" or "I'll write it" much later means something else.
            tutor.expire_offer()
        topic = detect_learn_request_any(text, language, board_in_use=has_ink or tutor.quiz_active)
        if topic is not None:
            tutor.start_learning(topic)
            return {"kind": "learn", "topic": topic, "stage": "offer"}
        asked = detect_mode_any(text, language)
        stuck = asked == "help" and tutor.quiz_active and not is_quiz_stop_any(text)
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
        elif not tutor.dictated or is_repeat_request_any(text):
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

    def _advance_quiz(
        self, task: Task, action: AvatarAction, display: str, *, graded: bool = True, language: str = "en"
    ) -> tuple[AvatarAction, str]:
        """Quiz bookkeeping for a finished turn. The VLM client says what it graded and read out (TutorAction
        verdict / dictated); for a plain AvatarAction the state stands in: celebrating = correct, error = wrong.
        Returns the action and the text shown for it (the final summary is added to both)."""
        tutor = self.tutor
        if not tutor.quiz_active or tutor.quiz_number != task["number"]:
            return action, display  # the quiz was stopped or restarted while the model was thinking
        if task["stage"] == "dictate":
            if action.animation_state != "error":
                tutor.mark_dictated()
            return action, display
        if task["stage"] != "check" or not graded:
            return action, display  # a hint, or nothing on the board to grade: the attempt does not count
        verdict = getattr(action, "verdict", _UNSET)
        if verdict is _UNSET:
            verdict = {"celebrating": "correct", "error": "wrong"}.get(action.animation_state)
        if verdict == "wrong":
            tutor.record(False)
        elif verdict == "correct":
            tutor.record(True)
            if not tutor.quiz_active:
                texts = TEXTS[_language(language)]
                ending = "" if _QUIZ_END_SAID.search(action.spoken_text) else texts["last"]
                shown = texts["summary"].format(ending=ending, score=tutor.score, total=tutor.quiz_total)
                # Spoken like every other reply: "You solved three of ten", never "3 of 10" read by the voice.
                summary = _speakable(shown, language)
                return replace(action, spoken_text=f"{action.spoken_text} {summary}"), f"{display} {shown}"
            read = getattr(action, "dictated", _UNSET)
            if read is _UNSET:
                read = task["next_problem"]
            if read:
                tutor.mark_dictated()  # the next problem was read out with the praise: the next turn checks it
        return action, display

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
        display: str | None = None,
        language: str = "en",
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
            display_text=display or action.spoken_text,
            language=_language(language),
        )
        self._next_id += 1
        self._history.append(item)
        self._emit("interaction", item.to_dict())


def _is_learn(task: Task | None) -> bool:
    return task is not None and task["kind"] == "learn"
