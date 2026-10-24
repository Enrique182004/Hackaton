"""The Sense -> Think -> Act pipeline: one query at a time, with a status state machine and history."""

from __future__ import annotations

import contextlib
import json
import logging
import os
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
from biobench_sar.schemas import AvatarAction, error_action

log = logging.getLogger(__name__)

WORDS_PER_SECOND = 2.5
HISTORY_SIZE = 50

Emit = Callable[[str, dict[str, Any]], None]
StartTask = Callable[..., Any]


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
        try:
            self._emit("transcript", {"text": text})
            self.emit_status("thinking")
            frame = svc.vision.latest_warped()
            ink = find_ink_regions(frame, **self.settings.ink_kwargs) if frame is not None else []
            context = self._retrieve(text)
            sources = list(dict.fromkeys(chunk.source for chunk in context))
            image = encode_jpeg(frame) if frame is not None else None
            action = svc.vlm.evaluate(text, image, context, ink, self.settings.projector_size)
            if action.animation_state == "pointing" and not ink:
                # Nothing on the board to point at; small VLMs still pick "pointing" for plain questions.
                action = replace(action, animation_state="idle")
            self._emit("avatar_update", action.to_dict())
            self.emit_status("speaking")
            audio_name = svc.tts.speak(action.spoken_text)
            self._emit("speak", {"url": f"/tts/{audio_name}" if audio_name else None, "text": action.spoken_text})
            self._speaking_until = time.monotonic() + estimate_speech_seconds(action.spoken_text)
            self._record(
                text, action, sources, ink, frame is not None, started, error=action.animation_state == "error"
            )
        except Exception:
            log.exception("query pipeline failed")
            action = error_action("Sorry, something went wrong. Please try again.", self.settings.projector_size)
            self._emit("avatar_update", action.to_dict())
            self._record(text, action, sources, ink, frame is not None, started, error=True)
        finally:
            self.emit_status("idle")
            self._busy.release()

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
        )
        self._next_id += 1
        self._history.append(item)
        self._emit("interaction", item.to_dict())
