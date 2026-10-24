from __future__ import annotations

import asyncio
import logging
import queue
import subprocess
import threading
import time
import uuid
from pathlib import Path

from biobench_sar.config import Settings

log = logging.getLogger(__name__)

MAX_FILES = 50
EDGE_TIMEOUT_S = 8.0
EDGE_RETRY_AFTER_S = 60.0
ORDER = ["edge", "say", "pyttsx3", "none"]


class TTSEngine:
    def __init__(self, backend: str = "edge", voice: str = "en-US-AriaNeural", out_dir: Path | None = None) -> None:
        self.backend = backend if backend in ORDER else "none"
        self.voice = voice
        self.out_dir = Path(out_dir) if out_dir is not None else Settings().tts_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._proc: subprocess.Popen | None = None
        self._q: queue.Queue[str | None] | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._edge_failed_at: float | None = None

    def speak(self, text: str) -> str | None:
        text = (text or "").strip()
        if not text:
            return None
        for name in ORDER[ORDER.index(self.backend):]:
            try:
                if name == "edge":
                    if self._edge_cooling_down():
                        continue
                    return self._edge(text)
                if name == "say" and self._say(text):
                    return None
                if name == "pyttsx3" and self._pyttsx3(text):
                    return None
            except Exception as exc:  # noqa: BLE001
                log.warning("tts backend %s failed: %s", name, exc)
        log.info("tts disabled/unavailable; text: %s", text)
        return None

    def stop(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001
                pass
        if self._q is not None:
            self._q.put(None)
            self._q = None
            self._thread = None

    def _edge_cooling_down(self) -> bool:
        failed_at = self._edge_failed_at
        return failed_at is not None and time.monotonic() - failed_at < EDGE_RETRY_AFTER_S

    def _edge(self, text: str) -> str:
        import edge_tts

        name = f"{uuid.uuid4().hex}.mp3"
        path = self.out_dir / name
        # Offline venues: fail fast so the say/pyttsx3 fallback speaks instead of hanging the pipeline.
        try:
            asyncio.run(asyncio.wait_for(edge_tts.Communicate(text, self.voice).save(str(path)), EDGE_TIMEOUT_S))
        except BaseException:
            self._edge_failed_at = time.monotonic()
            path.unlink(missing_ok=True)
            raise
        self._edge_failed_at = None
        self._cleanup()
        return name

    def _cleanup(self) -> None:
        files = sorted(self.out_dir.glob("*.mp3"), key=lambda p: p.stat().st_mtime)
        for old in files[: max(0, len(files) - MAX_FILES)]:
            try:
                old.unlink()
            except OSError:
                pass

    def _say(self, text: str) -> bool:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()
            proc = subprocess.Popen(["say", text])
            self._proc = proc
        return True

    def _pyttsx3(self, text: str) -> bool:
        import pyttsx3  # noqa: F401  (fail fast if missing)

        with self._lock:
            if self._q is None:
                self._q = queue.Queue()
                self._thread = threading.Thread(target=self._pyttsx3_worker, args=(self._q,), daemon=True, name="pyttsx3")
                self._thread.start()
            q = self._q
        q.put(text)
        return True

    @staticmethod
    def _pyttsx3_worker(q: queue.Queue) -> None:
        try:
            import pyttsx3

            engine = pyttsx3.init()
        except Exception:  # noqa: BLE001
            log.exception("pyttsx3 init failed")
            return
        while True:
            item = q.get()
            if item is None:
                return
            try:
                engine.say(item)
                engine.runAndWait()
            except Exception:  # noqa: BLE001
                log.exception("pyttsx3 speak failed")
