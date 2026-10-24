from __future__ import annotations

import logging
import re
import threading
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
WAKE_CHUNK_SECONDS = 3.0
SILENCE_RMS = 0.01


class AudioTrigger:
    def __init__(
        self,
        on_query: Callable[[str], None],
        whisper_model: str = "base",
        wake_word: str = "assistant",
        mode: str = "push",
        record_seconds: float = 5.0,
        on_state: Callable[[str], None] | None = None,
        recorder: Callable[[float], np.ndarray] | None = None,
        transcriber: Callable[[np.ndarray], str] | None = None,
        wake_chunk_seconds: float = WAKE_CHUNK_SECONDS,
        should_listen: Callable[[], bool] | None = None,
        silence_rms: float = SILENCE_RMS,
    ) -> None:
        self.on_query = on_query
        self.whisper_model = whisper_model
        self.wake_word = wake_word
        self.mode = mode
        self.record_seconds = record_seconds
        self.on_state = on_state
        self.wake_chunk_seconds = wake_chunk_seconds
        self._recorder = recorder
        self._transcriber = transcriber
        self._should_listen = should_listen or (lambda: True)
        self.silence_rms = silence_rms
        self._model = None
        self._model_lock = threading.Lock()
        self._busy_lock = threading.Lock()
        # idle -> pending (requested, waiting for the wake loop) -> recording -> idle
        self._push_state = "idle"
        self._push_requested = threading.Event()
        self._stop = threading.Event()
        self._wake_thread: threading.Thread | None = None
        self._available: bool | None = None

    @property
    def available(self) -> bool:
        if self._recorder is not None:
            return True
        if self._available is None:
            self._available = self._probe_mic()
        return self._available

    @staticmethod
    def _probe_mic() -> bool:
        try:
            import pyaudio

            pa = pyaudio.PyAudio()
            try:
                return any(
                    int(pa.get_device_info_by_index(i).get("maxInputChannels", 0)) > 0
                    for i in range(pa.get_device_count())
                )
            finally:
                pa.terminate()
        except Exception as exc:  # noqa: BLE001
            log.info("audio unavailable: %s", exc)
            return False

    def _state(self, name: str) -> None:
        if self.on_state is None:
            return
        try:
            self.on_state(name)
        except Exception:  # noqa: BLE001
            log.exception("on_state callback failed")

    def start(self) -> None:
        if self.mode != "wake" or not self.available:
            return
        if self._wake_thread and self._wake_thread.is_alive():
            return
        self._stop.clear()
        self._wake_thread = threading.Thread(target=self._wake_loop, daemon=True, name="wake-listener")
        self._wake_thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._wake_thread
        if t and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=self.wake_chunk_seconds + 5)
        self._wake_thread = None

    def trigger(self) -> bool:
        """Request a push-to-talk recording. True if one will happen (now, or right after
        the wake loop's current chunk); False only if one is already being recorded."""
        with self._busy_lock:
            if self._push_state == "recording":
                return False
            if self._push_state == "pending":
                return True
            self._push_state = "pending"
            wake_running = self.mode == "wake" and self._wake_thread is not None and self._wake_thread.is_alive()
            if wake_running:
                self._push_requested.set()
                return True
        threading.Thread(target=self._run_push, daemon=True, name="push-to-talk").start()
        return True

    def submit_text(self, text: str) -> None:
        text = (text or "").strip()
        if text:
            self.on_query(text)

    def transcribe(self, audio: np.ndarray) -> str:
        if self._transcriber is not None:
            return self._transcriber(audio).strip()
        with self._model_lock:
            if self._model is None:
                import whisper

                self._model = self._model or whisper.load_model(self.whisper_model)
            model = self._model
        result = model.transcribe(np.asarray(audio, dtype=np.float32), fp16=False, language="en")
        return str(result.get("text", "")).strip()

    def _record(self, seconds: float) -> np.ndarray:
        if self._recorder is not None:
            return self._recorder(seconds)
        import pyaudio

        pa = pyaudio.PyAudio()
        try:
            stream = pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=1024)
            try:
                n = int(SAMPLE_RATE * seconds)
                frames = []
                got = 0
                while got < n:
                    count = min(1024, n - got)
                    frames.append(stream.read(count, exception_on_overflow=False))
                    got += count
            finally:
                stream.stop_stream()
                stream.close()
        finally:
            pa.terminate()
        pcm = np.frombuffer(b"".join(frames), dtype=np.int16)
        return pcm.astype(np.float32) / 32768.0

    def _heard(self, audio: np.ndarray) -> str:
        """Transcribe unless the clip is silent (saves CPU; Whisper hallucinates text on silence)."""
        arr = np.asarray(audio, dtype=np.float32)
        if arr.size == 0 or float(np.sqrt(np.mean(np.square(arr)))) < self.silence_rms:
            return ""
        return self.transcribe(arr)

    def _run_push(self) -> None:
        self._deliver(self._do_push())

    def _do_push(self) -> str:
        with self._busy_lock:
            self._push_state = "recording"
        self._push_requested.clear()
        text = ""
        try:
            self._state("listening")
            audio = self._record(self.record_seconds)
            self._state("transcribing")
            text = self._heard(audio)
        except Exception:  # noqa: BLE001
            log.exception("push-to-talk failed")
        finally:
            self._state("idle")
            with self._busy_lock:
                self._push_state = "idle"
        return text

    def _deliver(self, text: str) -> None:
        if not text:
            return
        try:
            self.on_query(text)
        except Exception:  # noqa: BLE001
            log.exception("on_query callback failed")

    def extract_after_wake(self, text: str) -> str | None:
        m = re.search(rf"\b{re.escape(self.wake_word)}\b", text, re.IGNORECASE)
        if not m:
            return None
        return text[m.end():].strip(" \t\n,.:;!?-")

    def _wake_loop(self) -> None:
        try:
            while not self._stop.is_set():
                if self._push_requested.is_set():
                    self._deliver(self._do_push())
                    continue
                if not self._should_listen():
                    self._push_requested.wait(0.2)
                    continue
                query = ""
                listened = False
                try:
                    audio = self._record(self.wake_chunk_seconds)
                    if self._push_requested.is_set():
                        continue  # push-to-talk wins; drop this chunk
                    heard = self._heard(audio)
                    # The avatar may have started talking mid-chunk; its own voice must not count as a wake word.
                    rest = self.extract_after_wake(heard) if heard and self._should_listen() else None
                    if rest is not None:
                        if not rest:
                            listened = True
                            self._state("listening")
                            rest = self._heard(self._record(self.record_seconds))
                        query = rest
                except Exception:  # noqa: BLE001
                    log.exception("wake listener iteration failed")
                    self._stop.wait(1.0)
                finally:
                    if listened:
                        self._state("idle")
                if query:
                    self._deliver(query)
        finally:
            with self._busy_lock:
                if self._push_state == "pending":
                    self._push_state = "idle"
            self._push_requested.clear()
