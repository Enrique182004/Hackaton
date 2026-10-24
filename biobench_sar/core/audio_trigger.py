from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
WAKE_CHUNK_SECONDS = 3.0
SILENCE_RMS = 0.01
# Explicit queries (push-to-talk, wake follow-up) are only dropped when the mic delivered digital zeros:
# a quiet speaker must still be heard.
DIGITAL_SILENCE_RMS = 1e-4
# Trimming a wake chunk by wall clock lags the audio a little; keep some of what came just before the click.
PUSH_PREROLL_SECONDS = 0.5
LANGUAGES = ("en", "es")
# Spanish speakers say the wake word in Spanish.
WAKE_ALIASES = {"assistant": ("asistente",)}


def _normalize_language(code: str | None) -> str:
    c = (code or "").strip().lower().replace("_", "-")
    if c in ("en", "english", "inglés", "ingles") or c.startswith("en-"):
        return "en"
    if c in ("es", "spanish", "español", "espanol") or c.startswith("es-"):
        return "es"
    return "auto"


def _guess_language(text: str, default: str = "en") -> str:
    """Text heuristic from biobench_sar.lang; "en" when that module isn't installed."""
    try:
        from biobench_sar.lang import detect_language
    except ImportError:
        return "en"
    try:
        guess = detect_language(text, default=default)
    except Exception:
        log.exception("language detection failed")
        return default
    return guess if guess in LANGUAGES else default


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
        language: str = "auto",
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
        self.language = _normalize_language(language)
        # Language of the last non-empty transcript or typed query: the reply should use it.
        self.last_language = self.language if self.language in LANGUAGES else "en"
        self._model = None
        self._model_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._busy_lock = threading.Lock()
        # idle -> pending (requested, waiting for the wake loop) -> recording -> idle
        self._push_state = "idle"
        self._push_requested = threading.Event()
        self._push_requested_at = 0.0
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
        except Exception as exc:
            log.info("audio unavailable: %s", exc)
            return False

    def _state(self, name: str) -> None:
        if self.on_state is None:
            return
        try:
            self.on_state(name)
        except Exception:
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
            self._push_requested_at = time.monotonic()
            # Under the lock so it can never land after this push's own "idle".
            self._state("listening")
            wake_running = self.mode == "wake" and self._wake_thread is not None and self._wake_thread.is_alive()
            if wake_running:
                self._push_requested.set()
                return True
        threading.Thread(target=self._run_push, daemon=True, name="push-to-talk").start()
        return True

    def set_language(self, code: str | None) -> str:
        """Force "en"/"es" (Whisper decodes in it, replies use it) or "auto" to follow the student."""
        self.language = _normalize_language(code)
        if self.language in LANGUAGES:
            self.last_language = self.language
        return self.language

    def submit_text(self, text: str) -> None:
        text = (text or "").strip()
        if text:
            self._note_language(text, None)
            self.on_query(text)

    def _note_language(self, text: str, detected: object) -> None:
        if not text:
            return  # silence/warm-up says nothing about the student's language
        if self.language in LANGUAGES:
            self.last_language = self.language
        elif isinstance(detected, str) and detected.lower() in LANGUAGES:
            self.last_language = detected.lower()
        else:
            self.last_language = _guess_language(text, self.last_language)

    def transcribe(self, audio: np.ndarray) -> str:
        # Whisper hangs a kv-cache on the shared model for each decode, so concurrent decodes (warm-up,
        # wake chunks, push-to-talk) corrupt each other. An injected transcriber gets the same guarantee.
        if self._transcriber is not None:
            with self._infer_lock:
                text = self._transcriber(audio).strip()
                self._note_language(text, None)
            return text
        with self._model_lock:
            if self._model is None:
                import whisper

                self._model = whisper.load_model(self.whisper_model)
            model = self._model
        language = self.language
        with self._infer_lock:
            result = model.transcribe(
                np.asarray(audio, dtype=np.float32),
                fp16=False,
                language=language if language in LANGUAGES else None,  # None: Whisper detects it
            )
            text = str(result.get("text", "")).strip()
            self._note_language(text, result.get("language"))
        return text

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

    def _heard(self, audio: np.ndarray, min_rms: float) -> str:
        """Transcribe unless the clip's RMS is below min_rms (saves CPU; Whisper hallucinates text on silence)."""
        arr = np.asarray(audio, dtype=np.float32)
        if arr.size == 0 or float(np.sqrt(np.mean(np.square(arr)))) < min_rms:
            return ""
        return self.transcribe(arr)

    def _since_request(self, chunk: np.ndarray) -> np.ndarray:
        """The part of a wake chunk recorded after push-to-talk was requested."""
        arr = np.asarray(chunk, dtype=np.float32)
        seconds = time.monotonic() - self._push_requested_at + PUSH_PREROLL_SECONDS
        return arr[max(0, arr.size - int(seconds * SAMPLE_RATE)) :]

    def _run_push(self) -> None:
        self._deliver(self._do_push())

    def _do_push(self, lead: np.ndarray | None = None) -> str:
        """Record and transcribe one push-to-talk query; `lead` is audio already captured since the request."""
        with self._busy_lock:
            self._push_state = "recording"
        self._push_requested.clear()
        text = ""
        try:
            audio = np.asarray(self._record(self.record_seconds), dtype=np.float32)
            if lead is not None:
                audio = np.concatenate([lead, audio])
            self._state("transcribing")
            text = self._heard(audio, DIGITAL_SILENCE_RMS)
        except Exception:
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
        except Exception:
            log.exception("on_query callback failed")

    def extract_after_wake(self, text: str) -> str | None:
        words = (self.wake_word, *WAKE_ALIASES.get(self.wake_word.lower(), ()))
        m = re.search(rf"\b(?:{'|'.join(map(re.escape, words))})\b", text, re.IGNORECASE)
        if not m:
            return None
        return text[m.end() :].strip(" \t\n,.:;!?¿¡-")

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
                        # Push-to-talk wins, but the user may already be talking: keep what the chunk caught.
                        query = self._do_push(self._since_request(audio))
                    else:
                        heard = self._heard(audio, self.silence_rms)
                        # The avatar may have started talking mid-chunk; its own voice must not count as a wake word.
                        rest = self.extract_after_wake(heard) if heard and self._should_listen() else None
                        if rest is not None:
                            if not rest:
                                listened = True
                                self._state("listening")
                                rest = self._heard(self._record(self.record_seconds), DIGITAL_SILENCE_RMS)
                            query = rest
                except Exception:
                    log.exception("wake listener iteration failed")
                    self._stop.wait(1.0)
                finally:
                    if listened:
                        with self._busy_lock:
                            # A push requested meanwhile already announced "listening"; don't clobber it.
                            if self._push_state == "idle":
                                self._state("idle")
                if query:
                    self._deliver(query)
        finally:
            with self._busy_lock:
                if self._push_state == "pending":
                    self._push_state = "idle"
                    self._state("idle")
            self._push_requested.clear()
