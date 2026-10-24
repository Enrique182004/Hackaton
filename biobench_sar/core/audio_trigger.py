from __future__ import annotations

import logging
import re
import threading
import time
import unicodedata
from collections.abc import Callable
from typing import Any

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
_EN_NAMES = frozenset({"en", "eng", "english", "ingles"})
_ES_NAMES = frozenset({"es", "spa", "spanish", "espanol", "castellano"})


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


def normalize_language(code: str | None) -> str:
    """The setting for a code, locale or name: "es-MX" / "Español" -> "es", "English" -> "en", else "auto".

    The one normalizer of the voice layer (TTSEngine uses it too); biobench_sar.lang's when installed.
    """
    try:
        from biobench_sar.lang import normalize_language as shared
    except ImportError:
        pass
    else:
        return str(shared(code))
    key = _strip_accents((code or "").strip().lower()).replace("_", "-").split("-")[0]
    if key in _EN_NAMES:
        return "en"
    if key in _ES_NAMES:
        return "es"
    return "auto"


def guess_language(text: str, default: str = "en") -> str:
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
    """Push-to-talk / wake-word speech input, transcribed by Whisper in English or Spanish.

    language="auto" (the default) runs one Whisper language-ID pass, limited to en/es, before each decode.
    That costs what Whisper's own detection costs, but it is slower than a forced language (about 3.5 s vs
    2.2 s per clip for `base` on an M-series Mac, and wake mode is deaf meanwhile): English-only rooms
    should force "en". Whisper `base` garbles one-word and Spanglish utterances ("ayuda", "sí",
    "¿cómo hago this one?") in every mode, so ask students for full sentences, or use whisper_model="small"
    in Spanish-heavy classrooms.

    last_language is the language of the query being delivered (read it inside on_query) and, afterwards,
    of the last query delivered. Transcripts that never become a query (wake chunks without the wake word,
    the avatar's own voice, warm-up) leave it alone. on_query may return False to refuse a query (busy):
    last_language then goes back. Another thread reading it later may already see a newer query's language,
    so snapshot it inside on_query.
    """

    def __init__(
        self,
        on_query: Callable[[str], Any],
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
        self.language = normalize_language(language)
        self.last_language = self.language if self.language in LANGUAGES else "en"
        # One delivery at a time, so last_language stays on the query on_query is handling. Reentrant
        # because on_query may call set_language.
        self._deliver_lock = threading.RLock()
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
        with self._deliver_lock:
            self.language = normalize_language(code)
            if self.language in LANGUAGES:
                self.last_language = self.language
            return self.language

    def submit_text(self, text: str) -> None:
        text = (text or "").strip()
        if text:
            self._deliver(text, self._language_of(text, None), swallow=False)

    def _language_of(self, text: str, decoded: object) -> str:
        if isinstance(decoded, str) and decoded.lower() in LANGUAGES:
            return decoded.lower()
        if self.language in LANGUAGES:
            return self.language
        return guess_language(text, self.last_language)

    def transcribe(self, audio: np.ndarray) -> str:
        """The transcript only: last_language changes when a query is delivered, not here."""
        return self._transcribe(audio)[0]

    def _transcribe(self, audio: np.ndarray) -> tuple[str, str]:
        """(text, its language "en"/"es")."""
        # Whisper hangs a kv-cache on the shared model for each decode, so concurrent decodes (warm-up,
        # wake chunks, push-to-talk) corrupt each other. An injected transcriber gets the same guarantee.
        if self._transcriber is not None:
            with self._infer_lock:
                text = self._transcriber(audio).strip()
            return text, self._language_of(text, None)
        with self._model_lock:
            if self._model is None:
                import whisper

                self._model = whisper.load_model(self.whisper_model)
            model = self._model
        arr = np.asarray(audio, dtype=np.float32)
        forced = self.language
        with self._infer_lock:
            language = forced if forced in LANGUAGES else self._identify_language(model, arr)
            result = model.transcribe(arr, fp16=False, language=language)
        text = str(result.get("text", "")).strip()
        return text, self._language_of(text, language or result.get("language"))

    def _identify_language(self, model: Any, audio: np.ndarray) -> str | None:
        """Whichever of en/es Whisper rates likelier; None lets transcribe pick from all 99 languages.

        Left to choose freely, Whisper hears short Mexican-Spanish phrases as Portuguese, Italian or Korean
        and writes the transcript in that language. Call with the inference lock held.
        """
        if not getattr(model, "is_multilingual", True):
            return "en"
        try:
            import whisper

            n_mels = getattr(getattr(model, "dims", None), "n_mels", 80)
            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio), n_mels=n_mels, device=model.device)
            _, probs = model.detect_language(mel)
        except Exception:
            log.exception("Whisper language ID failed; letting transcribe detect the language")
            return None
        if isinstance(probs, list):
            probs = probs[0] if probs else {}
        last = self.last_language
        return max(LANGUAGES, key=lambda code: (float(probs.get(code, 0.0)), code == last))

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

    def _heard(self, audio: np.ndarray, min_rms: float) -> tuple[str, str]:
        """Transcribe unless the clip's RMS is below min_rms (saves CPU; Whisper hallucinates text on silence)."""
        arr = np.asarray(audio, dtype=np.float32)
        if arr.size == 0 or float(np.sqrt(np.mean(np.square(arr)))) < min_rms:
            return "", self.last_language
        return self._transcribe(arr)

    def _since_request(self, chunk: np.ndarray) -> np.ndarray:
        """The part of a wake chunk recorded after push-to-talk was requested."""
        arr = np.asarray(chunk, dtype=np.float32)
        seconds = time.monotonic() - self._push_requested_at + PUSH_PREROLL_SECONDS
        return arr[max(0, arr.size - int(seconds * SAMPLE_RATE)) :]

    def _run_push(self) -> None:
        self._deliver(*self._do_push())

    def _do_push(self, lead: np.ndarray | None = None) -> tuple[str, str]:
        """Record and transcribe one push-to-talk query; `lead` is audio already captured since the request."""
        with self._busy_lock:
            self._push_state = "recording"
        self._push_requested.clear()
        text, language = "", self.last_language
        try:
            audio = np.asarray(self._record(self.record_seconds), dtype=np.float32)
            if lead is not None:
                audio = np.concatenate([lead, audio])
            self._state("transcribing")
            text, language = self._heard(audio, DIGITAL_SILENCE_RMS)
        except Exception:
            log.exception("push-to-talk failed")
        finally:
            self._state("idle")
            with self._busy_lock:
                self._push_state = "idle"
        return text, language

    def _deliver(self, text: str, language: str, swallow: bool = True) -> None:
        if not text:
            return
        with self._deliver_lock:
            previous, self.last_language = self.last_language, language
            accepted: object = False
            try:
                accepted = self.on_query(text)
            except Exception:
                if not swallow:
                    raise
                log.exception("on_query callback failed")
            finally:
                if accepted is False:
                    self.last_language = previous

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
                    self._deliver(*self._do_push())
                    continue
                if not self._should_listen():
                    self._push_requested.wait(0.2)
                    continue
                query, language = "", self.last_language
                listened = False
                try:
                    audio = self._record(self.wake_chunk_seconds)
                    if self._push_requested.is_set():
                        # Push-to-talk wins, but the user may already be talking: keep what the chunk caught.
                        query, language = self._do_push(self._since_request(audio))
                    else:
                        heard, language = self._heard(audio, self.silence_rms)
                        # The avatar may have started talking mid-chunk; its own voice must not count as a wake word.
                        rest = self.extract_after_wake(heard) if heard and self._should_listen() else None
                        if rest is not None:
                            if not rest:
                                listened = True
                                self._state("listening")
                                rest, language = self._heard(self._record(self.record_seconds), DIGITAL_SILENCE_RMS)
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
                    self._deliver(query, language)
        finally:
            with self._busy_lock:
                if self._push_state == "pending":
                    self._push_state = "idle"
                    self._state("idle")
            self._push_requested.clear()
