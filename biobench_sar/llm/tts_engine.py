from __future__ import annotations

import asyncio
import contextlib
import logging
import queue
import re
import subprocess
import threading
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from biobench_sar.config import Settings

log = logging.getLogger(__name__)

MAX_FILES = 50
EDGE_TIMEOUT_S = 8.0
EDGE_RETRY_AFTER_S = 60.0
SAY_LIST_TIMEOUT_S = 5.0
ORDER = ["edge", "say", "pyttsx3", "none"]
LANGUAGES = ("en", "es")
EDGE_VOICES = {"en": "en-US-AriaNeural", "es": "es-MX-DaliaNeural"}
# Students are in El Paso / Chihuahua: Mexican Spanish first, any other Spanish only as a last resort.
SAY_ES_PREFERRED = ("Paulina", "Eddy (Spanish (Mexico))")
SAY_ES_LOCALES = (("es_MX",), ("es_419", "es_US"), ("es_",))
_SAY_LINE = re.compile(r"^(?P<name>\S.*?)\s+(?P<locale>[a-z]{2,3}_[A-Za-z0-9]+)\s+#")
_SPANISH = {"es", "spanish", "español", "espanol"}
_MEXICAN = {"mx", "mexico", "méxico", "paulina"}
_LATAM = {"419", "us", "latin", "latino", "latam"}


def _lang(code: str | None) -> str:
    c = (code or "").strip().lower()[:2]
    return c if c in LANGUAGES else "en"


def parse_say_voices(output: str) -> list[tuple[str, str]]:
    """(name, locale) pairs from `say -v ?`; names may contain spaces, e.g. "Eddy (Spanish (Mexico))"."""
    return [(m["name"], m["locale"]) for m in map(_SAY_LINE.match, output.splitlines()) if m]


def pick_say_voice(voices: list[tuple[str, str]], lang: str) -> str | None:
    if _lang(lang) != "es":
        return None
    names = {name for name, _ in voices}
    for preferred in SAY_ES_PREFERRED:
        if preferred in names:
            return preferred
    for prefixes in SAY_ES_LOCALES:
        for name, locale in voices:
            if locale.startswith(prefixes):
                return name
    return None


def _voice_tokens(voice: object) -> set[str]:
    parts = [str(getattr(voice, "id", "") or ""), str(getattr(voice, "name", "") or "")]
    for lang in getattr(voice, "languages", None) or []:
        # espeak reports languages as bytes with a priority prefix, e.g. b"\x05es".
        parts.append(lang.decode("utf-8", "ignore") if isinstance(lang, bytes) else str(lang))
    # Whole tokens only: a Windows SAPI id contains "...\Voices\Tokens\..." for every voice.
    return set(re.findall(r"[^\W_]+", " ".join(parts).lower()))


def pick_pyttsx3_voice(voices: Iterable[object], lang: str) -> str | None:
    """id of the best pyttsx3 voice for `lang`; None keeps the system default."""
    if _lang(lang) != "es":
        return None
    best: str | None = None
    best_rank = 3
    for voice in voices:
        tokens = _voice_tokens(voice)
        if not tokens & _SPANISH:
            continue
        rank = 0 if tokens & _MEXICAN else 1 if tokens & _LATAM else 2
        if rank < best_rank:
            best, best_rank = getattr(voice, "id", None), rank
    return best


def _list_say_voices() -> list[tuple[str, str]]:
    try:
        out = subprocess.run(
            ["say", "-v", "?"],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=SAY_LIST_TIMEOUT_S,
            check=False,
        ).stdout
    except Exception as exc:
        log.info("could not list macOS say voices: %s", exc)
        return []
    return parse_say_voices(out or "")


def _property(engine: Any, name: str) -> Any:
    try:
        return engine.getProperty(name)
    except Exception:
        return None


class TTSEngine:
    def __init__(
        self,
        backend: str = "edge",
        voice: str = "en-US-AriaNeural",
        out_dir: Path | None = None,
        voices: dict[str, str] | None = None,
    ) -> None:
        self.backend = backend if backend in ORDER else "none"
        self.voice = voice
        # The single legacy voice (SAR_TTS_VOICE) belongs to the language its name starts with.
        self.voices = {**EDGE_VOICES, _lang(voice): voice, **{_lang(k): v for k, v in (voices or {}).items()}}
        self.out_dir = Path(out_dir) if out_dir is not None else Settings().tts_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._proc: subprocess.Popen | None = None
        self._q: queue.Queue[tuple[str, str] | None] | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._edge_failed_at: float | None = None
        self._say_voices: list[tuple[str, str]] | None = None
        self._say_voices_lock = threading.Lock()

    def speak(self, text: str, lang: str = "en") -> str | None:
        text = (text or "").strip()
        if not text:
            return None
        lang = _lang(lang)
        for name in ORDER[ORDER.index(self.backend) :]:
            try:
                if name == "edge":
                    if self._edge_cooling_down():
                        continue
                    return self._edge(text, lang)
                if name == "say" and self._say(text, lang):
                    return None
                if name == "pyttsx3" and self._pyttsx3(text, lang):
                    return None
            except Exception as exc:
                log.warning("tts backend %s failed: %s", name, exc)
        log.info("tts disabled/unavailable; text: %s", text)
        return None

    def stop(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(Exception):
                proc.terminate()
        if self._q is not None:
            self._q.put(None)
            self._q = None
            self._thread = None

    def edge_voice(self, lang: str = "en") -> str:
        lang = _lang(lang)
        return self.voices.get(lang) or EDGE_VOICES[lang]

    def say_voice(self, lang: str = "en") -> str | None:
        """macOS `say` voice for `lang` (None = system default), from one cached `say -v ?`."""
        if _lang(lang) == "en":
            return None
        with self._say_voices_lock:
            if self._say_voices is None:
                self._say_voices = _list_say_voices()
            voices = self._say_voices
        return pick_say_voice(voices, lang)

    def _edge_cooling_down(self) -> bool:
        failed_at = self._edge_failed_at
        return failed_at is not None and time.monotonic() - failed_at < EDGE_RETRY_AFTER_S

    def _edge(self, text: str, lang: str = "en") -> str:
        import edge_tts

        name = f"{uuid.uuid4().hex}.mp3"
        path = self.out_dir / name
        voice = self.edge_voice(lang)
        # Offline venues: fail fast so the say/pyttsx3 fallback speaks instead of hanging the pipeline.
        try:
            asyncio.run(asyncio.wait_for(edge_tts.Communicate(text, voice).save(str(path)), EDGE_TIMEOUT_S))
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
            with contextlib.suppress(OSError):
                old.unlink()

    def _say(self, text: str, lang: str = "en") -> bool:
        voice = self.say_voice(lang)
        args = ["say", "-v", voice, text] if voice else ["say", text]
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()
            proc = subprocess.Popen(args)
            self._proc = proc
        return True

    def _pyttsx3(self, text: str, lang: str = "en") -> bool:
        import pyttsx3  # noqa: F401  (fail fast if missing)

        with self._lock:
            if self._q is None:
                self._q = queue.Queue()
                self._thread = threading.Thread(
                    target=self._pyttsx3_worker, args=(self._q,), daemon=True, name="pyttsx3"
                )
                self._thread.start()
            q = self._q
        q.put((text, lang))
        return True

    @staticmethod
    def _pyttsx3_worker(q: queue.Queue) -> None:
        try:
            import pyttsx3

            engine = pyttsx3.init()
        except Exception:
            log.exception("pyttsx3 init failed")
            return
        default = _property(engine, "voice")
        chosen: dict[str, Any] = {"en": default}
        current = default
        while True:
            item = q.get()
            if item is None:
                return
            text, lang = item
            try:
                if lang not in chosen:
                    chosen[lang] = pick_pyttsx3_voice(_property(engine, "voices") or [], lang) or default
                voice = chosen[lang]
                if voice is not None and voice != current:
                    engine.setProperty("voice", voice)
                    current = voice
            except Exception:
                log.exception("pyttsx3 voice switch failed")
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                log.exception("pyttsx3 speak failed")
