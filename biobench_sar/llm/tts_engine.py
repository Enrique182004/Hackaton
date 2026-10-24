from __future__ import annotations

import asyncio
import contextlib
import logging
import math
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
from biobench_sar.core.audio_trigger import guess_language, normalize_language

log = logging.getLogger(__name__)

MAX_FILES = 50
EDGE_TIMEOUT_S = 8.0
EDGE_RETRY_AFTER_S = 60.0
SAY_LIST_TIMEOUT_S = 5.0
# Voice speed (accessibility): 1.0 is each voice's own pace; set_rate clamps to this range.
MIN_RATE = 0.5
MAX_RATE = 2.0
# macOS `say` speaks about 175 words per minute unless told otherwise; its `-r` takes an absolute rate.
SAY_BASE_WPM = 175
ORDER = ["edge", "say", "pyttsx3", "none"]
LANGUAGES = ("en", "es")
# The assistant talks like a kid: Ana is Edge's child voice. Edge has no Mexican child voice, so Dalia is pitched up
# and sped up a little. Prosody belongs to the voice: any other voice a teacher picks plays as it is.
EDGE_VOICES = {"en": "en-US-AnaNeural", "es": "es-MX-DaliaNeural"}
EDGE_PROSODY = {"es-MX-DaliaNeural": ("+5%", "+30Hz")}  # voice -> (rate, pitch)
# Students are in El Paso / Chihuahua: Mexican Spanish first, any other Spanish only as a last resort.
SAY_ES_LOCALES = (("es_MX",), ("es_419", "es_US"), ("es_",))
SAY_ES_PREFERRED = "Paulina"
# The Eloquence voices sound robotic and most are novelty characters; Eddy is the plainest of them.
SAY_ELOQUENCE = ("Eddy", "Flo", "Grandma", "Grandpa", "Reed", "Rocko", "Sandy", "Shelley")
_SAY_LINE = re.compile(r"^(?P<name>\S.*?)\s+(?P<locale>[a-z]{2,3}_[A-Za-z0-9]+)\s+#")
_SPANISH = {"es", "spanish", "español", "espanol"}
_MEXICAN = {"mx", "mexico", "méxico", "paulina"}
_LATAM = {"419", "us", "latin", "latino", "latam"}
_ENGLISH = {"en", "english", "eng"}
_US = {"us", "usa", "america", "american"}
# macOS legacy and sound-effect voices (ids com.apple.speech.synthesis.voice.*), by id and display name.
_NOVELTY = {
    "albert",
    "badnews",
    "bahh",
    "bells",
    "boing",
    "bubbles",
    "cellos",
    "deranged",
    "wobble",
    "fred",
    "goodnews",
    "hysterical",
    "jester",
    "junior",
    "kathy",
    "organ",
    "princess",
    "superstar",
    "ralph",
    "trinoids",
    "whisper",
    "zarvox",
}
# Edge failures that mean "no network" rather than "bad voice": every voice cools down.
_ALL_VOICES = "*"


def _edge_factor(rate: str) -> float:
    """The speed factor of an edge-tts rate: "+5%" -> 1.05."""
    return 1 + int(rate.rstrip("%")) / 100


def edge_rate(factor: float) -> str:
    """edge-tts `rate` for a speed factor: a signed percentage, e.g. 0.75 -> "-25%", 1.25 -> "+25%"."""
    return f"{round((factor - 1) * 100):+d}%"


def parse_say_voices(output: str) -> list[tuple[str, str]]:
    """(name, locale) pairs from `say -v ?`; names may contain spaces, e.g. "Eddy (Spanish (Mexico))"."""
    return [(m["name"], m["locale"]) for m in map(_SAY_LINE.match, output.splitlines()) if m]


def _say_base_name(name: str) -> str:
    # macOS appends the language once several variants are installed: "Paulina (Spanish (Mexico))".
    return name.split(" (", 1)[0].strip()


def pick_say_voice(voices: list[tuple[str, str]], lang: str | None) -> str | None:
    """The `say -v` name (as listed) for Spanish: Paulina, other natural es_MX voices, Eddy, other Eloquence
    es_MX voices, then es_419/es_US, then any es_* voice. None for English (the system voice)."""
    if normalize_language(lang) != "es":
        return None
    best: str | None = None
    best_key: tuple[int, int] | None = None
    for name, locale in voices:
        tier = next((i for i, prefixes in enumerate(SAY_ES_LOCALES) if locale.startswith(prefixes)), None)
        if tier is None:
            continue
        base = _say_base_name(name)
        kind = 0 if base == SAY_ES_PREFERRED else 2 if base == "Eddy" else 3 if base in SAY_ELOQUENCE else 1
        if best_key is None or (tier, kind) < best_key:
            best, best_key = name, (tier, kind)
    return best


def _voice_tokens(voice: object) -> set[str]:
    parts = [str(getattr(voice, "id", "") or ""), str(getattr(voice, "name", "") or "")]
    for lang in getattr(voice, "languages", None) or []:
        # espeak reports languages as bytes with a priority prefix, e.g. b"\x05es".
        parts.append(lang.decode("utf-8", "ignore") if isinstance(lang, bytes) else str(lang))
    # Whole tokens only: a Windows SAPI id contains "...\Voices\Tokens\..." for every voice.
    return set(re.findall(r"[^\W_]+", " ".join(parts).lower()))


def _pyttsx3_rank(tokens: set[str], lang: str) -> tuple[int, int, int] | None:
    """(region, name, quality), lower is better; None if the voice doesn't speak `lang`."""
    quality = 2 if tokens & _NOVELTY else 1 if "eloquence" in tokens else 0
    if lang == "es":
        if not tokens & _SPANISH:
            return None
        region = 0 if tokens & _MEXICAN else 1 if tokens & _LATAM else 2
        return region, 0 if "paulina" in tokens else 1, quality
    if not tokens & _ENGLISH or tokens & _SPANISH:
        return None
    us = bool(tokens & _US) or {"united", "states"} <= tokens
    return 0 if us else 1, 0, quality


def pick_pyttsx3_voice(voices: Iterable[object], lang: str | None, default: object = None) -> str | None:
    """id of the best pyttsx3 voice for `lang` (None if none speaks it).

    `default` is the engine's current voice id: an English default is kept for English (the teacher chose
    it), and for Spanish it wins over a voice of the same region.
    """
    code = normalize_language(lang)
    if code not in LANGUAGES:
        return None
    ranked = []
    for index, voice in enumerate(voices):
        rank = _pyttsx3_rank(_voice_tokens(voice), code)
        if rank is not None:
            ranked.append((rank, index, getattr(voice, "id", None)))
    if not ranked:
        return None
    best = min(ranked)
    mine = next((rank for rank, _, voice_id in ranked if default is not None and voice_id == default), None)
    if mine is not None and (code == "en" or mine[0] == best[0][0]):
        return str(default)
    return best[2]


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


def _is_network_error(exc: BaseException) -> bool:
    return isinstance(exc, (OSError, asyncio.TimeoutError)) or type(exc).__module__.split(".")[0] == "aiohttp"


class TTSEngine:
    def __init__(
        self,
        backend: str = "edge",
        voice: str = EDGE_VOICES["en"],
        out_dir: Path | None = None,
        voices: dict[str, str] | None = None,
    ) -> None:
        self.backend = backend if backend in ORDER else "none"
        self.voice = voice
        # The single legacy voice (SAR_TTS_VOICE) serves its own language, and every speak() that names none.
        self._voice_lang = normalize_language(voice)
        self.voices = dict(EDGE_VOICES)
        if self._voice_lang in LANGUAGES:
            self.voices[self._voice_lang] = voice
        for key, value in (voices or {}).items():
            code = normalize_language(key)
            if code in LANGUAGES:
                self.voices[code] = value
            else:
                log.warning("ignoring TTS voice %r for unknown language %r", value, key)
        self.out_dir = Path(out_dir) if out_dir is not None else Settings().tts_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._proc: subprocess.Popen | None = None
        self._q: queue.Queue[tuple[str, str, float] | None] | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._edge_failed_at: dict[str, float] = {}
        self._say_voices: list[tuple[str, str]] | None = None
        self._say_picks: dict[str, str | None] = {}
        self._say_voices_lock = threading.Lock()
        # Voice speed factor (set_rate); at 1.0 no backend gets anything extra.
        self._rate = 1.0

    @property
    def default_language(self) -> str:
        """The language of a speak() call that names none: the legacy voice's, else English."""
        return self._voice_lang if self._voice_lang in LANGUAGES else "en"

    def set_rate(self, factor: float) -> None:
        """Voice speed for every later speak(): 0.75 slower, 1.25 faster, clamped to MIN_RATE..MAX_RATE."""
        rate = float(factor)
        self._rate = min(MAX_RATE, max(MIN_RATE, rate)) if math.isfinite(rate) else 1.0

    def speak(self, text: str, lang: str | None = None) -> str | None:
        """Say `text` in `lang` ("en"/"es"/a locale); "auto" or an unknown code guesses it from the text.

        Without `lang` the old single-voice behaviour holds: SAR_TTS_VOICE, whatever its language.
        """
        text = (text or "").strip()
        if not text:
            return None
        code = self._resolve(text, lang)
        for name in ORDER[ORDER.index(self.backend) :]:
            try:
                if name == "edge":
                    voice = self.edge_voice(code)
                    if self._edge_cooling_down(voice):
                        continue
                    return self._edge(text, voice)
                if name == "say" and self._say(text, code or self.default_language):
                    return None
                if name == "pyttsx3" and self._pyttsx3(text, code or self.default_language):
                    return None
            except Exception as exc:
                log.warning("tts backend %s failed: %s", name, exc)
        log.info("tts disabled/unavailable; text: %s", text)
        return None

    def _resolve(self, text: str, lang: str | None) -> str | None:
        if lang is None:
            return None
        code = normalize_language(lang)
        return code if code in LANGUAGES else guess_language(text, self.default_language)

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

    def edge_voice(self, lang: str | None = None) -> str:
        """edge-tts voice for "en"/"es"; for None or an unknown code, the legacy voice's slot."""
        code = normalize_language(lang)
        if code in LANGUAGES:
            return self.voices[code]
        return self.voices[self._voice_lang] if self._voice_lang in LANGUAGES else self.voice

    def say_voice(self, lang: str | None = None) -> str | None:
        """macOS `say` voice for `lang` (None = system default), from one cached `say -v ?`."""
        code = normalize_language(lang)
        if code not in LANGUAGES:
            code = self.default_language
        if code == "en":
            return None
        with self._say_voices_lock:
            if code not in self._say_picks:
                if self._say_voices is None:
                    self._say_voices = _list_say_voices()
                self._say_picks[code] = pick_say_voice(self._say_voices, code)
            return self._say_picks[code]

    def _edge_cooling_down(self, voice: str) -> bool:
        now = time.monotonic()
        failures = (self._edge_failed_at.get(_ALL_VOICES), self._edge_failed_at.get(voice))
        return any(at is not None and now - at < EDGE_RETRY_AFTER_S for at in failures)

    def _edge(self, text: str, voice: str) -> str:
        import edge_tts

        name = f"{uuid.uuid4().hex}.mp3"
        path = self.out_dir / name
        factor = self._rate
        # Offline venues: fail fast so the say/pyttsx3 fallback speaks instead of hanging the pipeline.
        try:
            own_rate, pitch = EDGE_PROSODY.get(voice, ("+0%", "+0Hz"))
            # The room's voice speed scales the voice's own pace (Dalia's +5% at 0.75 is -21%); at 1.0 it is unchanged.
            rate = edge_rate(_edge_factor(own_rate) * factor)
            # Inside the try: Communicate validates the voice and the rate as it is built.
            speech = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            asyncio.run(asyncio.wait_for(speech.save(str(path)), EDGE_TIMEOUT_S))
        except BaseException as exc:
            # A bad voice override must not silence edge for the other language.
            self._edge_failed_at[_ALL_VOICES if _is_network_error(exc) else voice] = time.monotonic()
            path.unlink(missing_ok=True)
            raise
        self._edge_failed_at.pop(voice, None)
        self._edge_failed_at.pop(_ALL_VOICES, None)
        self._cleanup()
        return name

    def _cleanup(self) -> None:
        files = sorted(self.out_dir.glob("*.mp3"), key=lambda p: p.stat().st_mtime)
        for old in files[: max(0, len(files) - MAX_FILES)]:
            with contextlib.suppress(OSError):
                old.unlink()

    def _say(self, text: str, lang: str = "en") -> bool:
        voice = self.say_voice(lang)
        rate = self._rate
        args = ["say", "-v", voice] if voice else ["say"]
        if rate != 1.0:
            args += ["-r", str(round(SAY_BASE_WPM * rate))]
        args.append(text)
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
        q.put((text, lang, self._rate))
        return True

    @staticmethod
    def _pyttsx3_worker(q: queue.Queue) -> None:
        try:
            import pyttsx3

            engine = pyttsx3.init()
        except Exception:
            log.exception("pyttsx3 init failed")
            return
        # espeak reports no voice until one is set, so English needs a picked voice there too.
        default = _property(engine, "voice")
        # The engine's own words per minute, read once: a voice speed scales it and 1.0 puts it back.
        base_rate = _property(engine, "rate")
        chosen: dict[str, Any] = {}
        current = default
        current_rate = 1.0
        while True:
            item = q.get()
            if item is None:
                return
            text, lang, rate = item
            try:
                if lang not in chosen:
                    voices = _property(engine, "voices") or []
                    chosen[lang] = pick_pyttsx3_voice(voices, lang, default) or default
                voice = chosen[lang]
                if voice is not None and voice != current:
                    engine.setProperty("voice", voice)
                    current = voice
            except Exception:
                log.exception("pyttsx3 voice switch failed")
            if rate != current_rate and isinstance(base_rate, (int, float)):
                try:
                    # Whole words per minute: the espeak driver hands the rate to C as an int.
                    engine.setProperty("rate", base_rate if rate == 1.0 else round(base_rate * rate))
                    current_rate = rate
                except Exception:
                    log.exception("pyttsx3 rate change failed")
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                log.exception("pyttsx3 speak failed")
