from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import time
import uuid
from pathlib import Path

from biobench_sar.config import Settings
from biobench_sar.core.audio_trigger import guess_language, normalize_language

log = logging.getLogger(__name__)

MAX_FILES = 50
EDGE_TIMEOUT_S = 8.0
EDGE_RETRY_AFTER_S = 60.0
# Voice speed (accessibility): 1.0 is each voice's own pace; set_rate clamps to this range.
MIN_RATE = 0.5
MAX_RATE = 2.0
# Only the kid voices exist, and they come from Microsoft's online voice service (edge-tts). The computer's own voices
# (macOS `say`, pyttsx3) are adult voices, so there is no offline voice: without the service an answer is shown only.
BACKENDS = ("edge", "none")
LANGUAGES = ("en", "es")
# The assistant talks like a kid: Ana is Edge's child voice. Edge has no Mexican child voice, so Dalia is pitched up
# and sped up a little. Prosody belongs to the voice: any other voice a teacher picks plays as it is.
EDGE_VOICES = {"en": "en-US-AnaNeural", "es": "es-MX-DaliaNeural"}
# +30 Hz: at +15 Hz she sounded like the old adult voice (chosen by ear against +15, +25 and +40 Hz).
EDGE_PROSODY = {"es-MX-DaliaNeural": ("+5%", "+30Hz")}  # voice -> (rate, pitch)
# Edge failures that mean "no network" rather than "bad voice": every voice cools down.
_ALL_VOICES = "*"


def _edge_factor(rate: str) -> float:
    """The speed factor of an edge-tts rate: "+5%" -> 1.05."""
    return 1 + int(rate.rstrip("%")) / 100


def edge_rate(factor: float) -> str:
    """edge-tts `rate` for a speed factor: a signed percentage, e.g. 0.75 -> "-25%", 1.25 -> "+25%"."""
    return f"{round((factor - 1) * 100):+d}%"


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
        self.backend = backend if backend in BACKENDS else "none"
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
        self._edge_failed_at: dict[str, float] = {}
        # Voice speed factor (set_rate); at 1.0 the voices get nothing extra.
        self._rate = 1.0

    @property
    def default_language(self) -> str:
        """The language of a speak() call that names none: the legacy voice's, else English."""
        return self._voice_lang if self._voice_lang in LANGUAGES else "en"

    @property
    def offline(self) -> bool:
        """The voice service failed with a network error in the last EDGE_RETRY_AFTER_S: answers are shown, not spoken."""
        at = self._edge_failed_at.get(_ALL_VOICES)
        return at is not None and time.monotonic() - at < EDGE_RETRY_AFTER_S

    def set_rate(self, factor: float) -> None:
        """Voice speed for every later speak(): 0.75 slower, 1.25 faster, clamped to MIN_RATE..MAX_RATE."""
        rate = float(factor)
        self._rate = min(MAX_RATE, max(MIN_RATE, rate)) if math.isfinite(rate) else 1.0

    def speak(self, text: str, lang: str | None = None) -> str | None:
        """The mp3 file name of `text` said in `lang` ("en"/"es"/a locale; "auto" or an unknown code guesses it from
        the text), or None when nothing is spoken: the voice is off, or the voice service can't be reached.

        Without `lang` the old single-voice behaviour holds: SAR_TTS_VOICE, whatever its language.
        """
        text = (text or "").strip()
        if not text or self.backend != "edge":
            return None
        voice = self.edge_voice(self._resolve(text, lang))
        if self._edge_cooling_down(voice):
            return None
        try:
            return self._edge(text, voice)
        except Exception as exc:
            log.warning("the voice service failed (%s): this answer is shown but not spoken", exc)
            return None

    def _resolve(self, text: str, lang: str | None) -> str | None:
        if lang is None:
            return None
        code = normalize_language(lang)
        return code if code in LANGUAGES else guess_language(text, self.default_language)

    def edge_voice(self, lang: str | None = None) -> str:
        """edge-tts voice for "en"/"es"; for None or an unknown code, the legacy voice's slot."""
        code = normalize_language(lang)
        if code in LANGUAGES:
            return self.voices[code]
        return self.voices[self._voice_lang] if self._voice_lang in LANGUAGES else self.voice

    def _edge_cooling_down(self, voice: str) -> bool:
        now = time.monotonic()
        failures = (self._edge_failed_at.get(_ALL_VOICES), self._edge_failed_at.get(voice))
        return any(at is not None and now - at < EDGE_RETRY_AFTER_S for at in failures)

    def _edge(self, text: str, voice: str) -> str:
        import edge_tts

        name = f"{uuid.uuid4().hex}.mp3"
        path = self.out_dir / name
        factor = self._rate
        # Offline venues: fail fast, so the answer is shown at once instead of hanging the pipeline.
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
