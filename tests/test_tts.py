from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path
from typing import ClassVar

import pytest

from biobench_sar.llm import tts_engine
from biobench_sar.llm.tts_engine import TTSEngine


class FakeComm:
    fail = False
    fail_voices: ClassVar[set[str]] = set()
    error: ClassVar[type[BaseException]] = RuntimeError
    calls = 0
    partial = False
    voices: ClassVar[list[str]] = []
    prosody: ClassVar[list[tuple[str, str]]] = []

    def __init__(self, text: str, voice: str, rate: str = "+0%", pitch: str = "+0Hz") -> None:
        self.text, self.voice = text, voice
        FakeComm.voices.append(voice)
        FakeComm.prosody.append((rate, pitch))

    async def save(self, path: str) -> None:
        FakeComm.calls += 1
        if FakeComm.partial:
            Path(path).write_bytes(b"")
        if FakeComm.fail or self.voice in FakeComm.fail_voices:
            raise FakeComm.error("offline")
        Path(path).write_bytes(b"mp3")


@pytest.fixture(autouse=True)
def fake_edge(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeComm.fail = False
    FakeComm.fail_voices = set()
    FakeComm.error = RuntimeError
    FakeComm.calls = 0
    FakeComm.partial = False
    FakeComm.voices = []
    FakeComm.prosody = []
    mod = types.ModuleType("edge_tts")
    mod.Communicate = FakeComm  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "edge_tts", mod)


def test_edge_writes_file(tmp_path: Path) -> None:
    eng = TTSEngine("edge", out_dir=tmp_path / "sub")
    name = eng.speak("hello")
    assert name and name.endswith(".mp3")
    assert (tmp_path / "sub" / name).read_bytes() == b"mp3"


def test_none_backend_and_empty_text(tmp_path: Path) -> None:
    assert TTSEngine("none", out_dir=tmp_path).speak("x") is None
    assert TTSEngine("edge", out_dir=tmp_path).speak("  ") is None
    assert list(tmp_path.glob("*.mp3")) == []


def test_cleanup_keeps_newest(tmp_path: Path) -> None:
    for i in range(60):
        f = tmp_path / f"old{i:02d}.mp3"
        f.write_bytes(b"x")
        os.utime(f, (1000 + i, 1000 + i))
    name = TTSEngine("edge", out_dir=tmp_path).speak("new")
    files = {p.name for p in tmp_path.glob("*.mp3")}
    assert len(files) == tts_engine.MAX_FILES
    assert name in files
    assert "old00.mp3" not in files and "old59.mp3" in files


@pytest.fixture(autouse=True)
def no_local_voice(monkeypatch: pytest.MonkeyPatch) -> None:
    """The computer's own voices are adult voices: nothing may start `say` or load pyttsx3."""

    def refuse(*args: object, **kwargs: object) -> None:
        pytest.fail(f"a local voice player was started: {args}")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setitem(sys.modules, "pyttsx3", None)  # importing it raises ImportError


def test_without_edge_the_answer_is_only_shown(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # Only the kid voices exist: offline, nothing is spoken (the page still shows the text).
    FakeComm.fail = True
    eng = TTSEngine("edge", out_dir=tmp_path)
    assert eng.speak("hi there", "en") is None
    assert eng.speak("hola", "es") is None
    assert list(tmp_path.glob("*.mp3")) == []
    assert "shown but not spoken" in caplog.text


def test_edge_failure_is_remembered(tmp_path: Path) -> None:
    FakeComm.fail = True
    eng = TTSEngine("edge", out_dir=tmp_path)
    assert eng.speak("one") is None
    assert eng.speak("two") is None
    assert FakeComm.calls == 1


def test_only_the_kid_voice_service_or_none() -> None:
    assert tts_engine.BACKENDS == ("edge", "none")
    for removed in ("say", "pyttsx3", "nonsense"):
        assert TTSEngine(removed).backend == "none"


def test_edge_retried_after_cooldown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeComm.fail = True
    now = {"t": 1000.0}
    monkeypatch.setattr(tts_engine.time, "monotonic", lambda: now["t"])
    eng = TTSEngine("edge", out_dir=tmp_path)
    eng.speak("one")
    now["t"] += tts_engine.EDGE_RETRY_AFTER_S - 1
    eng.speak("two")
    assert FakeComm.calls == 1
    now["t"] += 2
    FakeComm.fail = False
    name = eng.speak("three")
    assert FakeComm.calls == 2 and name and name.endswith(".mp3")
    FakeComm.fail = True  # a success clears the cooldown state
    eng.speak("four")
    assert FakeComm.calls == 3


def test_edge_failure_removes_partial_mp3(tmp_path: Path) -> None:
    FakeComm.fail = True
    FakeComm.partial = True
    TTSEngine("edge", out_dir=tmp_path).speak("x")
    assert FakeComm.calls == 1
    assert list(tmp_path.glob("*.mp3")) == []


@pytest.fixture
def no_lang_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "biobench_sar.lang", None)  # import raises ImportError


def _fake_lang_by_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spanish for text with ¿ ¡ ñ or an accent, English otherwise."""
    mod = types.ModuleType("biobench_sar.lang")
    mod.detect_language = lambda text, default="en": "es" if set(text) & set("¿¡ñáéíóú") else "en"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "biobench_sar.lang", mod)


def test_edge_voice_follows_the_language(tmp_path: Path, no_lang_module: None) -> None:
    eng = TTSEngine("edge", out_dir=tmp_path)
    eng.speak("hello")
    eng.speak("hola", "es")
    eng.speak("hola", lang="es-MX")
    eng.speak("hola", "Español")
    eng.speak("bonjour", "fr")  # unsupported: guessed from the text, English without the lang module
    assert FakeComm.voices == [
        "en-US-AnaNeural",
        "es-MX-DaliaNeural",
        "es-MX-DaliaNeural",
        "es-MX-DaliaNeural",
        "en-US-AnaNeural",
    ]


def test_default_voices_sound_like_a_kid(tmp_path: Path) -> None:
    # Ana is a child voice; Edge has no Mexican child voice, so Dalia is pitched up and a little faster.
    # +30 Hz: at +15 Hz she sounded like the old adult voice (chosen by ear against +15, +25 and +40 Hz).
    eng = TTSEngine("edge", out_dir=tmp_path)
    eng.speak("hello", "en")
    eng.speak("hola", "es")
    assert FakeComm.voices == ["en-US-AnaNeural", "es-MX-DaliaNeural"]
    assert FakeComm.prosody == [("+0%", "+0Hz"), ("+5%", "+30Hz")]


def test_other_voices_keep_their_own_sound(tmp_path: Path) -> None:
    eng = TTSEngine("edge", "en-GB-SoniaNeural", tmp_path, voices={"es": "es-MX-JorgeNeural"})
    eng.speak("hi", "en")
    eng.speak("hola", "es")
    assert FakeComm.prosody == [("+0%", "+0Hz"), ("+0%", "+0Hz")]


def test_edge_voices_can_be_overridden(tmp_path: Path) -> None:
    eng = TTSEngine("edge", "en-GB-SoniaNeural", tmp_path, voices={"es": "es-MX-JorgeNeural"})
    eng.speak("hi", "en")
    eng.speak("hola", "es")
    eng.speak("hi")
    assert FakeComm.voices == ["en-GB-SoniaNeural", "es-MX-JorgeNeural", "en-GB-SoniaNeural"]
    assert eng.voice == "en-GB-SoniaNeural"


def test_voice_languages_use_the_shared_normalizer(tmp_path: Path) -> None:
    eng = TTSEngine("edge", out_dir=tmp_path, voices={"spanish": "es-MX-JorgeNeural"})
    assert eng.voices == {"en": "en-US-AnaNeural", "es": "es-MX-JorgeNeural"}
    # Unknown keys are ignored instead of replacing the English voice.
    eng = TTSEngine("edge", out_dir=tmp_path, voices={"fr": "fr-FR-DeniseNeural", "auto": "x", "estonian": "y"})
    assert eng.voices == tts_engine.EDGE_VOICES
    assert eng.edge_voice("Español") == "es-MX-DaliaNeural"
    assert eng.edge_voice("ES_mx") == "es-MX-DaliaNeural"
    assert eng.edge_voice("estonian") == "en-US-AnaNeural"  # not "es" just because it starts with it


def test_auto_language_is_guessed_from_the_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_lang_by_text(monkeypatch)
    eng = TTSEngine("edge", out_dir=tmp_path)
    eng.speak("¿Cuánto es un medio más un tercio?", "auto")
    eng.speak("What is one half plus one third?", "auto")
    assert FakeComm.voices == ["es-MX-DaliaNeural", "en-US-AnaNeural"]


def test_legacy_spanish_voice_keeps_serving_calls_without_a_language(tmp_path: Path) -> None:
    # SAR_TTS_VOICE=es-MX-JorgeNeural used to make all speech Spanish; speak(text) must still do that.
    eng = TTSEngine("edge", "es-MX-JorgeNeural", tmp_path)
    assert eng.edge_voice("es") == "es-MX-JorgeNeural"
    assert eng.edge_voice("en") == "en-US-AnaNeural"
    assert eng.default_language == "es"
    eng.speak("hola")
    eng.speak("hi", "en")
    assert FakeComm.voices == ["es-MX-JorgeNeural", "en-US-AnaNeural"]


def test_legacy_voice_in_another_language_serves_only_calls_without_a_language(tmp_path: Path) -> None:
    eng = TTSEngine("edge", "fr-FR-DeniseNeural", tmp_path)
    eng.speak("bonjour")
    eng.speak("hi", "en")
    eng.speak("hola", "es")
    assert FakeComm.voices == ["fr-FR-DeniseNeural", "en-US-AnaNeural", "es-MX-DaliaNeural"]
    assert eng.default_language == "en"


def test_bad_spanish_voice_does_not_turn_edge_off_for_english(tmp_path: Path) -> None:
    FakeComm.fail_voices = {"es-MX-NoSuchNeural"}
    FakeComm.error = ValueError
    eng = TTSEngine("edge", out_dir=tmp_path, voices={"es": "es-MX-NoSuchNeural"})
    assert eng.speak("hola", "es") is None
    name = eng.speak("hi", "en")
    assert name and name.endswith(".mp3")
    assert eng.speak("otra vez", "es") is None
    assert FakeComm.voices == ["es-MX-NoSuchNeural", "en-US-AnaNeural"]  # the bad voice is cooling down


@pytest.mark.parametrize("error", [OSError, TimeoutError])
def test_network_failure_turns_edge_off_for_every_voice(tmp_path: Path, error: type[BaseException]) -> None:
    FakeComm.fail = True
    FakeComm.error = error
    eng = TTSEngine("edge", out_dir=tmp_path)
    assert eng.speak("hola", "es") is None
    assert eng.speak("hi", "en") is None
    assert FakeComm.calls == 1  # offline: English does not wait for its own timeout


# ---- voice speed (accessibility) -------------------------------------------------------------------------------------


def test_edge_rate_is_a_signed_percentage() -> None:
    assert tts_engine.edge_rate(1.0) == "+0%"
    assert tts_engine.edge_rate(0.75) == "-25%"
    assert tts_engine.edge_rate(1.25) == "+25%"
    assert tts_engine.edge_rate(0.5) == "-50%"
    assert tts_engine.edge_rate(2.0) == "+100%"
    assert tts_engine.edge_rate(1.1) == "+10%"  # (1.1 - 1) * 100 is 10.000000000000009


def test_edge_rate_scales_each_voices_own_pace(tmp_path: Path) -> None:
    eng = TTSEngine("edge", out_dir=tmp_path)
    eng.speak("hello", "en")
    eng.speak("hola", "es")
    eng.set_rate(0.75)
    eng.speak("slower", "en")
    eng.speak("más lento", "es")
    eng.set_rate(1.25)
    eng.speak("faster", "en")
    eng.speak("más rápido", "es")
    eng.set_rate(1.0)
    eng.speak("normal again", "en")
    eng.speak("otra vez normal", "es")
    # The speed never touches the pitch: Dalia keeps whatever kid pitch EDGE_PROSODY gives her.
    own_rate, pitch = tts_engine.EDGE_PROSODY["es-MX-DaliaNeural"]
    assert own_rate == "+5%", "the rates below are worked out from Dalia's own +5%"
    assert FakeComm.prosody == [
        ("+0%", "+0Hz"),
        ("+5%", pitch),  # at 1.0 every voice sounds exactly as it does without the setting
        ("-25%", "+0Hz"),
        ("-21%", pitch),  # Dalia's own +5% at 0.75: 1.05 * 0.75 = 0.7875
        ("+25%", "+0Hz"),
        ("+31%", pitch),  # 1.05 * 1.25 = 1.3125
        ("+0%", "+0Hz"),
        ("+5%", pitch),
    ]


def test_set_rate_is_clamped_and_non_finite_means_normal(tmp_path: Path) -> None:
    assert (tts_engine.MIN_RATE, tts_engine.MAX_RATE) == (0.5, 2.0)
    eng = TTSEngine("edge", out_dir=tmp_path)
    for factor in (0.1, 9, float("nan"), 0.75, float("inf")):
        eng.set_rate(factor)
        eng.speak("x", "en")
    assert [rate for rate, _ in FakeComm.prosody] == ["-50%", "+100%", "+0%", "-25%", "+0%"]


def test_offline_while_a_network_failure_cools_down(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    now = {"t": 1000.0}
    monkeypatch.setattr(tts_engine.time, "monotonic", lambda: now["t"])
    eng = TTSEngine("edge", out_dir=tmp_path)
    assert eng.offline is False, "not known to be down before the first try"
    FakeComm.fail, FakeComm.error = True, OSError
    eng.speak("hi", "en")
    assert eng.offline is True
    now["t"] += tts_engine.EDGE_RETRY_AFTER_S + 1
    assert eng.offline is False, "tried again after the cooldown"
    FakeComm.error = ValueError  # a bad voice is not a dead network
    eng.speak("hola", "es")
    assert eng.offline is False
