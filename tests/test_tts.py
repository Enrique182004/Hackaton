from __future__ import annotations

import os
import sys
import threading
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

    def __init__(self, text: str, voice: str) -> None:
        self.text, self.voice = text, voice
        FakeComm.voices.append(voice)

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
    mod = types.ModuleType("edge_tts")
    mod.Communicate = FakeComm  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "edge_tts", mod)


class FakeProc:
    def __init__(self, args: list[str]) -> None:
        self.args = args
        self.terminated = False

    def poll(self) -> None:
        return None

    def terminate(self) -> None:
        self.terminated = True


def test_edge_writes_file(tmp_path: Path) -> None:
    eng = TTSEngine("edge", out_dir=tmp_path / "sub")
    name = eng.speak("hello")
    assert name and name.endswith(".mp3")
    assert (tmp_path / "sub" / name).read_bytes() == b"mp3"


def test_edge_fails_falls_back_to_say(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeComm.fail = True
    procs: list[FakeProc] = []
    monkeypatch.setattr(tts_engine.subprocess, "Popen", lambda a, *k, **kw: procs.append(FakeProc(a)) or procs[-1])
    eng = TTSEngine("edge", out_dir=tmp_path)
    assert eng.speak("hi there") is None
    assert procs[0].args == ["say", "hi there"]
    eng.stop()
    assert procs[0].terminated


def test_say_fails_falls_back_to_pyttsx3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeComm.fail = True
    spoken: list[str] = []
    import threading

    ev = threading.Event()

    class Engine:
        def say(self, t: str) -> None:
            spoken.append(t)

        def runAndWait(self) -> None:
            ev.set()

    mod = types.ModuleType("pyttsx3")
    mod.init = lambda: Engine()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyttsx3", mod)

    def no_say(*a: object, **k: object) -> None:
        raise FileNotFoundError("say")

    monkeypatch.setattr(tts_engine.subprocess, "Popen", no_say)
    eng = TTSEngine("edge", out_dir=tmp_path)
    assert eng.speak("yo") is None
    assert ev.wait(3)
    assert spoken == ["yo"]
    eng.stop()


def test_all_fail_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeComm.fail = True
    monkeypatch.setitem(sys.modules, "pyttsx3", None)  # import raises ImportError

    def no_say(*a: object, **k: object) -> None:
        raise FileNotFoundError("say")

    monkeypatch.setattr(tts_engine.subprocess, "Popen", no_say)
    assert TTSEngine("edge", out_dir=tmp_path).speak("x") is None


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


def _no_say(monkeypatch: pytest.MonkeyPatch) -> list[FakeProc]:
    procs: list[FakeProc] = []
    monkeypatch.setattr(tts_engine.subprocess, "Popen", lambda a, *k, **kw: procs.append(FakeProc(a)) or procs[-1])
    return procs


def test_edge_failure_is_remembered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeComm.fail = True
    procs = _no_say(monkeypatch)
    eng = TTSEngine("edge", out_dir=tmp_path)
    eng.speak("one")
    eng.speak("two")
    assert FakeComm.calls == 1
    assert [p.args for p in procs] == [["say", "one"], ["say", "two"]]


def test_edge_retried_after_cooldown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeComm.fail = True
    _no_say(monkeypatch)
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


def test_edge_failure_removes_partial_mp3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeComm.fail = True
    FakeComm.partial = True
    _no_say(monkeypatch)
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
        "en-US-AriaNeural",
        "es-MX-DaliaNeural",
        "es-MX-DaliaNeural",
        "es-MX-DaliaNeural",
        "en-US-AriaNeural",
    ]


def test_edge_voices_can_be_overridden(tmp_path: Path) -> None:
    eng = TTSEngine("edge", "en-GB-SoniaNeural", tmp_path, voices={"es": "es-MX-JorgeNeural"})
    eng.speak("hi", "en")
    eng.speak("hola", "es")
    eng.speak("hi")
    assert FakeComm.voices == ["en-GB-SoniaNeural", "es-MX-JorgeNeural", "en-GB-SoniaNeural"]
    assert eng.voice == "en-GB-SoniaNeural"


def test_voice_languages_use_the_shared_normalizer(tmp_path: Path) -> None:
    eng = TTSEngine("edge", out_dir=tmp_path, voices={"spanish": "es-MX-JorgeNeural"})
    assert eng.voices == {"en": "en-US-AriaNeural", "es": "es-MX-JorgeNeural"}
    # Unknown keys are ignored instead of replacing the English voice.
    eng = TTSEngine("edge", out_dir=tmp_path, voices={"fr": "fr-FR-DeniseNeural", "auto": "x", "estonian": "y"})
    assert eng.voices == tts_engine.EDGE_VOICES
    assert eng.edge_voice("Español") == "es-MX-DaliaNeural"
    assert eng.edge_voice("ES_mx") == "es-MX-DaliaNeural"
    assert eng.edge_voice("estonian") == "en-US-AriaNeural"  # not "es" just because it starts with it


def test_auto_language_is_guessed_from_the_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_lang_by_text(monkeypatch)
    eng = TTSEngine("edge", out_dir=tmp_path)
    eng.speak("¿Cuánto es un medio más un tercio?", "auto")
    eng.speak("What is one half plus one third?", "auto")
    assert FakeComm.voices == ["es-MX-DaliaNeural", "en-US-AriaNeural"]


def test_legacy_spanish_voice_keeps_serving_calls_without_a_language(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # SAR_TTS_VOICE=es-MX-JorgeNeural used to make all speech Spanish; speak(text) must still do that.
    eng = TTSEngine("edge", "es-MX-JorgeNeural", tmp_path)
    assert eng.edge_voice("es") == "es-MX-JorgeNeural"
    assert eng.edge_voice("en") == "en-US-AriaNeural"
    assert eng.default_language == "es"
    eng.speak("hola")
    eng.speak("hi", "en")
    assert FakeComm.voices == ["es-MX-JorgeNeural", "en-US-AriaNeural"]
    FakeComm.fail = True
    monkeypatch.setattr(tts_engine.subprocess, "run", _Run())
    procs = _no_say(monkeypatch)
    eng.speak("dos tercios")
    assert procs[0].args == ["say", "-v", "Paulina", "dos tercios"]


def test_legacy_voice_in_another_language_serves_only_calls_without_a_language(tmp_path: Path) -> None:
    eng = TTSEngine("edge", "fr-FR-DeniseNeural", tmp_path)
    eng.speak("bonjour")
    eng.speak("hi", "en")
    eng.speak("hola", "es")
    assert FakeComm.voices == ["fr-FR-DeniseNeural", "en-US-AriaNeural", "es-MX-DaliaNeural"]
    assert eng.default_language == "en"


SAY_LIST = """Albert              en_US    # Hello! My name is Albert.
Eddy (English (US)) en_US    # Hello! My name is Eddy.
Eddy (Spanish (Spain)) es_ES    # ¡Hola! Me llamo Eddy.
Eddy (Spanish (Mexico)) es_MX    # ¡Hola! Me llamo Eddy.
Flo (Spanish (Mexico)) es_MX    # ¡Hola! Me llamo Flo.
Mónica              es_ES    # ¡Hola! Me llamo Mónica.
Paulina             es_MX    # ¡Hola! Me llamo Paulina.
"""

# `say -v ?` on macOS 27 once several variants are installed: names carry the language, Paulina is listed twice.
SAY_LIST_MACOS_27 = """Albert              en_US    # Hello! My name is Albert.
Eddy (English (US)) en_US    # Hello! My name is Eddy.
Eddy (Spanish (Spain)) es_ES    # ¡Hola! Me llamo Eddy.
Eddy (Spanish (Mexico)) es_MX    # ¡Hola! Me llamo Eddy.
Flo (Spanish (Mexico)) es_MX    # ¡Hola! Me llamo Flo.
Grandma (Spanish (Mexico)) es_MX    # ¡Hola! Me llamo Grandma.
Mónica              es_ES    # ¡Hola! Me llamo Mónica.
Paulina (Spanish (Mexico)) es_MX    # ¡Hola! Me llamo Paulina.
Paulina (Spanish (Mexico)) es_MX    # ¡Hola! Me llamo Paulina.
Rocko (Spanish (Mexico)) es_MX    # ¡Hola! Me llamo Rocko.
Samantha (English (US)) en_US    # Hello! My name is Samantha.
"""


def test_parse_say_voices() -> None:
    voices = tts_engine.parse_say_voices(SAY_LIST + "garbage line\n")
    assert voices[0] == ("Albert", "en_US")
    assert ("Eddy (Spanish (Mexico))", "es_MX") in voices
    assert ("Mónica", "es_ES") in voices
    assert len(voices) == 7
    assert ("Paulina (Spanish (Mexico))", "es_MX") in tts_engine.parse_say_voices(SAY_LIST_MACOS_27)


def test_pick_say_voice_prefers_mexican_spanish() -> None:
    voices = tts_engine.parse_say_voices(SAY_LIST)
    pick = tts_engine.pick_say_voice
    assert pick(voices, "es") == "Paulina"
    assert pick([v for v in voices if v[0] != "Paulina"], "es") == "Eddy (Spanish (Mexico))"
    assert pick([("Juan", "es_MX"), ("Jorge", "es_ES")], "es") == "Juan"
    assert pick([("Jorge", "es_ES"), ("Diego", "es_419")], "es") == "Diego"
    assert pick([("Jorge", "es_ES"), ("Albert", "en_US")], "es") == "Jorge"
    assert pick([("Albert", "en_US")], "es") is None
    assert pick(voices, "en") is None


def test_pick_say_voice_finds_paulina_under_her_long_name() -> None:
    voices = tts_engine.parse_say_voices(SAY_LIST_MACOS_27)
    assert tts_engine.pick_say_voice(voices, "es") == "Paulina (Spanish (Mexico))"
    assert tts_engine.pick_say_voice(voices, "es-MX") == "Paulina (Spanish (Mexico))"


def test_pick_say_voice_ranks_natural_voices_above_eloquence() -> None:
    pick = tts_engine.pick_say_voice
    assert pick([("Eddy (Spanish (Mexico))", "es_MX"), ("Juan", "es_MX")], "es") == "Juan"
    assert pick([("Rocko (Spanish (Mexico))", "es_MX"), ("Eddy (Spanish (Mexico))", "es_MX")], "es") == (
        "Eddy (Spanish (Mexico))"
    )
    assert pick([("Grandma (Spanish (Mexico))", "es_MX"), ("Mónica", "es_ES")], "es") == (
        "Grandma (Spanish (Mexico))"
    )  # Mexican first, even robotic
    assert pick([("Eddy (Spanish (Spain))", "es_ES"), ("Mónica", "es_ES")], "es") == "Mónica"
    assert pick([("Paulina", "es_ES"), ("Flo (Spanish (Mexico))", "es_MX")], "es") == "Flo (Spanish (Mexico))"


class _Run:
    def __init__(self, stdout: str = SAY_LIST, error: Exception | None = None) -> None:
        self.stdout, self.error = stdout, error
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str], **kwargs: object) -> types.SimpleNamespace:
        self.calls.append(args)
        if self.error is not None:
            raise self.error
        return types.SimpleNamespace(stdout=self.stdout, returncode=0)


def test_say_speaks_spanish_with_a_discovered_voice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _Run()
    monkeypatch.setattr(tts_engine.subprocess, "run", run)
    procs = _no_say(monkeypatch)
    eng = TTSEngine("say", out_dir=tmp_path)
    eng.speak("hi")
    eng.speak("hi", "en")
    assert run.calls == []  # English never needs the voice list
    eng.speak("hola", "es")
    eng.speak("dos tercios", "es")
    assert [p.args for p in procs] == [
        ["say", "hi"],
        ["say", "hi"],
        ["say", "-v", "Paulina", "hola"],
        ["say", "-v", "Paulina", "dos tercios"],
    ]
    assert run.calls == [["say", "-v", "?"]]  # listed once, then cached


def test_say_uses_paulina_from_the_macos_27_listing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _Run(SAY_LIST_MACOS_27)
    monkeypatch.setattr(tts_engine.subprocess, "run", run)
    procs = _no_say(monkeypatch)
    eng = TTSEngine("say", out_dir=tmp_path)
    eng.speak("¿Y ahora qué sigue?", "es")
    eng.speak("Un medio más un tercio", "es")
    assert [p.args[:3] for p in procs] == [["say", "-v", "Paulina (Spanish (Mexico))"]] * 2
    assert eng.say_voice("es") == "Paulina (Spanish (Mexico))"
    assert len(run.calls) == 1


def test_say_without_voice_list_uses_the_default_voice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _Run(error=FileNotFoundError("say"))
    monkeypatch.setattr(tts_engine.subprocess, "run", run)
    procs = _no_say(monkeypatch)
    eng = TTSEngine("say", out_dir=tmp_path)
    eng.speak("hola", "es")
    eng.speak("adiós", "es")
    assert [p.args for p in procs] == [["say", "hola"], ["say", "adiós"]]
    assert len(run.calls) == 1


def test_edge_fallback_to_say_keeps_the_language(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeComm.fail = True
    monkeypatch.setattr(tts_engine.subprocess, "run", _Run())
    procs = _no_say(monkeypatch)
    TTSEngine("edge", out_dir=tmp_path).speak("hola", "es")
    assert FakeComm.voices == ["es-MX-DaliaNeural"]
    assert procs[0].args == ["say", "-v", "Paulina", "hola"]


def test_bad_spanish_voice_does_not_turn_edge_off_for_english(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeComm.fail_voices = {"es-MX-NoSuchNeural"}
    FakeComm.error = ValueError
    monkeypatch.setattr(tts_engine.subprocess, "run", _Run())
    procs = _no_say(monkeypatch)
    eng = TTSEngine("edge", out_dir=tmp_path, voices={"es": "es-MX-NoSuchNeural"})
    assert eng.speak("hola", "es") is None
    name = eng.speak("hi", "en")
    assert name and name.endswith(".mp3")
    assert eng.speak("otra vez", "es") is None
    assert FakeComm.voices == ["es-MX-NoSuchNeural", "en-US-AriaNeural"]  # the bad voice is cooling down
    assert [p.args for p in procs] == [["say", "-v", "Paulina", "hola"], ["say", "-v", "Paulina", "otra vez"]]


@pytest.mark.parametrize("error", [OSError, TimeoutError])
def test_network_failure_turns_edge_off_for_every_voice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: type[BaseException]
) -> None:
    FakeComm.fail = True
    FakeComm.error = error
    monkeypatch.setattr(tts_engine.subprocess, "run", _Run())
    procs = _no_say(monkeypatch)
    eng = TTSEngine("edge", out_dir=tmp_path)
    eng.speak("hola", "es")
    eng.speak("hi", "en")
    assert FakeComm.calls == 1  # offline: English does not wait for its own timeout
    assert [p.args for p in procs] == [["say", "-v", "Paulina", "hola"], ["say", "hi"]]


def _voice(id: str, name: str = "", languages: list[object] | None = None) -> types.SimpleNamespace:
    return types.SimpleNamespace(id=id, name=name, languages=languages or [])


_SAPI = "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\"
WINDOWS = [
    _voice(_SAPI + "TTS_MS_EN-US_ZIRA_11.0", "Microsoft Zira Desktop - English (United States)"),
    _voice(_SAPI + "TTS_MS_ES-ES_HELENA_11.0", "Microsoft Helena Desktop - Spanish (Spain)"),
    _voice(_SAPI + "TTS_MS_ES-MX_SABINA_11.0", "Microsoft Sabina Desktop - Spanish (Mexico)"),
]
# pyttsx3 on macOS 27, in its own order: legacy novelty and Eloquence voices come before Paulina and Samantha.
MAC = [
    _voice("com.apple.speech.synthesis.voice.Albert", "Albert", ["en_US"]),
    _voice("com.apple.speech.synthesis.voice.BadNews", "Bad News", ["en_US"]),
    _voice("com.apple.eloquence.en-US.Eddy", "Eddy (English (US))", ["en_US"]),
    _voice("com.apple.eloquence.es-ES.Eddy", "Eddy (Spanish (Spain))", ["es_ES"]),
    _voice("com.apple.eloquence.es-MX.Eddy", "Eddy (Spanish (Mexico))", ["es_MX"]),
    _voice("com.apple.eloquence.es-MX.Flo", "Flo (Spanish (Mexico))", ["es_MX"]),
    _voice("com.apple.voice.compact.es-ES.Monica", "Mónica", ["es_ES"]),
    _voice("com.apple.voice.compact.es-MX.Paulina", "Paulina (Spanish (Mexico))", ["es_MX"]),
    _voice("com.apple.voice.compact.es-MX.Paulina", "Paulina (Spanish (Mexico))", ["es_MX"]),
    _voice("com.apple.voice.compact.en-US.Samantha", "Samantha (English (US))", ["en_US"]),
]
ESPEAK = [
    _voice("english", "english", [b"\x05en"]),
    _voice("spanish", "spanish", [b"\x05es"]),
    _voice("spanish-latin-am", "spanish-latin-am", [b"\x05es-la"]),
]


def test_pick_pyttsx3_voice() -> None:
    pick = tts_engine.pick_pyttsx3_voice
    assert pick(WINDOWS, "es") == WINDOWS[2].id
    mac_spain_only = [
        _voice("com.apple.voice.compact.en-US.Samantha", "Samantha", ["en_US"]),
        _voice("com.apple.voice.compact.es-ES.Monica", "Mónica", ["es_ES"]),
    ]
    assert pick(mac_spain_only, "es") == "com.apple.voice.compact.es-ES.Monica"
    assert pick(ESPEAK, "es") == "spanish-latin-am"
    assert pick(WINDOWS[:1], "es") is None  # "Voices\Tokens" is not Spanish
    assert pick(WINDOWS[1:], "en") is None
    assert pick(WINDOWS, "fr") is None


def test_pick_pyttsx3_voice_prefers_paulina_on_macos() -> None:
    assert tts_engine.pick_pyttsx3_voice(MAC, "es") == "com.apple.voice.compact.es-MX.Paulina"
    without_paulina = [v for v in MAC if "Paulina" not in v.id]
    assert tts_engine.pick_pyttsx3_voice(without_paulina, "es") == "com.apple.eloquence.es-MX.Eddy"


def test_pick_pyttsx3_voice_for_english() -> None:
    pick = tts_engine.pick_pyttsx3_voice
    assert pick(MAC, "en") == "com.apple.voice.compact.en-US.Samantha"  # not Albert, not Eloquence Eddy
    assert pick(WINDOWS, "en") == WINDOWS[0].id
    assert pick(ESPEAK, "en") == "english"
    uk_us = [_voice("en-gb", "English (Great Britain)", [b"\x02en-gb"]), _voice("en-us", "English (America)")]
    assert pick(uk_us, "en") == "en-us"
    assert pick(uk_us, "en", default="en-gb") == "en-gb"  # the teacher's English voice stays


def test_pick_pyttsx3_voice_keeps_a_default_that_speaks_the_language() -> None:
    pick = tts_engine.pick_pyttsx3_voice
    assert pick(MAC, "en", default="com.apple.eloquence.en-US.Eddy") == "com.apple.eloquence.en-US.Eddy"
    assert pick(WINDOWS, "en", default=WINDOWS[2].id) == WINDOWS[0].id  # a Spanish default can't read English
    assert pick(WINDOWS, "es", default=WINDOWS[2].id) == WINDOWS[2].id
    assert pick(WINDOWS, "es", default=WINDOWS[1].id) == WINDOWS[2].id  # Mexican Spanish beats a Spain default
    assert pick(WINDOWS, "es", default=WINDOWS[0].id) == WINDOWS[2].id


class _Pyttsx3Engine:
    def __init__(self, voices: list[types.SimpleNamespace], default: object, log: list[tuple[str, str]]) -> None:
        self.props: dict[str, object] = {"voice": default, "voices": voices}
        self.log = log
        self.done = threading.Event()
        self.expected = 0

    def getProperty(self, name: str) -> object:
        return self.props[name]

    def setProperty(self, name: str, value: object) -> None:
        self.props[name] = value
        self.log.append(("set", str(value)))

    def say(self, t: str) -> None:
        self.log.append(("say", t))

    def runAndWait(self) -> None:
        if sum(1 for kind, _ in self.log if kind == "say") == self.expected:
            self.done.set()


def _run_pyttsx3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    voices: list[types.SimpleNamespace],
    default: object,
    lines: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    FakeComm.fail = True
    log: list[tuple[str, str]] = []
    engine = _Pyttsx3Engine(voices, default, log)
    engine.expected = len(lines)
    mod = types.ModuleType("pyttsx3")
    mod.init = lambda: engine  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyttsx3", mod)
    eng = TTSEngine("pyttsx3", out_dir=tmp_path)
    for text, lang in lines:
        eng.speak(text, lang)
    assert engine.done.wait(3)
    eng.stop()
    return log


def test_pyttsx3_switches_voice_per_language(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    voices = [_voice("default-en", "Samantha", ["en_US"]), _voice("es-voice", "Paulina", ["es_MX"])]
    log = _run_pyttsx3(tmp_path, monkeypatch, voices, "default-en", [("hola", "es"), ("otra vez", "es"), ("hi", "en")])
    assert log == [
        ("set", "es-voice"),
        ("say", "hola"),
        ("say", "otra vez"),
        ("set", "default-en"),
        ("say", "hi"),
    ]


def test_pyttsx3_reads_english_with_an_english_voice_on_a_spanish_system(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = _run_pyttsx3(tmp_path, monkeypatch, WINDOWS, WINDOWS[2].id, [("hi", "en"), ("hola", "es")])
    assert log == [("set", WINDOWS[0].id), ("say", "hi"), ("set", WINDOWS[2].id), ("say", "hola")]


def test_pyttsx3_espeak_switches_back_to_english(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # espeak reports no current voice until one is set.
    log = _run_pyttsx3(tmp_path, monkeypatch, ESPEAK, None, [("hola", "es"), ("hi", "en")])
    assert log == [("set", "spanish-latin-am"), ("say", "hola"), ("set", "english"), ("say", "hi")]
