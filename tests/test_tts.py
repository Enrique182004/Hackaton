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
        if FakeComm.fail:
            raise RuntimeError("offline")
        Path(path).write_bytes(b"mp3")


@pytest.fixture(autouse=True)
def fake_edge(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeComm.fail = False
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


def test_edge_voice_follows_the_language(tmp_path: Path) -> None:
    eng = TTSEngine("edge", out_dir=tmp_path)
    eng.speak("hello")
    eng.speak("hola", "es")
    eng.speak("hola", lang="es-MX")
    eng.speak("bonjour", "fr")  # unsupported -> English
    assert FakeComm.voices == ["en-US-AriaNeural", "es-MX-DaliaNeural", "es-MX-DaliaNeural", "en-US-AriaNeural"]


def test_edge_voices_can_be_overridden(tmp_path: Path) -> None:
    eng = TTSEngine("edge", "en-GB-SoniaNeural", tmp_path, voices={"es": "es-MX-JorgeNeural"})
    eng.speak("hi")
    eng.speak("hola", "es")
    assert FakeComm.voices == ["en-GB-SoniaNeural", "es-MX-JorgeNeural"]
    assert eng.voice == "en-GB-SoniaNeural"


def test_legacy_spanish_voice_serves_spanish_only(tmp_path: Path) -> None:
    eng = TTSEngine("edge", "es-MX-JorgeNeural", tmp_path)
    assert eng.edge_voice("es") == "es-MX-JorgeNeural"
    assert eng.edge_voice("en") == "en-US-AriaNeural"


SAY_LIST = """Albert              en_US    # Hello! My name is Albert.
Eddy (English (US)) en_US    # Hello! My name is Eddy.
Eddy (Spanish (Spain)) es_ES    # ¡Hola! Me llamo Eddy.
Eddy (Spanish (Mexico)) es_MX    # ¡Hola! Me llamo Eddy.
Flo (Spanish (Mexico)) es_MX    # ¡Hola! Me llamo Flo.
Mónica              es_ES    # ¡Hola! Me llamo Mónica.
Paulina             es_MX    # ¡Hola! Me llamo Paulina.
"""


def test_parse_say_voices() -> None:
    voices = tts_engine.parse_say_voices(SAY_LIST + "garbage line\n")
    assert voices[0] == ("Albert", "en_US")
    assert ("Eddy (Spanish (Mexico))", "es_MX") in voices
    assert ("Mónica", "es_ES") in voices
    assert len(voices) == 7


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
    assert run.calls == []  # English never needs the voice list
    eng.speak("hola", "es")
    eng.speak("dos tercios", "es")
    assert [p.args for p in procs] == [
        ["say", "hi"],
        ["say", "-v", "Paulina", "hola"],
        ["say", "-v", "Paulina", "dos tercios"],
    ]
    assert run.calls == [["say", "-v", "?"]]  # listed once, then cached


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


def _voice(id: str, name: str = "", languages: list[object] | None = None) -> types.SimpleNamespace:
    return types.SimpleNamespace(id=id, name=name, languages=languages or [])


def test_pick_pyttsx3_voice() -> None:
    windows = [
        _voice(r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0", "Microsoft Zira"),
        _voice(r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ES-ES_HELENA_11.0", "Helena"),
        _voice(r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ES-MX_SABINA_11.0", "Sabina"),
    ]
    assert tts_engine.pick_pyttsx3_voice(windows, "es") == windows[2].id
    mac = [
        _voice("com.apple.voice.compact.en-US.Samantha", "Samantha", ["en_US"]),
        _voice("com.apple.voice.compact.es-ES.Monica", "Mónica", ["es_ES"]),
    ]
    assert tts_engine.pick_pyttsx3_voice(mac, "es") == "com.apple.voice.compact.es-ES.Monica"
    espeak = [_voice("english", "english", [b"\x05en"]), _voice("spanish-latin-am", "spanish-latin-am", [b"\x05es-la"])]
    assert tts_engine.pick_pyttsx3_voice(espeak, "es") == "spanish-latin-am"
    assert tts_engine.pick_pyttsx3_voice(windows[:1], "es") is None  # "Voices\Tokens" is not Spanish
    assert tts_engine.pick_pyttsx3_voice(windows, "en") is None


def test_pyttsx3_switches_voice_per_language(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeComm.fail = True
    log: list[tuple[str, str]] = []
    done = threading.Event()

    class Engine:
        def __init__(self) -> None:
            self.props: dict[str, object] = {
                "voice": "default-en",
                "voices": [_voice("en-voice", "English", ["en_US"]), _voice("es-voice", "Paulina", ["es_MX"])],
            }

        def getProperty(self, name: str) -> object:
            return self.props[name]

        def setProperty(self, name: str, value: object) -> None:
            self.props[name] = value
            log.append(("set", str(value)))

        def say(self, t: str) -> None:
            log.append(("say", t))

        def runAndWait(self) -> None:
            if sum(1 for kind, _ in log if kind == "say") == 3:
                done.set()

    mod = types.ModuleType("pyttsx3")
    mod.init = lambda: Engine()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyttsx3", mod)
    eng = TTSEngine("pyttsx3", out_dir=tmp_path)
    eng.speak("hola", "es")
    eng.speak("otra vez", "es")
    eng.speak("hi")
    assert done.wait(3)
    eng.stop()
    assert log == [
        ("set", "es-voice"),
        ("say", "hola"),
        ("say", "otra vez"),
        ("set", "default-en"),
        ("say", "hi"),
    ]
