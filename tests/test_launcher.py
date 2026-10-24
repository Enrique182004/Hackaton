from __future__ import annotations

import builtins
import dataclasses
import errno
import http.client
import importlib.machinery
import importlib.util
import json
import os
import re
import socket
import sys
import threading
import urllib.error
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from biobench_sar import app as app_module
from biobench_sar import config, envfile

LAUNCHER_PATH = Path(__file__).resolve().parent.parent / "packaging" / "launcher.py"
CHECKER_PATH = LAUNCHER_PATH.parent / "check_macos_app.py"
ENV_KEYS = (
    "SAR_DATA_DIR",
    "SAR_ENV_FILE",
    "SAR_PORT",
    "SAR_CAMERA_INDEX",
    "SAR_VLM_BACKEND",
    "SAR_TTS_BACKEND",
    "SAR_LISTEN_MODE",
    "HF_HUB_OFFLINE",
    "HF_HUB_DISABLE_TELEMETRY",
    "HF_HOME",
    "SENTENCE_TRANSFORMERS_HOME",
    "ANONYMIZED_TELEMETRY",
    "SAR_WHISPER_MODEL",
    "SAR_HOST",
    "SAR_RECORD_SECONDS",
)


def _load(path: Path, name: str) -> ModuleType:
    # By path: a `packaging` import would find the PyPI package of that name, not this folder.
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


launcher = _load(LAUNCHER_PATH, "sar_launcher")
checker = _load(CHECKER_PATH, "sar_check_macos_app")
REAL_BUSY_PORT_ERROR = launcher.busy_port_error


@pytest.fixture(autouse=True)
def clean_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Whatever the launcher sets in os.environ / config.DATA_DIR is undone after each test.

    Failure reports go to a log in a temporary data folder, never the developer's own, and run_app's port
    check is stubbed: a copy of the app already running on 5050 must not fail these tests.
    """
    environ = dict(os.environ)  # sar.env keys go straight into os.environ, past monkeypatch
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(config, "DATA_DIR", config.DATA_DIR)
    monkeypatch.setattr(envfile, "_SOURCES", {})  # which sar.env set each key: per test, like per process
    data = tmp_path / "default-userdata"
    data.mkdir()
    monkeypatch.setattr(launcher, "user_data_dir", lambda: data)
    monkeypatch.setattr(launcher, "_log_redirect", None)
    monkeypatch.setattr(launcher, "busy_port_error", lambda host, port: None)
    yield
    os.environ.clear()
    os.environ.update(environ)


@pytest.fixture
def alerts(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    shown: list[str] = []
    monkeypatch.setattr(launcher, "alert", shown.append)
    return shown


@pytest.fixture
def folders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    exe_dir, data_dir, bundle = tmp_path / "app", tmp_path / "userdata", tmp_path / "bundle"
    for folder in (exe_dir, data_dir, bundle):
        folder.mkdir()
    monkeypatch.setattr(launcher, "executable_dir", lambda: exe_dir)
    monkeypatch.setattr(launcher, "user_data_dir", lambda: data_dir)
    monkeypatch.setattr(launcher, "resource_dir", lambda: bundle)
    return {"exe": exe_dir, "data": data_dir, "bundle": bundle}


def _bundle_models(bundle: Path) -> Path:
    models = bundle / "models"
    (models / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots").mkdir(parents=True)
    return models


# ---- environment -------------------------------------------------------------------------------------


def test_data_dir_defaults_to_the_user_data_folder(folders: dict[str, Path]) -> None:
    assert launcher.prepare_environment(frozen=False) == []
    assert os.environ["SAR_DATA_DIR"] == str(folders["data"] / "data")
    expected = folders["data"] / "data"
    assert expected == config.DATA_DIR
    assert config.Settings().chroma_dir == folders["data"] / "data" / "chroma_db"


def test_real_environment_beats_every_sar_env(folders: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    (folders["exe"] / "sar.env").write_text("SAR_DATA_DIR=/from/file\nSAR_CAMERA_INDEX=3\n")
    monkeypatch.setenv("SAR_DATA_DIR", str(folders["exe"] / "mine"))
    launcher.prepare_environment(frozen=False)
    assert os.environ["SAR_DATA_DIR"] == str(folders["exe"] / "mine")
    assert os.environ["SAR_CAMERA_INDEX"] == "3"


def test_sar_env_next_to_the_app_beats_the_data_folder_one(folders: dict[str, Path]) -> None:
    (folders["exe"] / "sar.env").write_text("SAR_CAMERA_INDEX=2\n")
    (folders["data"] / "sar.env").write_text("SAR_CAMERA_INDEX=5\nSAR_PORT=6060\n")
    loaded = launcher.prepare_environment(frozen=False)
    assert loaded == [folders["exe"] / "sar.env", folders["data"] / "sar.env"]
    assert os.environ["SAR_CAMERA_INDEX"] == "2"
    assert os.environ["SAR_PORT"] == "6060"


def test_sar_env_may_move_the_data_folder(folders: dict[str, Path]) -> None:
    target = folders["exe"] / "classroom-data"
    (folders["data"] / "sar.env").write_text(f"SAR_DATA_DIR={target}\n")
    launcher.prepare_environment(frozen=False)
    assert config.Settings().upload_dir == target / "uploads"


def test_sar_env_file_override_comes_first(folders: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    custom = folders["exe"] / "room-12.env"
    custom.write_text("SAR_CAMERA_INDEX=7\n")
    (folders["exe"] / "sar.env").write_text("SAR_CAMERA_INDEX=2\n")
    monkeypatch.setenv("SAR_ENV_FILE", str(custom))
    assert launcher.prepare_environment(frozen=False)[0] == custom
    assert os.environ["SAR_CAMERA_INDEX"] == "7"


def test_frozen_build_uses_bundled_models_offline(folders: dict[str, Path]) -> None:
    models = _bundle_models(folders["bundle"])
    launcher.prepare_environment(frozen=True)
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["SENTENCE_TRANSFORMERS_HOME"] == str(models)
    assert os.environ["HF_HOME"] == str(models)


def test_frozen_build_without_bundled_models_stays_online(folders: dict[str, Path]) -> None:
    (folders["bundle"] / "models").mkdir()  # empty: fetch_models.py didn't run
    launcher.prepare_environment(frozen=True)
    assert "HF_HUB_OFFLINE" not in os.environ
    assert "SENTENCE_TRANSFORMERS_HOME" not in os.environ


def test_bundled_whisper_checkpoint_replaces_the_download(folders: dict[str, Path]) -> None:
    whisper = folders["bundle"] / "models" / "whisper"
    whisper.mkdir(parents=True)
    (whisper / "base.pt").write_bytes(b"checkpoint")
    launcher.prepare_environment(frozen=True)
    assert os.environ["SAR_WHISPER_MODEL"] == str(whisper / "base.pt")
    assert "HF_HUB_OFFLINE" not in os.environ, "no embedding model bundled"


def test_a_different_whisper_model_still_downloads(folders: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    whisper = folders["bundle"] / "models" / "whisper"
    whisper.mkdir(parents=True)
    (whisper / "base.pt").write_bytes(b"checkpoint")
    monkeypatch.setenv("SAR_WHISPER_MODEL", "small")
    launcher.prepare_environment(frozen=True)
    assert os.environ["SAR_WHISPER_MODEL"] == "small"


def test_source_checkout_leaves_model_settings_alone(folders: dict[str, Path]) -> None:
    _bundle_models(folders["bundle"])
    launcher.prepare_environment(frozen=False)
    assert "HF_HUB_OFFLINE" not in os.environ and "HF_HOME" not in os.environ


def test_offline_switch_can_be_overridden_in_sar_env(folders: dict[str, Path]) -> None:
    _bundle_models(folders["bundle"])
    (folders["data"] / "sar.env").write_text("HF_HUB_OFFLINE=0\n")
    launcher.prepare_environment(frozen=True)
    assert os.environ["HF_HUB_OFFLINE"] == "0"


def test_frozen_build_leaves_a_settings_template_in_the_data_folder(folders: dict[str, Path]) -> None:
    (folders["bundle"] / "sar.env.example").write_text("SAR_CAMERA_INDEX=1\n")
    launcher.prepare_environment(frozen=True)
    assert (folders["data"] / "sar.env.example").read_text() == "SAR_CAMERA_INDEX=1\n"
    assert not (folders["data"] / "sar.env").exists(), "the template must not change any setting"


def test_chromadb_telemetry_is_off_unless_asked(folders: dict[str, Path]) -> None:
    launcher.prepare_environment(frozen=False)
    assert os.environ["ANONYMIZED_TELEMETRY"] == "False"


# ---- Ollama ------------------------------------------------------------------------------------------


def test_ollama_check_asks_tags_with_a_short_timeout() -> None:
    calls: list[tuple[str, float]] = []

    def fetch(url: str, timeout: float) -> bytes:
        calls.append((url, timeout))
        return json.dumps({"models": [{"name": "qwen2.5vl:3b", "model": "qwen2.5vl:3b"}]}).encode()

    status = launcher.check_ollama("http://localhost:11434/", "qwen2.5vl:3b", fetch=fetch)
    assert status.reachable and status.has_model
    assert calls == [("http://localhost:11434/api/tags", 2.0)]


@pytest.mark.parametrize(
    "body",
    [b'{"models": null}', b'{"models": 5}', b'{"models": [5, {"name": 3}, {"name": null}]}', b"[]", b'"ok"', b""],
)
def test_a_malformed_ollama_answer_is_reachable_without_the_model(body: bytes) -> None:
    status = launcher.check_ollama(fetch=lambda url, timeout: body)
    assert status.reachable and not status.has_model


def test_ollama_model_names_reads_name_and_model() -> None:
    body = {"models": [{"name": "a:latest", "model": "b:1"}, {"name": "c"}, "junk"]}
    assert launcher.ollama_model_names(body) == {"a:latest", "b:1", "c"}


def test_ollama_without_the_model_is_reachable_but_not_ready() -> None:
    body = json.dumps({"models": [{"name": "llama3:latest"}]}).encode()
    status = launcher.check_ollama(fetch=lambda url, timeout: body)
    assert status.reachable and not status.has_model


def test_untagged_model_names_match_latest() -> None:
    body = json.dumps({"models": [{"name": "llava:latest"}]}).encode()
    assert launcher.check_ollama(model="llava", fetch=lambda url, timeout: body).has_model


@pytest.mark.parametrize(
    "failure",
    [
        urllib.error.URLError("connection refused"),
        TimeoutError("timed out"),
        b"not json",
        http.client.BadStatusLine("SSH-2.0-OpenSSH"),  # another program on Ollama's port
    ],
)
def test_unreachable_or_garbled_ollama_is_reported(failure: Any) -> None:
    def fetch(url: str, timeout: float) -> bytes:
        if isinstance(failure, bytes):
            return failure
        raise failure

    status = launcher.check_ollama(fetch=fetch)
    assert not status.reachable and not status.has_model and status.error


def test_advice_for_missing_ollama_has_install_and_pull_steps() -> None:
    lines = launcher.ollama_advice(launcher.OllamaStatus(False, False, "refused"), "qwen2.5vl:3b")
    text = "\n".join(lines)
    assert "https://ollama.com/download" in text
    assert "ollama pull qwen2.5vl:3b" in text
    assert "dashboard still opens" in text


def test_advice_for_missing_model_is_just_the_pull() -> None:
    text = "\n".join(launcher.ollama_advice(launcher.OllamaStatus(True, False), "qwen2.5vl:7b"))
    assert "ollama pull qwen2.5vl:7b" in text and "ollama.com/download" not in text


def test_no_advice_when_ollama_is_ready() -> None:
    assert launcher.ollama_advice(launcher.OllamaStatus(True, True), "qwen2.5vl:3b") == []


# ---- browser -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("host", "url"),
    [
        ("127.0.0.1", "http://127.0.0.1:5050/dashboard"),
        ("0.0.0.0", "http://127.0.0.1:5050/dashboard"),
        ("::1", "http://[::1]:5050/dashboard"),
    ],
)
def test_dashboard_url(host: str, url: str) -> None:
    assert launcher.dashboard_url(host, 5050) == url


def test_wait_for_port_retries_until_the_server_answers() -> None:
    attempts: list[tuple[str, int]] = []

    class Conn:
        def close(self) -> None: ...

    def connect(address: tuple[str, int], timeout: float) -> Conn:
        attempts.append(address)
        if len(attempts) < 3:
            raise ConnectionRefusedError
        return Conn()

    assert launcher.wait_for_port("0.0.0.0", 5050, 10, connect=connect, sleep=lambda s: None)
    assert attempts == [("127.0.0.1", 5050)] * 3


def test_wait_for_port_gives_up_at_the_deadline() -> None:
    now = [0.0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    def refuse(address: tuple[str, int], timeout: float) -> Any:
        raise ConnectionRefusedError

    assert not launcher.wait_for_port("127.0.0.1", 1, 2.0, connect=refuse, sleep=sleep, clock=lambda: now[0])
    assert now[0] >= 2.0


def test_browser_opens_once_the_port_answers() -> None:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        opened: list[str] = []
        url = launcher.dashboard_url("127.0.0.1", port)
        launcher.open_browser_when_ready(url, "127.0.0.1", port, timeout=5, opener=opened.append).join(5)
    assert opened == [url]


def test_busy_port_error_sees_a_listening_socket() -> None:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        busy = REAL_BUSY_PORT_ERROR("127.0.0.1", server.getsockname()[1])
        assert isinstance(busy, OSError)
    assert REAL_BUSY_PORT_ERROR("127.0.0.1", launcher.free_port()) is None


def test_a_listener_on_every_address_makes_localhost_busy_too() -> None:
    # macOS: SO_REUSEADDR lets 127.0.0.1:port bind next to 0.0.0.0:port, so only the knock finds the listener.
    with socket.socket() as server:
        server.bind(("0.0.0.0", 0))
        server.listen()
        busy = REAL_BUSY_PORT_ERROR("127.0.0.1", server.getsockname()[1])
    assert isinstance(busy, OSError) and busy.errno in launcher.busy_errnos()


def test_an_address_this_computer_does_not_have_is_raised_not_called_busy() -> None:
    # 192.0.2.0/24 is TEST-NET-1: never assigned to a real interface (EADDRNOTAVAIL, errno 49 on macOS, 99 on Linux).
    with pytest.raises(OSError) as raised:
        REAL_BUSY_PORT_ERROR("192.0.2.55", launcher.free_port())
    assert raised.value.errno not in launcher.busy_errnos()


class _FailingSocket:
    """A socket whose bind fails with a given error (the platform-specific ones can't be produced for real)."""

    error: OSError = OSError()

    def __init__(self, *args: Any) -> None: ...

    def __enter__(self) -> _FailingSocket:
        return self

    def __exit__(self, *exc: object) -> None: ...

    def setsockopt(self, *args: Any) -> None: ...

    def bind(self, address: Any) -> None:
        raise self.error


def _windows_error(winerror: int) -> OSError:
    error = OSError(winerror, "socket error")
    error.winerror = winerror  # type: ignore[attr-defined]  # only Windows fills it in
    return error


@pytest.mark.parametrize(
    ("platform", "error", "busy"),
    [
        ("darwin", OSError(errno.EADDRINUSE, "Address already in use"), True),
        ("darwin", OSError(errno.EADDRNOTAVAIL, "Can't assign requested address"), False),
        ("darwin", socket.gaierror(8, "nodename nor servname provided"), False),
        ("win32", _windows_error(10048), True),
        ("win32", _windows_error(10013), True),  # a port inside a range Hyper-V reserved
        ("win32", _windows_error(10049), False),  # WSAEADDRNOTAVAIL
    ],
)
def test_only_an_address_in_use_counts_as_a_busy_port(
    platform: str, error: OSError, busy: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_FailingSocket, "error", error)
    monkeypatch.setattr(launcher.socket, "socket", _FailingSocket)
    monkeypatch.setattr(launcher.sys, "platform", platform)
    if busy:
        assert REAL_BUSY_PORT_ERROR("127.0.0.1", 5050) is error
    else:
        with pytest.raises(OSError) as raised:
            REAL_BUSY_PORT_ERROR("127.0.0.1", 5050)
        assert raised.value is error


# ---- self-test ---------------------------------------------------------------------------------------


def test_self_test_serves_the_real_pages(fakes: app_module.Services) -> None:
    lines: list[str] = []
    assert launcher.run_self_test(services=fakes, bundle=False, out=lines.append) == 0
    text = "\n".join(lines)
    assert text.endswith("SELF-TEST OK")
    for name in ("api/status", "dashboard", "projector page", "static files", "socket.io"):
        assert f"ok   {name}" in text
    assert os.environ["SAR_VLM_BACKEND"] == "mock" and os.environ["SAR_TTS_BACKEND"] == "none"


def test_self_test_fails_on_a_broken_page(fakes: app_module.Services) -> None:
    def get(url: str, timeout: float) -> Any:
        if url.endswith("/dashboard"):
            return launcher.HttpResponse(500, "boom")
        return launcher.http_get(url, timeout)

    lines: list[str] = []
    assert launcher.run_self_test(services=fakes, bundle=False, get=get, out=lines.append) == 1
    assert any(line.startswith("  FAIL dashboard: CheckFailed: GET /dashboard returned 500") for line in lines)
    assert lines[-1] == "SELF-TEST FAILED: dashboard"


def test_self_test_reports_a_crash_during_startup(fakes: app_module.Services, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("templates missing")

    monkeypatch.setattr(app_module, "create_app", broken)
    lines: list[str] = []
    assert launcher.run_self_test(services=fakes, bundle=False, out=lines.append) == 1
    assert lines[-1] == "SELF-TEST FAILED: RuntimeError: templates missing"


def test_self_test_gives_up_after_its_timeout(fakes: app_module.Services) -> None:
    release = threading.Event()

    def hang(url: str, timeout: float) -> Any:
        release.wait(10)
        return launcher.HttpResponse(200, "")

    lines: list[str] = []
    try:
        assert launcher.run_self_test(timeout_s=0.5, services=fakes, bundle=False, get=hang, out=lines.append) == 1
    finally:
        release.set()
        # Let the abandoned run finish, so it stops its server and deletes its temp folder.
        for thread in threading.enumerate():
            if thread.name == "self-test":
                thread.join(10)
    assert any("no result after" in line for line in lines)


def test_self_test_keeps_every_data_file_in_its_temp_folder(tmp_path: Path) -> None:
    data_dir = tmp_path / "self-test"
    settings = launcher.self_test_settings(data_dir, 5051)
    paths = {f.name: getattr(settings, f.name) for f in dataclasses.fields(settings)}
    files = {name: path for name, path in paths.items() if isinstance(path, Path)}
    assert "accessibility_file" in files
    outside = sorted(name for name, path in files.items() if not path.is_relative_to(data_dir))
    assert outside == [], "the self-test never reads or writes the teacher's own data folder"


def test_page_checks_need_the_app_title() -> None:
    def get(url: str, timeout: float) -> Any:
        return launcher.HttpResponse(200, "<html><title>Index of /</title></html>")

    dashboard = next(c for c in launcher.http_checks("http://x", get) if c.name == "dashboard")
    with pytest.raises(launcher.CheckFailed, match="Index of /"):
        dashboard.run()


def test_socketio_check_needs_an_engineio_handshake() -> None:
    def get(url: str, timeout: float) -> Any:
        return launcher.HttpResponse(200, "<html>not socket.io</html>")

    check = next(c for c in launcher.http_checks("http://x", get) if c.name == "socket.io")
    with pytest.raises(launcher.CheckFailed, match="handshake"):
        check.run()


def test_status_check_needs_the_mock_backend() -> None:
    def get(url: str, timeout: float) -> Any:
        return launcher.HttpResponse(200, json.dumps({"vlm_backend": "ollama"}))

    check = next(c for c in launcher.http_checks("http://x", get) if c.name == "api/status")
    with pytest.raises(launcher.CheckFailed, match="ollama"):
        check.run()


def test_page_title_unescapes_entities() -> None:
    assert launcher.page_title("<TITLE>\n  Whiteboard Assistant &ndash; projector </TITLE>") == (
        "Whiteboard Assistant – projector"
    )


def test_bundle_check_reports_every_missing_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launcher, "RUNTIME_MODULES", ("json", "no_such_module_a", "no_such_module_b"))
    imports = next(c for c in launcher.bundle_checks(None) if c.name == "imports")
    with pytest.raises(launcher.CheckFailed, match=r"no_such_module_a.*no_such_module_b"):
        imports.run()


def _fake_grading(monkeypatch: pytest.MonkeyPatch, grade: Any) -> None:
    module = ModuleType("biobench_sar.grading")
    module.__spec__ = importlib.machinery.ModuleSpec("biobench_sar.grading", None)
    module.grade = grade  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "biobench_sar.grading", module)


def _grading_check() -> launcher.Check:
    return next(c for c in launcher.bundle_checks(None) if c.name == "grading")


def test_bundle_check_solves_with_sympy_and_grades(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "sympy" in launcher.RUNTIME_MODULES
    _fake_grading(monkeypatch, lambda student, expected: student.replace(" ", "") == expected.replace(" ", ""))
    detail = _grading_check().run()
    assert detail.startswith("sympy ") and "grade() marks a right and a wrong answer" in detail


def test_bundle_check_fails_when_grading_cannot_tell_right_from_wrong(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_grading(monkeypatch, lambda student, expected: None)
    with pytest.raises(launcher.CheckFailed, match=r"\(None, None\)"):
        _grading_check().run()


def test_bundle_check_without_a_grading_module_still_checks_sympy(monkeypatch: pytest.MonkeyPatch) -> None:
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a: None if name == "biobench_sar.grading" else find_spec(name, *a)
    )
    assert "no biobench_sar.grading" in _grading_check().run()


def test_utf8_mode_is_required_in_a_frozen_build_only() -> None:
    assert launcher.utf8_mode_detail(frozen=True, utf8_mode=1) == "on"
    assert launcher.utf8_mode_detail(frozen=False, utf8_mode=0).startswith("off")
    with pytest.raises(launcher.CheckFailed, match="X utf8"):
        launcher.utf8_mode_detail(frozen=True, utf8_mode=0)


def test_spec_turns_on_utf8_mode_with_the_option_the_bootloader_knows() -> None:
    # The bootloader ignores unknown keys silently: "X utf8_mode=1" did nothing.
    spec = (LAUNCHER_PATH.parent / "sar.spec").read_text(encoding="utf-8")
    assert '("X utf8", None, "OPTION")' in spec and '("X utf8_mode' not in spec


def test_spec_bundles_sympy_for_quiz_grading() -> None:
    spec = (LAUNCHER_PATH.parent / "sar.spec").read_text(encoding="utf-8")
    hidden = spec[spec.index("hiddenimports = [") : spec.index("if IS_WIN:")]
    for module in ('"sympy"', '"sympy.parsing.sympy_parser"', '"mpmath"'):
        assert module in hidden
    assert '_existing("biobench_sar.grading")' in spec
    assert 'collect_submodules("biobench_sar"' in spec


def test_requirements_pin_sympy_and_explain_the_missing_speechrecognition() -> None:
    lines = (LAUNCHER_PATH.parent.parent / "requirements.txt").read_text(encoding="utf-8").splitlines()
    packages = [re.split(r"[<>=]", line)[0].strip().lower() for line in lines if line and not line.startswith("#")]
    assert "sympy" in packages
    assert "speechrecognition" not in packages
    notes = " ".join(line for line in lines if line.startswith("#"))
    assert "SpeechRecognition" in notes and "PyAudio" in notes and "Whisper" in notes


def test_macos_minimum_matches_the_build_runner_and_the_docs() -> None:
    # pip picks wheels built for the runner's macOS, so the app needs at least that version.
    root = LAUNCHER_PATH.parent.parent
    spec = (LAUNCHER_PATH.parent / "sar.spec").read_text(encoding="utf-8")
    workflow = (root / ".github" / "workflows" / "build.yml").read_text(encoding="utf-8")
    docs = (root / "docs" / "DESKTOP_BUILDS.md").read_text(encoding="utf-8")
    runner = re.search(r"os: macos-(\d+)", workflow)
    minimum = re.search(r'"LSMinimumSystemVersion": "(\d+)\.0"', spec)
    assert runner and minimum and minimum.group(1) == runner.group(1)
    assert f"| macOS {minimum.group(1)} (" in docs and "macOS 12" not in docs


def test_torch_on_cpu_hides_mps_only_inside_the_block() -> None:
    torch = pytest.importorskip("torch")
    before = torch.backends.mps.is_available()
    with launcher.torch_on_cpu():
        assert torch.backends.mps.is_available() is False
    assert torch.backends.mps.is_available() == before


@pytest.mark.skipif(os.environ.get("SAR_SLOW") != "1", reason="set SAR_SLOW=1 (loads chromadb, torch, whisper)")
def test_full_self_test_with_real_services(folders: dict[str, Path]) -> None:
    launcher.prepare_environment(frozen=False)
    lines: list[str] = []
    assert launcher.run_self_test(out=lines.append) == 0, "\n".join(lines)


# ---- command line --------------------------------------------------------------------------------------


def test_flags() -> None:
    args = launcher.parse_args(["--no-browser", "--port", "6000", "-psn_0_12345"])
    assert args.no_browser and args.port == 6000 and not args.self_test
    assert launcher.parse_args(["--self-test"]).self_test


@pytest.mark.parametrize("port", ["0", "70000", "http"])
def test_bad_port_is_rejected(port: str) -> None:
    with pytest.raises(SystemExit):
        launcher.parse_args(["--port", port])


def test_main_self_test_skips_the_app(folders: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launcher, "run_self_test", lambda: 0)
    monkeypatch.setattr(launcher, "run_app", lambda **kwargs: pytest.fail("self-test must not start the app"))
    assert launcher.main(["--self-test", "--port", "6001"]) == 0
    assert "SAR_PORT" not in os.environ


def test_main_port_flag_beats_sar_env(folders: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    (folders["data"] / "sar.env").write_text("SAR_PORT=6060\n")
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(launcher, "run_app", lambda **kwargs: calls.append(kwargs) or 0)
    assert launcher.main(["--port", "6001", "--no-browser"]) == 0
    assert os.environ["SAR_PORT"] == "6001"
    assert calls == [{"open_browser": False, "env_files": [folders["data"] / "sar.env"]}]


def test_run_app_warns_about_ollama_and_still_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[bool] = []
    monkeypatch.setenv("SAR_VLM_BACKEND", "ollama")
    monkeypatch.setattr(launcher, "check_ollama", lambda url, model: launcher.OllamaStatus(False, False, "refused"))
    monkeypatch.setattr(app_module, "main", lambda: started.append(True))
    lines: list[str] = []
    assert launcher.run_app(open_browser=False, out=lines.append) == 0
    assert started == [True]
    assert any("https://ollama.com/download" in line for line in lines)


def test_run_app_shows_the_network_address_when_serving_the_lan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAR_HOST", "0.0.0.0")
    monkeypatch.setenv("SAR_VLM_BACKEND", "mock")
    monkeypatch.setattr(launcher, "lan_address", lambda: "192.168.1.20")
    monkeypatch.setattr(app_module, "main", lambda: None)
    lines: list[str] = []
    assert launcher.run_app(open_browser=False, out=lines.append) == 0
    assert "  Dashboard    http://127.0.0.1:5050/dashboard" in lines
    assert any(line.endswith("http://192.168.1.20:5050/dashboard") for line in lines)


def test_run_app_keeps_quiet_about_the_lan_on_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAR_VLM_BACKEND", "mock")
    monkeypatch.setattr(launcher, "lan_address", lambda: pytest.fail("localhost only"))
    monkeypatch.setattr(app_module, "main", lambda: None)
    assert launcher.run_app(open_browser=False, out=lambda line: None) == 0


def test_lan_address_is_never_loopback() -> None:
    address = launcher.lan_address()
    assert address is None or not address.startswith("127.")


def test_run_app_skips_the_ollama_check_for_other_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAR_VLM_BACKEND", "mock")
    monkeypatch.setattr(launcher, "check_ollama", lambda *a: pytest.fail("mock backend needs no Ollama"))
    monkeypatch.setattr(app_module, "main", lambda: None)
    assert launcher.run_app(open_browser=False, out=lambda line: None) == 0


def test_run_app_passes_on_the_app_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    def busy_port() -> None:
        raise SystemExit(1)

    monkeypatch.setenv("SAR_VLM_BACKEND", "mock")
    monkeypatch.setattr(app_module, "main", busy_port)
    assert launcher.run_app(open_browser=False, out=lambda line: None) == 1


def test_a_crash_in_the_app_is_reported_and_exits_1(monkeypatch: pytest.MonkeyPatch, alerts: list[str]) -> None:
    def crash() -> None:
        raise RuntimeError("camera exploded")

    monkeypatch.setenv("SAR_VLM_BACKEND", "mock")
    monkeypatch.setattr(app_module, "main", crash)
    lines: list[str] = []
    assert launcher.run_guarded(lambda: launcher.run_app(open_browser=False, out=lines.append), out=lines.append) == 1
    text = "\n".join(lines)
    assert "stopped because of an unexpected error" in text
    assert "Technical detail: RuntimeError: camera exploded" in text
    log = launcher.log_file()
    assert lines[-1] == f"Details are saved in {log}"
    assert "Traceback" in log.read_text(encoding="utf-8") and "camera exploded" in log.read_text(encoding="utf-8")
    assert alerts and str(log) in alerts[0]


def test_a_busy_port_stops_with_plain_advice(monkeypatch: pytest.MonkeyPatch, alerts: list[str]) -> None:
    monkeypatch.setattr(launcher, "busy_port_error", REAL_BUSY_PORT_ERROR)
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        monkeypatch.setenv("SAR_PORT", str(port))
        monkeypatch.setenv("SAR_VLM_BACKEND", "mock")
        monkeypatch.setattr(launcher, "open_browser_when_ready", lambda *a, **k: pytest.fail("someone else's port"))
        monkeypatch.setattr(app_module, "main", lambda: pytest.fail("the app must not start on a busy port"))
        lines: list[str] = []
        assert launcher.run_guarded(lambda: launcher.run_app(out=lines.append), out=lines.append) == 1
    text = "\n".join(lines)
    assert f"{launcher.APP_NAME} could not start." in text
    assert f"Port {port} is already in use" in text and "SAR_PORT=5051" in text
    assert alerts and f"Port {port} is already in use" in alerts[0]
    # The real bind error, not the advice again: the errno tells a helper what happened.
    assert "Technical detail: OSError: [Errno" in text


def test_an_old_lan_address_is_not_called_a_busy_port(monkeypatch: pytest.MonkeyPatch) -> None:
    # The laptop's Wi-Fi address from yesterday, after DHCP gave it a new one.
    monkeypatch.setattr(launcher, "busy_port_error", REAL_BUSY_PORT_ERROR)
    monkeypatch.setenv("SAR_HOST", "192.0.2.55")
    monkeypatch.setenv("SAR_VLM_BACKEND", "mock")
    monkeypatch.setattr(app_module, "main", lambda: pytest.fail("must not start"))
    lines: list[str] = []
    assert launcher.run_guarded(lambda: launcher.run_app(open_browser=False, out=lines.append), out=lines.append) == 1
    text = "\n".join(lines)
    assert "192.0.2.55:5050 is not an address this computer can serve on" in text and "SAR_HOST" in text
    assert "already in use" not in text


def test_a_host_that_isnt_an_address_is_not_called_a_busy_port(monkeypatch: pytest.MonkeyPatch) -> None:
    def unresolvable(host: str, port: int) -> OSError | None:
        raise socket.gaierror(8, "nodename nor servname provided, or not known")

    monkeypatch.setenv("SAR_HOST", "my-laptop.local:5050")
    monkeypatch.setenv("SAR_VLM_BACKEND", "mock")
    monkeypatch.setattr(launcher, "busy_port_error", unresolvable)
    monkeypatch.setattr(app_module, "main", lambda: pytest.fail("must not start"))
    lines: list[str] = []
    assert launcher.run_guarded(lambda: launcher.run_app(open_browser=False, out=lines.append), out=lines.append) == 1
    text = "\n".join(lines)
    assert "my-laptop.local:5050:5050 is not an address" in text and "SAR_HOST" in text
    assert "already in use" not in text


def _settings() -> object:  # reads its numbers like config.Settings does
    return (float(os.environ.get("SAR_RECORD_SECONDS", "5")), int(os.environ.get("SAR_CAMERA_INDEX", "0")))


def _advice(lines: list[str]) -> str:
    return next(line for line in lines if line.startswith(("This setting", "These settings")))


def test_a_settings_typo_names_the_bad_line_and_its_file(
    folders: dict[str, Path], monkeypatch: pytest.MonkeyPatch, alerts: list[str]
) -> None:
    good, bad = folders["exe"] / "sar.env", folders["data"] / "sar.env"
    good.write_text("SAR_CAMERA_INDEX=2\n")
    bad.write_text("SAR_RECORD_SECONDS=5s\n")
    loaded = launcher.prepare_environment(frozen=False)
    assert loaded == [good, bad]
    monkeypatch.setattr(config, "Settings", _settings)
    monkeypatch.setattr(app_module, "main", lambda: pytest.fail("must not start"))
    lines: list[str] = []
    run = lambda: launcher.run_app(open_browser=False, out=lines.append, env_files=loaded)  # noqa: E731
    assert launcher.run_guarded(run, out=lines.append) == 1
    advice = _advice(lines)
    assert "SAR_RECORD_SECONDS=5s" in advice and f"the settings file {bad}." in advice
    assert str(good) not in advice, "only the file the bad value came from"
    assert "SAR_CAMERA_INDEX" not in advice
    assert "Technical detail: ValueError" in "\n".join(lines)
    assert os.environ["SAR_RECORD_SECONDS"] == "5s" and os.environ["SAR_CAMERA_INDEX"] == "2", "env restored"
    assert alerts


def test_a_bad_environment_variable_is_not_blamed_on_sar_env(
    folders: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    (folders["data"] / "sar.env").write_text("SAR_CAMERA_INDEX=2\nSAR_RECORD_SECONDS=4\n")
    monkeypatch.setenv("SAR_RECORD_SECONDS", "5s")  # the real environment beats the file
    loaded = launcher.prepare_environment(frozen=False)
    monkeypatch.setattr(config, "Settings", _settings)
    monkeypatch.setattr(app_module, "main", lambda: pytest.fail("must not start"))
    lines: list[str] = []
    run = lambda: launcher.run_app(open_browser=False, out=lines.append, env_files=loaded)  # noqa: E731
    assert launcher.run_guarded(run, out=lines.append) == 1
    advice = _advice(lines)
    assert "SAR_RECORD_SECONDS=5s" in advice and "the environment variables (not in a sar.env)" in advice
    assert "sar.env (" not in advice and str(folders["data"]) not in advice


def test_bad_values_from_different_places_each_name_their_own(
    folders: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    (folders["data"] / "sar.env").write_text("SAR_CAMERA_INDEX=two\n")
    monkeypatch.setenv("SAR_RECORD_SECONDS", "5s")
    launcher.prepare_environment(frozen=False)
    monkeypatch.setattr(config, "Settings", _settings)
    advice = launcher.settings_advice([folders["data"] / "sar.env"])
    assert advice.startswith(
        "These settings have values the app can't read: SAR_CAMERA_INDEX=two, SAR_RECORD_SECONDS=5s."
    )
    assert f"SAR_CAMERA_INDEX in the settings file {folders['data'] / 'sar.env'}" in advice
    assert "SAR_RECORD_SECONDS in the environment variables" in advice


def test_bad_values_from_one_file_name_it_once(folders: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = folders["data"] / "sar.env"
    env_file.write_text("SAR_CAMERA_INDEX=two\nSAR_RECORD_SECONDS=5s\n")
    launcher.prepare_environment(frozen=False)
    monkeypatch.setattr(config, "Settings", _settings)
    advice = launcher.settings_advice([env_file])
    assert advice.count(str(env_file)) == 1 and "Fix or delete them in the settings file" in advice


def test_the_reported_sar_vlm_timeout_typo_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAR_VLM_TIMEOUT", "90s")
    monkeypatch.setenv("SAR_VLM_BACKEND", "mock")
    try:
        config.Settings()
    except ValueError:
        pass
    else:
        pytest.skip("config.Settings now accepts SAR_VLM_TIMEOUT=90s")
    lines: list[str] = []
    assert launcher.run_guarded(lambda: launcher.run_app(open_browser=False, out=lines.append), out=lines.append) == 1
    assert any("SAR_VLM_TIMEOUT=90s" in line for line in lines)


def test_unreadable_settings_finds_every_bad_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    def settings() -> object:
        return [int(os.environ.get(key, "0")) for key in ("SAR_A", "SAR_B", "SAR_C")]

    monkeypatch.setenv("SAR_A", "x")
    monkeypatch.setenv("SAR_B", "1")
    monkeypatch.setenv("SAR_C", "y")
    assert launcher.unreadable_settings(settings) == ["SAR_A", "SAR_C"]
    assert (os.environ["SAR_A"], os.environ["SAR_B"], os.environ["SAR_C"]) == ("x", "1", "y")


def test_unreadable_settings_blames_no_variable_when_settings_fail_anyway(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> object:
        raise RuntimeError("not about the settings")

    monkeypatch.setenv("SAR_A", "1")
    assert launcher.unreadable_settings(broken) == []
    assert os.environ["SAR_A"] == "1"


def test_an_unreadable_sar_env_is_reported(folders: dict[str, Path], alerts: list[str]) -> None:
    env_file = folders["data"] / "sar.env"
    env_file.write_bytes("# cámara\nSAR_CAMERA_INDEX=1\n".encode("cp1252"))  # Notepad's "ANSI"
    lines: list[str] = []
    assert launcher.run_guarded(lambda: launcher.main(["--no-browser"]), out=lines.append) == 1
    text = "\n".join(lines)
    assert f"The settings file {env_file} could not be read" in text and "UTF-8" in text
    assert "UnicodeDecodeError" in text
    assert (folders["data"] / "logs" / "sar.log").is_file()


def test_an_app_that_cannot_be_imported_says_to_reinstall(monkeypatch: pytest.MonkeyPatch) -> None:
    import biobench_sar

    monkeypatch.setenv("SAR_VLM_BACKEND", "mock")
    monkeypatch.delattr(biobench_sar, "app")
    monkeypatch.setitem(sys.modules, "biobench_sar.app", None)
    lines: list[str] = []
    assert launcher.run_guarded(lambda: launcher.run_app(open_browser=False, out=lines.append), out=lines.append) == 1
    assert any("pip install -r requirements.txt" in line for line in lines)
    assert any(line.startswith("Technical detail: ModuleNotFoundError") for line in lines)


@pytest.mark.parametrize(("exit", "code"), [(SystemExit(3), 3), (SystemExit(None), 0), (KeyboardInterrupt(), 0)])
def test_run_guarded_passes_exit_codes_through(exit: BaseException, code: int) -> None:
    def entry() -> int:
        raise exit

    assert launcher.run_guarded(entry, out=lambda line: pytest.fail("nothing to report")) == code


def test_run_guarded_prints_a_text_exit() -> None:
    def entry() -> int:
        raise SystemExit("bad flag")

    lines: list[str] = []
    assert launcher.run_guarded(entry, out=lines.append) == 1
    assert lines == ["bad flag"]


def test_failure_log_is_the_redirected_log_when_output_already_goes_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "sar.log"
    monkeypatch.setattr(launcher, "_log_redirect", log)
    assert launcher.save_failure_log("report") == log
    assert not log.exists(), "the report was already printed into it"


def test_failure_without_a_writable_log_still_explains(monkeypatch: pytest.MonkeyPatch, alerts: list[str]) -> None:
    def no_log() -> Path:
        raise PermissionError("read-only home")

    monkeypatch.setattr(launcher, "log_file", no_log)
    lines: list[str] = []
    assert launcher.report_failure(launcher.StartupFailed("Do this."), out=lines.append) == 1
    assert lines[-2:] == [
        "Technical detail: StartupFailed: Do this.",
        "The details above could not be saved to a log file.",
    ]
    assert alerts == [f"{launcher.APP_NAME} could not start.\n\nDo this."]


def test_startup_failure_keeps_the_console_open_and_shows_a_dialog(
    monkeypatch: pytest.MonkeyPatch, alerts: list[str]
) -> None:
    def main() -> int:
        with launcher.startup_step("Check sar.env."):
            raise ValueError("could not convert string to float: '90s'")

    pauses: list[int] = []
    exits: list[int] = []
    monkeypatch.setattr(launcher, "main", main)
    monkeypatch.setattr(launcher, "_utf8_output", lambda: None)
    monkeypatch.setattr(launcher, "_pause_if_console_closes", pauses.append)
    monkeypatch.setattr(launcher.os, "_exit", exits.append)
    launcher._run()
    assert exits == [1] and pauses == [1]
    assert alerts and "Check sar.env." in alerts[0] and str(launcher.log_file()) in alerts[0]
    assert "could not convert string to float" in launcher.log_file().read_text(encoding="utf-8")


class _Stdin:
    def __init__(self, tty: bool) -> None:
        self.tty = tty

    def isatty(self) -> bool:
        return self.tty


@pytest.mark.parametrize(
    ("code", "platform", "tty", "paused"),
    [(1, "win32", True, True), (0, "win32", True, False), (1, "win32", False, False), (1, "darwin", True, False)],
)
def test_pause_only_keeps_a_failed_windows_console_open(
    code: int, platform: str, tty: bool, paused: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts: list[str] = []
    monkeypatch.setattr(launcher, "is_frozen", lambda: True)
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(sys, "stdin", _Stdin(tty))
    monkeypatch.setattr(builtins, "input", lambda prompt="": prompts.append(prompt) or "")
    launcher._pause_if_console_closes(code)
    assert prompts == (["\nPress Enter to close this window..."] if paused else [])


def test_alert_is_silent_outside_a_finder_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: pytest.fail("no dialog from a terminal"))
    launcher.alert("hello")


def test_applescript_strings_are_escaped() -> None:
    assert launcher._applescript_string('say "hi" \\ bye') == '"say \\"hi\\" \\\\ bye"'


def test_log_redirect_keeps_the_previous_run(folders: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    logs = folders["data"] / "logs"
    logs.mkdir()
    (logs / "sar.log").write_text("old run\n")
    path = launcher.redirect_output_to_log()
    print("new run")
    sys.stdout.close()
    assert path == logs / "sar.log"
    assert path.read_text() == "new run\n"
    assert (logs / "sar.previous.log").read_text() == "old run\n"


class _Stream:
    def __init__(self, tty: bool) -> None:
        self.tty = tty
        self.reconfigured: dict[str, str] = {}

    def isatty(self) -> bool:
        return self.tty

    def reconfigure(self, **kwargs: str) -> None:
        self.reconfigured = kwargs


def test_piped_output_is_switched_to_utf8_but_consoles_are_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe, console = _Stream(tty=False), _Stream(tty=True)
    monkeypatch.setattr(sys, "stdout", pipe)
    monkeypatch.setattr(sys, "stderr", console)
    launcher._utf8_output()
    assert pipe.reconfigured == {"encoding": "utf-8", "errors": "replace"}
    assert console.reconfigured == {}


def test_page_title_accepts_attributes_on_the_title_tag() -> None:
    assert launcher.page_title('<title data-i18n="page.title">Whiteboard Assistant</title>') == "Whiteboard Assistant"
    assert launcher.page_title("<TITLE>Plain</TITLE>") == "Plain"
    assert launcher.page_title("<header><titles>no</titles></header>") == ""


# ---- macOS app check (packaging/check_macos_app.py) ------------------------------------------------------

OTOOL_SAMPLE = """\
/App/Contents/Frameworks/torch/lib/libtorch_cpu.dylib:
Load command 0
      cmd LC_SEGMENT_64
  cmdsize 72
Load command 9
      cmd LC_BUILD_VERSION
  cmdsize 32
 platform 1
    minos 14.0
      sdk 14.5
Load command 10
      cmd LC_SOURCE_VERSION
  cmdsize 16
  version 99.0
/App/Contents/Frameworks/old.so (architecture x86_64):
Load command 4
      cmd LC_VERSION_MIN_MACOSX
  cmdsize 16
  version 10.9
      sdk 10.14
/App/Contents/Frameworks/old.so (architecture arm64):
Load command 4
      cmd LC_BUILD_VERSION
  cmdsize 32
    minos 11.0
"""


def test_otool_parsing_takes_each_binarys_newest_slice_and_ignores_other_versions() -> None:
    assert checker.parse_minimum_os(OTOOL_SAMPLE) == {
        "/App/Contents/Frameworks/torch/lib/libtorch_cpu.dylib": (14, 0),
        "/App/Contents/Frameworks/old.so": (11, 0),
    }


def _fake_app(tmp_path: Path, **plist: Any) -> Path:
    import plistlib

    app = tmp_path / "SAR Whiteboard Assistant.app"
    (app / "Contents" / "Frameworks").mkdir(parents=True)
    info = {
        "CFBundleIdentifier": checker.BUNDLE_ID,
        "CFBundleShortVersionString": "0.1.0",
        "LSMinimumSystemVersion": "14.0",
        "NSCameraUsageDescription": "Watches the whiteboard.",
        "NSMicrophoneUsageDescription": "Listens for questions.",
    }
    info.update(plist)
    (app / "Contents" / "Info.plist").write_bytes(plistlib.dumps({k: v for k, v in info.items() if v is not None}))
    (app / "Contents" / "Frameworks" / "torch.so").write_bytes(bytes.fromhex("cffaedfe") + b"\0" * 28)
    (app / "Contents" / "Frameworks" / "notes.txt").write_text("not a binary")
    return app


def _otool_needing(version: str) -> Any:
    def otool(paths: Any) -> str:
        assert [p.name for p in paths] == ["torch.so"], "only Mach-O files go to otool"
        return f"{paths[0]}:\n      cmd LC_BUILD_VERSION\n    minos {version}\n"

    return otool


def test_macos_check_passes_a_good_app(tmp_path: Path) -> None:
    errors, report = checker.check_app(_fake_app(tmp_path), "0.1.0", otool=_otool_needing("14.0"))
    assert errors == []
    assert "NSCameraUsageDescription: Watches the whiteboard." in report
    assert report[-1] == "LSMinimumSystemVersion 14.0; 1 binaries need at most macOS 14.0"


@pytest.mark.parametrize("key", ["NSCameraUsageDescription", "NSMicrophoneUsageDescription"])
def test_macos_check_fails_without_a_usage_string(key: str, tmp_path: Path) -> None:
    errors, _ = checker.check_app(_fake_app(tmp_path, **{key: None}), "0.1.0", otool=_otool_needing("14.0"))
    assert errors == [f"{key} is missing or empty in Info.plist: macOS would refuse access without asking"]


def test_macos_check_fails_when_a_binary_needs_a_newer_macos(tmp_path: Path) -> None:
    # The original bug: Info.plist said 12.0 while the bundled wheels were built for macOS 14.
    errors, _ = checker.check_app(
        _fake_app(tmp_path, LSMinimumSystemVersion="12.0"), "0.1.0", otool=_otool_needing("14.0")
    )
    assert len(errors) == 1 and "need a newer macOS: torch.so (14.0)" in errors[0]


def test_macos_check_fails_on_a_wrong_version_or_bundle_id(tmp_path: Path) -> None:
    app = _fake_app(tmp_path, CFBundleIdentifier="org.example")
    errors, _ = checker.check_app(app, "0.2.0", otool=_otool_needing("14.0"))
    assert any("bundle id" in e for e in errors) and any("expected '0.2.0'" in e for e in errors)


def test_macos_check_main_prints_github_errors_and_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(checker, "run_otool", _otool_needing("14.0"))
    app = _fake_app(tmp_path, NSMicrophoneUsageDescription="  ")
    assert checker.main([str(app), "--version", "0.1.0"]) == 1
    assert "::error::NSMicrophoneUsageDescription is missing or empty" in capsys.readouterr().out
    assert checker.main([str(_fake_app(tmp_path / "good")), "--version", "0.1.0"]) == 0
