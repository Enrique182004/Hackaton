from __future__ import annotations

import importlib.util
import json
import os
import socket
import sys
import threading
import urllib.error
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from biobench_sar import app as app_module
from biobench_sar import config

LAUNCHER_PATH = Path(__file__).resolve().parent.parent / "packaging" / "launcher.py"
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
)


def _load_launcher() -> ModuleType:
    # By path: a `packaging` import would find the PyPI package of that name, not this folder.
    spec = importlib.util.spec_from_file_location("sar_launcher", LAUNCHER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


launcher = _load_launcher()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whatever the launcher sets in os.environ / config.DATA_DIR is undone after each test."""
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(config, "DATA_DIR", config.DATA_DIR)


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


def test_ollama_without_the_model_is_reachable_but_not_ready() -> None:
    body = json.dumps({"models": [{"name": "llama3:latest"}]}).encode()
    status = launcher.check_ollama(fetch=lambda url, timeout: body)
    assert status.reachable and not status.has_model


def test_untagged_model_names_match_latest() -> None:
    body = json.dumps({"models": [{"name": "llava:latest"}]}).encode()
    assert launcher.check_ollama(model="llava", fetch=lambda url, timeout: body).has_model


@pytest.mark.parametrize(
    "failure", [urllib.error.URLError("connection refused"), TimeoutError("timed out"), b"not json"]
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


def test_port_free_sees_a_listening_socket() -> None:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        assert not launcher.port_free("127.0.0.1", server.getsockname()[1])
    assert launcher.port_free("127.0.0.1", launcher.free_port())


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
    assert calls == [{"open_browser": False}]


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


def test_run_app_turns_a_crash_into_exit_code_1(monkeypatch: pytest.MonkeyPatch) -> None:
    def crash() -> None:
        raise RuntimeError("camera exploded")

    monkeypatch.setenv("SAR_VLM_BACKEND", "mock")
    monkeypatch.setattr(app_module, "main", crash)
    lines: list[str] = []
    assert launcher.run_app(open_browser=False, out=lines.append) == 1
    assert "RuntimeError: camera exploded" in lines[-1]


def test_run_app_doesnt_open_a_browser_on_a_busy_port(monkeypatch: pytest.MonkeyPatch) -> None:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        monkeypatch.setenv("SAR_PORT", str(server.getsockname()[1]))
        monkeypatch.setenv("SAR_VLM_BACKEND", "mock")
        monkeypatch.setattr(launcher, "open_browser_when_ready", lambda *a, **k: pytest.fail("someone else's port"))
        monkeypatch.setattr(app_module, "main", lambda: None)
        assert launcher.run_app(open_browser=True, out=lambda line: None) == 0


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
