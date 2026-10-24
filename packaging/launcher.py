"""Entry point of the desktop builds; also runs from a checkout: `python packaging/launcher.py [--self-test]`.

Before biobench_sar.app is imported it settles the environment Settings() reads: sar.env files, the per-user
data folder and (in a frozen build) the bundled embedding model. Then it checks Ollama, opens the dashboard
in the default browser once the server answers, and hands over to biobench_sar.app.main().
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import html
import json
import logging
import multiprocessing
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
import webbrowser
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if not getattr(sys, "frozen", False):
    # `python packaging/launcher.py` puts packaging/ on sys.path, not the checkout that holds biobench_sar.
    _CHECKOUT = str(Path(__file__).resolve().parent.parent)
    if _CHECKOUT not in sys.path:
        sys.path.insert(0, _CHECKOUT)

from biobench_sar import __version__
from biobench_sar.paths import APP_NAME, executable_dir, is_frozen, resource_dir, user_data_dir

ENV_FILE_NAME = "sar.env"
OLLAMA_DOWNLOAD_URL = "https://ollama.com/download"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_VLM_MODEL = "qwen2.5vl:3b"
OLLAMA_TIMEOUT_S = 2.0
BROWSER_WAIT_S = 180.0
SELF_TEST_TIMEOUT_S = 120.0
SELF_TEST_ENV = {"SAR_VLM_BACKEND": "mock", "SAR_TTS_BACKEND": "none", "SAR_LISTEN_MODE": "push"}
PAGE_TITLE = "Whiteboard Assistant"
# Imported lazily by the app (uploads, OCR, TTS, audio), so a build missing one only fails when a teacher
# uses that feature; the self-test imports them up front.
RUNTIME_MODULES = (
    "cv2",
    "numpy",
    "chromadb",
    "sentence_transformers",
    "torch",
    "whisper",
    "pyaudio",
    "edge_tts",
    "pyttsx3",
    "pymupdf",
    "docx",
    "pptx",
    "pandas",
    "pytesseract",
    "PIL",
    "requests",
)

Out = Callable[[str], None]


def say(line: str) -> None:
    """print() that survives consoles whose encoding can't show every character (cp1252 pipes on Windows)."""
    stream = sys.stdout
    if stream is None:
        return
    encoding = getattr(stream, "encoding", None) or "utf-8"
    with contextlib.suppress(OSError, ValueError):
        print(line.encode(encoding, "replace").decode(encoding), file=stream, flush=True)


# ---- startup failures ----------------------------------------------------------------------------------
# A double-clicked app has no terminal: an uncaught error would close the Windows console (or end the macOS
# app) before anyone can read it, so each startup step says in plain words what went wrong and what to do.


class StartupFailed(Exception):
    """A startup step failed; the message is advice for the teacher, the cause (__cause__) the technical error."""


@contextlib.contextmanager
def startup_step(advice: str | Callable[[], str]) -> Iterator[None]:
    """Re-raise any error inside the block as StartupFailed(advice), chained to the original error."""
    try:
        yield
    except StartupFailed:
        raise
    except Exception as exc:
        if callable(advice):
            try:
                advice = advice()
            except Exception:
                advice = "Something went wrong while starting."
        raise StartupFailed(advice) from exc


def unreadable_settings(build: Callable[[], object]) -> list[str]:
    """The SAR_* variables that make `build` (Settings) fail, each tried on its own; [] if none is to blame."""
    saved = {key: value for key, value in os.environ.items() if key.startswith("SAR_")}
    for key in saved:
        del os.environ[key]
    try:
        try:
            build()
        except Exception:
            return []  # fails with no settings at all: not a settings problem
        bad: list[str] = []
        for key, value in saved.items():
            os.environ[key] = value
            try:
                build()
            except Exception:
                bad.append(key)
            finally:
                del os.environ[key]
        return sorted(bad)
    finally:
        os.environ.update(saved)


def settings_advice(env_files: Sequence[Path]) -> str:
    from biobench_sar.config import Settings

    where = "sar.env (" + ", ".join(str(p) for p in env_files) + ")" if env_files else "the environment variables"
    bad = unreadable_settings(Settings)
    if not bad:
        return f"The settings could not be read. Check the SAR_* lines in {where}."
    values = ", ".join(f"{key}={os.environ.get(key, '')}" for key in bad)
    if len(bad) == 1:
        lead, it = f"This setting has a value the app can't read: {values}.", "it"
    else:
        lead, it = f"These settings have values the app can't read: {values}.", "them"
    return f"{lead} Fix or delete {it} in {where}. Numbers must be plain numbers, for example SAR_VLM_TIMEOUT=90."


def damaged_install_advice() -> str:
    if is_frozen():
        return (
            "Part of the program could not be loaded, so this installation is probably damaged or incomplete "
            "(an antivirus program may have removed a file). Install the app again."
        )
    return "Part of the program could not be loaded. Install the packages again: pip install -r requirements.txt"


def busy_port_advice(port: int) -> str:
    other = 5051 if port != 5051 else 5052
    where = "its icon in the Dock" if sys.platform == "darwin" else "its other window"
    return (
        f"Port {port} is already in use: the assistant is probably running already (look for {where}), or "
        f"another program uses that port. Close it and start again, or set SAR_PORT={other} in sar.env."
    )


def exit_code(exc: SystemExit) -> int:
    if exc.code is None:
        return 0
    return exc.code if isinstance(exc.code, int) else 1


def _known_log_file() -> Path | None:
    if _log_redirect is not None:
        return _log_redirect
    try:
        return log_file()
    except OSError:
        return None


def save_failure_log(text: str) -> Path | None:
    """Put a failure report where the teacher can find it and send it on; None if no log can be written."""
    if _log_redirect is not None:
        return _log_redirect  # all output already goes there, the report included
    try:
        path = log_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        _keep_previous_log(path)
        path.write_text(text + "\n", encoding="utf-8")
    except OSError:
        return None
    return path


def report_failure(exc: BaseException, out: Out = say) -> int:
    """Print the traceback, then what to do in plain words; save both to the log; show a dialog from Finder.

    Returns the exit code, 1.
    """
    details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip()
    if isinstance(exc, StartupFailed):
        headline, advice = f"{APP_NAME} could not start.", str(exc)
    else:
        headline = f"{APP_NAME} stopped because of an unexpected error."
        advice = (
            "Start it again; if this keeps happening, send the log file below to whoever looks after this computer."
        )
    cause = exc.__cause__ if isinstance(exc, StartupFailed) and exc.__cause__ is not None else exc
    summary = [headline, advice, f"Technical detail: {type(cause).__name__}: {cause}"]
    out(details)
    log = save_failure_log(f"{APP_NAME} {__version__} ({sys.platform})\n\n{details}\n\n" + "\n".join(summary))
    for line in ["", "-" * 72, *summary]:
        out(line)
    out(f"Details are saved in {log}" if log else "The details above could not be saved to a log file.")
    alert("\n\n".join([headline, advice, f"Details are saved in {log}" if log else ""]).strip())
    return 1


def run_guarded(entry: Callable[[], int], out: Out = say) -> int:
    """Run `entry` and return its exit code; any error that escapes it is reported by report_failure."""
    try:
        return entry()
    except SystemExit as exc:
        if isinstance(exc.code, str):
            out(exc.code)
        return exit_code(exc)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        return report_failure(exc, out)


# ---- environment -------------------------------------------------------------------------------------


def env_file_candidates() -> list[Path]:
    """sar.env files in priority order: $SAR_ENV_FILE, next to the app, then the per-user data folder."""
    candidates: list[Path] = []
    override = os.environ.get("SAR_ENV_FILE")
    if override:
        candidates.append(Path(override))
    candidates += [executable_dir() / ENV_FILE_NAME, user_data_dir() / ENV_FILE_NAME]
    unique: list[Path] = []
    for path in candidates:
        if path not in unique:
            unique.append(path)
    return unique


def bundled_models_dir() -> Path:
    return resource_dir() / "models"


def configure_bundled_models(models: Path) -> bool:
    """Point sentence-transformers / Hugging Face (and Whisper, if bundled) at the models shipped in the build.

    Offline only when the embedding model is really there: otherwise the first run must download it.
    """
    found = False
    if models.is_dir() and any(models.glob("models--*")):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(models))
        os.environ.setdefault("HF_HOME", str(models))
        found = True
    # whisper.load_model() takes a checkpoint path as well as a name.
    whisper = models / "whisper" / f"{os.environ.get('SAR_WHISPER_MODEL', 'base')}.pt"
    if whisper.is_file():
        os.environ["SAR_WHISPER_MODEL"] = str(whisper)
        found = True
    return found


def _sync_config_data_dir() -> None:
    # envfile imports config, which read SAR_DATA_DIR at import time; Settings() reads DATA_DIR at build time.
    from biobench_sar import config

    config.DATA_DIR = Path(os.environ["SAR_DATA_DIR"])


def prepare_environment(frozen: bool | None = None) -> list[Path]:
    """Settle what Settings() will read; returns the sar.env files applied.

    Real environment variables beat every sar.env, and a sar.env next to the app beats the one in the data
    folder (load_env_file never overwrites a key that is already set). SAR_DATA_DIR defaults to the per-user
    data folder, so an installed app never writes next to its own files.
    """
    frozen = is_frozen() if frozen is None else frozen
    from biobench_sar.envfile import load_env_file

    loaded: list[Path] = []
    for path in env_file_candidates():
        if path.is_file():
            # Notepad's "Unicode" encoding (UTF-16) is the likely culprit; load_env_file only expects OSError.
            with startup_step(
                f"The settings file {path} could not be read. Open it in Notepad or TextEdit and save it as "
                "plain text (UTF-8), or delete it."
            ):
                load_env_file(path)
            loaded.append(path)
    os.environ.setdefault("SAR_DATA_DIR", str(user_data_dir() / "data"))
    _sync_config_data_dir()
    # chromadb phones home by default; a classroom install shouldn't.
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    if frozen:
        configure_bundled_models(bundled_models_dir())
        _offer_env_template()
    return loaded


def _offer_env_template() -> None:
    """Drop sar.env.example in the data folder once, so teachers have a template to copy to sar.env."""
    template = resource_dir() / "sar.env.example"
    target = user_data_dir() / "sar.env.example"
    if template.is_file() and not target.exists():
        with contextlib.suppress(OSError):
            shutil.copyfile(template, target)


# ---- Ollama ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OllamaStatus:
    reachable: bool
    has_model: bool
    error: str = ""


Fetch = Callable[[str, float], bytes]


def _direct_opener() -> urllib.request.OpenerDirector:
    # A school proxy in HTTP_PROXY must not swallow requests to this machine.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _fetch(url: str, timeout: float) -> bytes:
    with _direct_opener().open(url, timeout=timeout) as response:
        return bytes(response.read())


def check_ollama(
    base_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_VLM_MODEL,
    timeout: float = OLLAMA_TIMEOUT_S,
    fetch: Fetch = _fetch,
) -> OllamaStatus:
    try:
        body = json.loads(fetch(base_url.rstrip("/") + "/api/tags", timeout) or b"{}")
    # Not just OSError: another program answering on Ollama's port can raise http.client.HTTPException, and a
    # failed check must only produce advice, never stop the app.
    except Exception as exc:
        return OllamaStatus(reachable=False, has_model=False, error=f"{type(exc).__name__}: {exc}")
    models = body.get("models", []) if isinstance(body, dict) else []
    names = {str(m.get(key, "")) for m in models if isinstance(m, dict) for key in ("name", "model")}
    wanted = {model, model if ":" in model else f"{model}:latest"}
    return OllamaStatus(reachable=True, has_model=bool(names & wanted))


def ollama_advice(status: OllamaStatus, model: str) -> list[str]:
    """What to tell the teacher; empty when Ollama is ready."""
    if status.reachable and status.has_model:
        return []
    if status.reachable:
        return [
            f"Ollama is running, but the model {model} is not downloaded yet.",
            f"  Open a terminal and run:  ollama pull {model}   (about 3 GB, once)",
            "  Then ask your question again; no restart needed.",
        ]
    return [
        "Ollama is not running (or not installed), so the assistant cannot answer yet.",
        f"  1. Install Ollama: {OLLAMA_DOWNLOAD_URL}",
        f"  2. Download the model (about 3 GB, once):  ollama pull {model}",
        "  3. Make sure Ollama is running (it starts in the background after install), then restart this app.",
        "  The dashboard still opens; answers will say the model is unreachable until Ollama runs.",
    ]


# ---- browser -----------------------------------------------------------------------------------------


def _connect_host(host: str) -> str:
    return "127.0.0.1" if serves_network(host) else host


def dashboard_url(host: str, port: int) -> str:
    host = _connect_host(host)
    if ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{port}/dashboard"


def serves_network(host: str) -> bool:
    return host in ("", "0.0.0.0", "::")


def lan_address() -> str | None:
    """This computer's address on the local network, for phones and tablets on the same Wi-Fi."""
    with contextlib.suppress(OSError), socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect(("10.254.254.254", 1))  # UDP connect sends nothing; it only picks the outgoing interface
        address = str(sock.getsockname()[0])
        if not address.startswith(("127.", "0.")):
            return address
    return None


def port_free(host: str, port: int) -> bool:
    """False when something already listens there; raises for a host or port that can never work."""
    with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM) as sock:
        # On Windows SO_REUSEADDR lets a bind succeed next to a live listener, which would hide a busy port.
        if sys.platform != "win32":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except socket.gaierror:  # SAR_HOST isn't an address (a typo), which "port busy" would misreport
            raise
        except OSError:
            return False
    return True


def wait_for_port(
    host: str,
    port: int,
    timeout: float,
    connect: Callable[..., Any] = socket.create_connection,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> bool:
    deadline = clock() + timeout
    while True:
        try:
            connect((_connect_host(host), port), timeout=1.0).close()
            return True
        except OSError:
            if clock() >= deadline:
                return False
            sleep(0.25)


def open_browser_when_ready(
    url: str,
    host: str,
    port: int,
    timeout: float = BROWSER_WAIT_S,
    opener: Callable[[str], Any] = webbrowser.open,
) -> threading.Thread:
    def run() -> None:
        if wait_for_port(host, port, timeout):
            with contextlib.suppress(Exception):
                opener(url)

    thread = threading.Thread(target=run, daemon=True, name="open-browser")
    thread.start()
    return thread


# ---- macOS: launched from Finder (no terminal) ---------------------------------------------------------


def launched_from_finder() -> bool:
    """A double-clicked .app has launchd as parent and no terminal to print to."""
    return is_frozen() and sys.platform == "darwin" and os.getppid() == 1


def log_file() -> Path:
    return user_data_dir() / "logs" / "sar.log"


_log_redirect: Path | None = None  # set once stdout/stderr go to the log file


def _keep_previous_log(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.replace(path.with_name("sar.previous.log"))


def redirect_output_to_log() -> Path:
    global _log_redirect
    path = log_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    _keep_previous_log(path)
    # Not a with-block: the stream stays open for the life of the process.
    stream = open(path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
    sys.stdout = stream
    sys.stderr = stream
    _log_redirect = path
    return path


def _applescript_string(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def alert(message: str) -> None:
    """Show a dialog when there is no terminal to read (Finder launch on macOS); no-op otherwise."""
    if not launched_from_finder():
        return
    script = f"display alert {_applescript_string(APP_NAME)} message {_applescript_string(message)}"
    with contextlib.suppress(OSError):
        subprocess.Popen(["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---- self-test ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: str


HttpGet = Callable[[str, float], HttpResponse]


def http_get(url: str, timeout: float) -> HttpResponse:
    try:
        with _direct_opener().open(url, timeout=timeout) as response:
            return HttpResponse(response.status, response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        return HttpResponse(exc.code, exc.read().decode("utf-8", "replace"))


@dataclass(frozen=True)
class Check:
    name: str
    run: Callable[[], str]  # returns a detail line; raises on failure


class CheckFailed(Exception):
    pass


def page_title(body: str) -> str:
    # The pages translate their title client-side, so the tag carries attributes (<title data-i18n=...>).
    match = re.search(r"<title\b[^>]*>(.*?)</title>", body, re.IGNORECASE | re.DOTALL)
    return html.unescape(match.group(1)).strip() if match else ""


def http_checks(base_url: str, get: HttpGet = http_get, timeout: float = 15.0, backend: str = "mock") -> list[Check]:
    def fetch(path: str) -> HttpResponse:
        response = get(base_url + path, timeout)
        if response.status != 200:
            raise CheckFailed(f"GET {path} returned {response.status}")
        return response

    def status() -> str:
        data = json.loads(fetch("/api/status").body)
        if data.get("vlm_backend") != backend:
            raise CheckFailed(f"vlm_backend is {data.get('vlm_backend')!r}, expected {backend!r}")
        return f"200, backend {data['vlm_backend']}, profile {data.get('profile')}"

    def page(path: str) -> Callable[[], str]:
        def run() -> str:
            title = page_title(fetch(path).body)
            if PAGE_TITLE not in title:
                raise CheckFailed(f"GET {path} has title {title!r}, expected {PAGE_TITLE!r}")
            return f"200, title {title!r}"

        return run

    def static() -> str:
        size = len(fetch("/static/js/vendor/socket.io.min.js").body)
        if size < 1000:
            raise CheckFailed(f"socket.io client is only {size} bytes")
        return f"200, {size} bytes"

    def socketio() -> str:
        body = fetch("/socket.io/?EIO=4&transport=polling").body
        if not body.startswith("0{") or '"sid"' not in body:
            raise CheckFailed(f"unexpected Engine.IO handshake {body[:80]!r}")
        return "200, Engine.IO handshake"

    return [
        Check("api/status", status),
        Check("dashboard", page("/dashboard")),
        Check("projector page", page("/")),
        Check("static files", static),
        Check("socket.io", socketio),
    ]


@contextlib.contextmanager
def torch_on_cpu() -> Iterator[None]:
    """Hide Apple's MPS GPU while a model loads: hosted CI Macs report one that can't allocate memory.

    The self-test checks what the build bundles, not the GPU; the app itself still uses MPS on real Macs.
    """
    import torch

    mps = torch.backends.mps
    original = mps.is_available
    mps.is_available = lambda: False
    try:
        yield
    finally:
        mps.is_available = original


def utf8_mode_detail(frozen: bool, utf8_mode: int) -> str:
    """sar.spec turns on UTF-8 mode, so open() without an encoding reads sar.env-style files right on Windows."""
    if utf8_mode:
        return "on"
    if frozen:
        raise CheckFailed('UTF-8 mode is off: the build lost the "X utf8" run-time option in packaging/sar.spec')
    return "off (source checkout, not required)"


def bundle_checks(services: Any) -> list[Check]:
    """What a frozen build can lose: lazily imported modules, package data and the bundled embedding model."""

    def imports() -> str:
        import importlib

        missing = []
        for name in RUNTIME_MODULES:
            try:
                importlib.import_module(name)
            except Exception as exc:
                missing.append(f"{name} ({type(exc).__name__}: {exc})")
        if missing:
            raise CheckFailed("cannot import " + "; ".join(missing))
        return f"{len(RUNTIME_MODULES)} modules"

    def embeddings() -> str:
        from biobench_sar.rag.ingestor import Chunk

        store = services.store
        with torch_on_cpu():
            store.add_chunks([Chunk("Acanthamoeba is a free-living amoeba found in lakes.", "self-test.md", {})])
            hits = store.query("free-living amoeba", k=1)
        if not hits or hits[0].source != "self-test.md":
            raise CheckFailed(f"vector store query returned {hits!r}")
        return f"chromadb + sentence-transformers, score {hits[0].score:.2f}"

    def whisper_assets() -> str:
        from whisper.audio import mel_filters
        from whisper.tokenizer import get_tokenizer

        filters = mel_filters("cpu", 80)
        tokens = get_tokenizer(multilingual=True).encode("hola")
        if not tokens:
            raise CheckFailed("whisper tokenizer returned no tokens")
        return f"mel filters {tuple(filters.shape)}, tokenizer ok"

    def utf8() -> str:
        return utf8_mode_detail(is_frozen(), sys.flags.utf8_mode)

    return [
        Check("utf-8 mode", utf8),
        Check("imports", imports),
        Check("embeddings", embeddings),
        Check("whisper assets", whisper_assets),
    ]


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def self_test_settings(data_dir: Path, port: int) -> Any:
    from biobench_sar.config import Settings

    return dataclasses.replace(
        Settings(),
        host="127.0.0.1",
        port=port,
        vlm_backend="mock",
        tts_backend="none",
        listen_mode="push",
        seed_knowledge=False,
        calibration_file=data_dir / "calibration.json",
        chroma_dir=data_dir / "chroma_db",
        upload_dir=data_dir / "uploads",
        tts_dir=data_dir / "tts",
        selection_file=data_dir / "selection.json",
        quiz_bank_file=data_dir / "quiz_bank.json",
    )


def run_checks(checks: Sequence[Check], out: Out) -> list[str]:
    failures: list[str] = []
    for check in checks:
        started = time.monotonic()
        try:
            detail = check.run()
        except Exception as exc:
            failures.append(check.name)
            out(f"  FAIL {check.name}: {type(exc).__name__}: {exc}")
            continue
        out(f"  ok   {check.name}: {detail} ({time.monotonic() - started:.1f} s)")
    return failures


def _self_test(services: Any, bundle: bool, get: HttpGet, out: Out) -> int:
    os.environ.update(SELF_TEST_ENV)
    data_dir = Path(tempfile.mkdtemp(prefix="sar-self-test-"))
    server: Any = None
    try:
        from werkzeug.serving import make_server

        from biobench_sar import app as app_module

        logging.getLogger("werkzeug").setLevel(logging.WARNING)  # each check prints its own status line
        settings = self_test_settings(data_dir, free_port())
        flask_app, _socketio = app_module.create_app(settings, services)
        server = make_server(settings.host, settings.port, flask_app, threaded=True)
        threading.Thread(target=server.serve_forever, daemon=True, name="self-test-server").start()
        out(f"  server on http://{settings.host}:{settings.port} (mock model, no camera, no audio)")
        checks = http_checks(f"http://{settings.host}:{settings.port}", get)
        if bundle:
            checks += bundle_checks(flask_app.config["SAR_SERVICES"])
        failures = run_checks(checks, out)
    except Exception as exc:
        out(traceback.format_exc().rstrip())
        out(f"SELF-TEST FAILED: {type(exc).__name__}: {exc}")
        return 1
    finally:
        if server is not None:
            server.shutdown()
        shutil.rmtree(data_dir, ignore_errors=True)
    if failures:
        out(f"SELF-TEST FAILED: {', '.join(failures)}")
        return 1
    out("SELF-TEST OK")
    return 0


def run_self_test(
    *,
    timeout_s: float = SELF_TEST_TIMEOUT_S,
    services: Any = None,
    bundle: bool = True,
    get: HttpGet = http_get,
    out: Out = say,
) -> int:
    """Start the real app on a free port with a mock model and check it answers; 0 = pass, 1 = fail.

    `services=None` builds the real services (camera and microphone are never opened), which is what a
    frozen build needs exercised; tests pass fakes and `bundle=False`.
    """
    kind = "frozen build" if is_frozen() else "source checkout"
    python = f"Python {sys.version.split()[0]}, UTF-8 mode {'on' if sys.flags.utf8_mode else 'off'}"
    out(f"{APP_NAME} {__version__} self-test ({kind}, {python}, {sys.platform})")
    outcome: list[int] = []
    worker = threading.Thread(
        target=lambda: outcome.append(_self_test(services, bundle, get, out)), daemon=True, name="self-test"
    )
    worker.start()
    worker.join(timeout_s)
    if not outcome:
        out(f"SELF-TEST FAILED: no result after {timeout_s:.0f} s")
        return 1
    return outcome[0]


# ---- run -------------------------------------------------------------------------------------------------


def run_app(open_browser: bool = True, out: Out = say, env_files: Sequence[Path] = ()) -> int:
    """Start the dashboard server and return its exit code; a startup step that fails raises StartupFailed.

    `env_files` are the sar.env files applied, named in the advice when a setting can't be read.
    """
    with startup_step(lambda: settings_advice(env_files)):
        from biobench_sar.config import Settings

        settings = Settings()
    with startup_step(damaged_install_advice):
        from biobench_sar import app as app_module
    url = dashboard_url(settings.host, settings.port)
    out(f"{APP_NAME} {__version__}")
    out(f"  Data folder  {os.environ.get('SAR_DATA_DIR', '')}")
    out(f"  Dashboard    {url}")
    with startup_step(
        f"{settings.host}:{settings.port} is not an address this computer can serve on. Set SAR_HOST to "
        "127.0.0.1 (this computer only) or 0.0.0.0 (phones and tablets too), and SAR_PORT to a number "
        "from 1024 to 65535."
    ):
        free = port_free(settings.host, settings.port)
    # app.main() would stop on a busy port too, with a message meant for developers.
    if not free:
        raise StartupFailed(busy_port_advice(settings.port))
    if serves_network(settings.host):
        address = lan_address()
        if address:
            out(f"  On a phone or tablet (same Wi-Fi): http://{address}:{settings.port}/dashboard")
    if settings.vlm_backend == "ollama":
        advice = ollama_advice(check_ollama(settings.vlm_base_url, settings.vlm_model), settings.vlm_model)
        for line in advice:
            out(line)
        if advice:
            alert("\n".join(advice))
    if open_browser:
        open_browser_when_ready(url, settings.host, settings.port)

    try:
        app_module.main()
    except KeyboardInterrupt:
        return 0
    except SystemExit as exc:
        code = exit_code(exc)
        if code:
            out(f"{APP_NAME} stopped (exit code {code}); the reason is in the messages above.")
            alert(f"{APP_NAME} could not start. Details are in {_known_log_file()}")
        return code
    return 0


def _port_number(text: str) -> int:
    port = int(text)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"{port} is not a port number (1-65535)")
    return port


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=APP_NAME, description=f"{APP_NAME} {__version__}")
    parser.add_argument("--no-browser", action="store_true", help="don't open the dashboard in the browser")
    parser.add_argument("--port", type=_port_number, help="serve on this port (default: SAR_PORT or 5050)")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="start on a free port with a mock model, check the pages and bundled models, then exit",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = list(sys.argv[1:] if argv is None else argv)
    # Older macOS passes a process serial number to apps opened from Finder.
    return parser.parse_args([a for a in args if not a.startswith("-psn_")])


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if launched_from_finder() or sys.stdout is None or sys.stderr is None:
        with contextlib.suppress(OSError):  # no log is no reason not to start; a failure still shows a dialog
            redirect_output_to_log()
    with startup_step(
        "The app could not set up its data folder or read its settings (sar.env). "
        "Check that your user account can create files in its own folders."
    ):
        loaded = prepare_environment()
    if args.self_test:
        return run_self_test()
    for path in loaded:
        say(f"Settings from {path}")
    if args.port is not None:
        os.environ["SAR_PORT"] = str(args.port)
    return run_app(open_browser=not args.no_browser, env_files=loaded)


def _pause_if_console_closes(code: int) -> None:
    # A double-clicked .exe closes its console on exit, taking the error message with it.
    if not (code and is_frozen() and sys.platform == "win32" and sys.stdin is not None):
        return
    with contextlib.suppress(EOFError, OSError, ValueError):  # ValueError: stdin already closed
        if sys.stdin.isatty():
            input("\nPress Enter to close this window...")


def _utf8_output() -> None:
    """Pipes and log files get UTF-8 whatever the locale (Windows would use its ANSI code page)."""
    for stream in (sys.stdout, sys.stderr):
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        with contextlib.suppress(Exception):
            if not stream.isatty():
                stream.reconfigure(encoding="utf-8", errors="replace")


def _run() -> None:
    multiprocessing.freeze_support()
    _utf8_output()
    code = run_guarded(main)
    _pause_if_console_closes(code)
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()
    # Library threads (chromadb, torch, the camera loop) must not keep a finished process alive.
    os._exit(code)


if __name__ == "__main__":
    _run()
