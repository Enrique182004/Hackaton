"""Flask + Socket.IO server wiring vision, RAG, VLM, audio and TTS together."""

from __future__ import annotations

import contextlib
import ipaddress
import logging
import math
import mimetypes
import re
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import flask_socketio
from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    send_from_directory,
)
from flask_socketio import SocketIO, disconnect
from jinja2 import TemplateNotFound
from markupsafe import escape
from werkzeug.utils import secure_filename

from biobench_sar.assistant import LANGUAGE_SETTINGS, Assistant
from biobench_sar.config import Settings, lan_requested
from biobench_sar.lan import (
    COOKIE_NAME,
    LAN_MODE_HEADER,
    JoinPin,
    PhoneSessions,
    SocketIOGate,
    cookie_kwargs,
    is_allowed_host,
    is_local_request,
    is_loopback_bind,
    is_request_allowed,
    join_url,
    local_ip_addresses,
    looks_like_navigation,
)
from biobench_sar.ports import BROWSER_BLOCKED_PORTS
from biobench_sar.tutor import MODES

log = logging.getLogger("biobench_sar.app")
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
CALIBRATION_SETTLE_SECONDS = 1.5
MAX_SEARCH_RESULTS = 10
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
DEFAULT_PORTS = {"http": 80, "https": 443}
NO_TEXT_ERROR = "no readable text found (scanned PDF or blank image?)"
ONLY_ANSWERS_ERROR = "only answers found: they are kept out of the notes, so there is nothing to store"
# Any text works: query() only needs one to rank a file's chunks, and every one of them comes back.
ALL_CHUNKS_PROBE = "answer"
# The projector page sends a handful (avatar, bubble, quiz card, highlight); more is a broken or hostile client.
MAX_UI_RECTS = 32
JOIN_LANGUAGES = ("en", "es")
JOIN_TEXTS = {
    "en": {
        "title": "Join the whiteboard",
        "hint": "Type the PIN shown on the laptop's dashboard.",
        "button": "Join",
        "wrong_pin": "That PIN didn't work. Check the number on the laptop and try again.",
        "locked_minutes": "Too many wrong PINs. Try again in {n} minutes.",
        "locked_minute": "Too many wrong PINs. Try again in 1 minute.",
        "locked_seconds": "Too many wrong PINs. Try again in {n} seconds.",
    },
    "es": {
        "title": "Únete al pizarrón",
        "hint": "Escribe el PIN que aparece en el panel de la computadora.",
        "button": "Entrar",
        "wrong_pin": "Ese PIN no funcionó. Revisa el número en la computadora e inténtalo de nuevo.",
        "locked_minutes": "Demasiados PIN incorrectos. Inténtalo de nuevo en {n} minutos.",
        "locked_minute": "Demasiados PIN incorrectos. Inténtalo de nuevo en 1 minuto.",
        "locked_seconds": "Demasiados PIN incorrectos. Inténtalo de nuevo en {n} segundos.",
    },
}


@dataclass
class Services:
    vision: Any  # VisionLoop
    store: Any  # VectorStore
    vlm: Any  # VLMClient
    audio: Any  # AudioTrigger (on_query/on_state are wired by create_app)
    tts: Any  # TTSEngine


def build_services(settings: Settings, on_query: Any, on_state: Any, should_listen: Any = None) -> Services:
    """Construct real services. Imports are lazy so importing app.py stays light."""
    from biobench_sar.core.audio_trigger import AudioTrigger
    from biobench_sar.core.homography import Calibration
    from biobench_sar.core.vision_loop import VisionLoop
    from biobench_sar.llm.tts_engine import TTSEngine
    from biobench_sar.llm.vlm_client import VLMClient
    from biobench_sar.rag.vector_store import VectorStore

    calibration = Calibration.load(settings.calibration_file, expected_projector_size=settings.projector_size)
    if calibration is None:
        calibration = Calibration.identity((settings.camera_width, settings.camera_height), settings.projector_size)
    vision = VisionLoop(
        camera_index=settings.camera_index,
        width=settings.camera_width,
        height=settings.camera_height,
        calibration=calibration,
    )
    store = VectorStore(settings.chroma_dir, embed_model=settings.embed_model)
    vlm = VLMClient(
        settings.vlm_backend,
        settings.vlm_model,
        settings.vlm_base_url,
        settings.vlm_api_key,
        settings.vlm_timeout,
        profile=settings.profile,
    )
    audio = AudioTrigger(
        on_query=on_query,
        whisper_model=settings.whisper_model,
        wake_word=settings.wake_word,
        mode=settings.listen_mode,
        record_seconds=settings.record_seconds,
        on_state=on_state,
        should_listen=should_listen,
        silence_rms=settings.silence_rms,
        language=settings.language,
    )
    tts = TTSEngine(settings.tts_backend, settings.tts_voice, settings.tts_dir)
    return Services(vision=vision, store=store, vlm=vlm, audio=audio, tts=tts)


def create_app(
    settings: Settings | None = None,
    services: Services | None = None,
    template_folder: str | None = None,
) -> tuple[Flask, SocketIO]:
    settings = settings or Settings()
    for directory in (settings.upload_dir, settings.tts_dir):
        directory.mkdir(parents=True, exist_ok=True)

    app = Flask(__name__, template_folder=template_folder or "templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    # None = same-origin only: other pages open in the demo browser can't drive the avatar.
    # always_connect sends CONNECT before on_connect runs: socket.io-client replays events that arrive
    # before CONNECT ahead of its "connect" event, whose idle reset would then bury on_connect's status.
    socketio = SocketIO(app, async_mode="threading", cors_allowed_origins=None, always_connect=True)
    # LAN mode (docs/MOBILE.md): phones on the Wi-Fi join with the PIN; the laptop itself (loopback) never needs it.
    lan_on = settings.lan
    lan_names = settings.lan_hostnames
    gate = JoinPin()
    phones = PhoneSessions()
    mimetypes.add_type("application/manifest+json", ".webmanifest")  # Windows' MIME table may not have it
    if lan_on:
        # Wraps SocketIO's own middleware, so devices without a token are refused before python-socketio
        # opens a session, and no broadcast ever reaches them.
        app.wsgi_app = SocketIOGate(app.wsgi_app, gate, extra_names=lan_names)  # type: ignore[method-assign]

    assistant = Assistant(settings, socketio.emit, socketio.start_background_task)
    svc = services or build_services(settings, assistant.handle_query, assistant.on_state, assistant.should_listen)
    assistant.services = svc
    assistant.push_speech_rate()  # the voice starts at the room's saved speed
    if services is None:
        assistant.calibrated = _load_initial_calibration(settings)
    _prune_quiz_bank(assistant, svc.store)
    strip_answers = settings.profile == "math"
    # Files stored before answers were stripped at upload, that could not be re-ingested: they need a re-upload.
    flagged: set[str] = set()
    flagged_lock = threading.Lock()

    def flag_answers(names: list[str]) -> None:
        with flagged_lock:
            flagged.update(names)

    def unflag_answers(name: str) -> None:
        with flagged_lock:
            flagged.discard(name)

    def documents_body() -> dict[str, Any]:
        documents = svc.store.list_sources()
        active = assistant.active_sources
        with flagged_lock:
            stale = set(flagged)
        listed = [
            {
                **d,
                "active": active is None or d["source"] in active,
                **({"has_answers": True} if d["source"] in stale else {}),
            }
            for d in documents
        ]
        return {
            "documents": listed,
            "total_chunks": sum(d["chunks"] for d in documents),
            "active_count": sum(1 for d in listed if d["active"]),
            "all_active": active is None,
        }

    def notify_documents_changed(body: dict[str, Any] | None = None) -> None:
        """Push the document list to every open dashboard, so the student's file picker stays in sync."""
        # The change already succeeded; a failed push must not turn it into a 500.
        try:
            socketio.emit("documents_changed", body if body is not None else documents_body())
        except Exception:
            log.exception("could not push documents_changed")

    @app.before_request
    def block_cross_origin() -> Any:
        # Plain form POSTs skip CORS preflight, so any page open in the operator's browser could upload
        # documents or fire the mic/calibration. Clients sending neither header (curl, scripts) pass.
        if request.method in SAFE_METHODS:
            return None
        claimed = request.headers.get("Origin") or request.headers.get("Referer")
        if not claimed:
            return None
        origin = _origin_of(claimed)
        if origin is None or origin != _origin_of(request.host_url):
            return jsonify({"error": "cross-origin request blocked"}), 403
        return None

    @app.before_request
    def lan_gate() -> Any:
        if not lan_on:
            return None
        # Before /join too, so a DNS-rebound name can't loop through it; the laptop's loopback trust needs this.
        if not is_allowed_host(request.host, lan_names):
            return jsonify({"error": "unknown host"}), 400
        if is_request_allowed(
            gate,
            remote_addr=request.remote_addr,  # the TCP peer: never X-Forwarded-For, there is no proxy
            host=request.host,
            path=request.path,
            token=request.cookies.get(COOKIE_NAME),
            extra_names=lan_names,
        ):
            return None
        if request.path.startswith("/api/") or request.method not in ("GET", "HEAD"):
            return jsonify({"error": "join with the PIN first"}), 401
        return redirect("/join")

    @app.after_request
    def lan_headers(response: Response) -> Response:
        # Live data must never be stale, and GET /api/lan carries the PIN: nothing under /api/ may be cached.
        if request.path.startswith("/api/") or request.path == "/join":
            response.headers["Cache-Control"] = "no-store"
        if request.path == "/join":
            # Not no-referrer: browsers then send "Origin: null" with the form POST, which block_cross_origin
            # refuses. same-origin keeps the PIN in ?pin= out of every other site's logs all the same.
            response.headers["Referrer-Policy"] = "same-origin"
        if lan_on:
            response.headers[LAN_MODE_HEADER] = "1"  # the service worker retires itself without it
        return response

    def socket_allowed() -> bool:
        """The gate again inside Socket.IO handlers: flask.request is the handshake, so a revoked token fails here."""
        return not lan_on or is_request_allowed(
            gate,
            remote_addr=request.remote_addr,
            host=request.host,
            path="/socket.io/",
            token=request.cookies.get(COOKIE_NAME),
            extra_names=lan_names,
        )

    # ---- pages -------------------------------------------------------------
    def page(name: str) -> str:
        return render_template(
            name,
            projector_width=settings.projector_width,
            projector_height=settings.projector_height,
            lan_enabled=lan_on,
            # The projector's first paint already has the room's text size, captions, calm mode and contrast.
            access=assistant.accessibility_body(),
        )

    @app.get("/")
    def index() -> str:
        return page("index.html")

    @app.get("/precharge")
    @app.get("/dashboard")
    def precharge() -> str:
        return page("precharge.html")

    def join_language() -> str:
        # A phone can't read /api/language before joining, so the page picks its language here.
        setting = assistant.language_setting
        if setting in JOIN_LANGUAGES:
            return setting
        return request.accept_languages.best_match(JOIN_LANGUAGES) or assistant.current_language

    def join_page(*, failed: bool) -> Any:
        language = join_language()
        if language not in JOIN_TEXTS:
            language = JOIN_LANGUAGES[0]
        locked_for = math.ceil(gate.locked_for(request.remote_addr))
        lockout = _lockout_text(locked_for, language)
        message = lockout or (JOIN_TEXTS[language]["wrong_pin"] if failed else None)
        variables = {
            "error": failed or lockout is not None,
            "language": language,
            "lan_enabled": lan_on,
            "locked_for": locked_for,
            "lockout": lockout,
            "message": message,
        }
        try:
            body = render_template("join.html", **variables)
        except TemplateNotFound:
            body = _fallback_join_html(language, message)
        return make_response(body)

    @app.route("/join", methods=["GET", "POST"])
    def join() -> Any:
        # The laptop never needs the PIN, and outside LAN mode there is nothing to join.
        if not lan_on or is_local_request(request.remote_addr):
            return redirect("/dashboard", code=303)
        if request.method == "POST":
            pin: str | None = request.form.get("pin") or ""
        else:
            pin = request.args.get("pin")
            # The QR link is a GET: count it only when it looks like a page load, so an <img> tag on some other
            # site can't burn this device's guesses.
            if pin is not None and not looks_like_navigation(
                accept=request.headers.get("Accept"),
                sec_fetch_mode=request.headers.get("Sec-Fetch-Mode"),
                sec_fetch_site=request.headers.get("Sec-Fetch-Site"),
            ):
                pin = None
            if pin is None and gate.is_valid_token(request.cookies.get(COOKIE_NAME)):
                return redirect("/dashboard", code=303)
        if pin is None:
            return join_page(failed=False)
        token = gate.redeem(pin, request.remote_addr) if pin.strip() else None
        if token is None:
            return join_page(failed=True)
        # 303 takes the PIN out of the address bar and the history.
        response = redirect("/dashboard", code=303)
        response.set_cookie(COOKIE_NAME, token, **cookie_kwargs())
        return response

    @app.get("/sw.js")
    def service_worker() -> Any:
        # Outside LAN mode nothing new may register: a worker would serve stale JS and CSS during development.
        if not lan_on:
            abort(404)
        response = send_from_directory(str(app.static_folder), "sw.js", mimetype="text/javascript", max_age=0)
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Service-Worker-Allowed"] = "/"
        return response

    # ---- API ---------------------------------------------------------------
    @app.post("/api/upload")
    def upload() -> Any:
        from biobench_sar.rag.ingestor import SUPPORTED_EXTENSIONS, ingest_file
        from biobench_sar.rag.quiz_bank import parse_quiz_file

        results: list[dict[str, Any]] = []
        total = 0
        stored: list[str] = []
        emptied: list[str] = []
        for storage in request.files.getlist("files[]") + request.files.getlist("files"):
            name = secure_filename(storage.filename or "")
            if not name:
                results.append({"name": storage.filename or "", "chunks": 0, "error": "invalid filename"})
                continue
            suffix = Path(name).suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                results.append({"name": name, "chunks": 0, "error": f"unsupported file type {suffix or '(none)'}"})
                continue
            path = settings.upload_dir / f"{uuid.uuid4().hex}{suffix}"
            try:
                storage.save(path)
                # Math: answers come out of the whole file before it is chunked, so the store never holds one.
                ingested = ingest_file(path, strip_answers=strip_answers)
                chunks = [replace(c, source=name, metadata={**c.metadata, "source": name}) for c in ingested]
                if not any(c.text.strip() for c in chunks):
                    # Keep whatever an earlier upload of this name stored: a blank rescan must not wipe it.
                    answers_only = strip_answers and any(c.text.strip() for c in ingest_file(path))
                    results.append(
                        {"name": name, "chunks": 0, "error": ONLY_ANSWERS_ERROR if answers_only else NO_TEXT_ERROR}
                    )
                    continue
                # Replace, not add: ids are content hashes, so an edited file would keep its old chunks.
                added, removed = svc.store.replace_source(name, chunks)
                unflag_answers(name)
                total += added
                count = svc.store.source_count(name)
                (stored if count else emptied).append(name)
                results.append({"name": name, "chunks": count, "replaced": removed})
                # Parsed from the saved copy, which the finally below deletes.
                assistant.quiz_bank.replace_source(name, parse_quiz_file(path, name) if count else [])
            except Exception as exc:  # UnsupportedFileType, parse errors, embed errors
                log.warning("upload %s failed: %s", name, exc)
                results.append({"name": name, "chunks": 0, "error": str(exc) or type(exc).__name__})
            finally:
                path.unlink(missing_ok=True)
        assistant.selection.activate(stored)
        for name in emptied:
            assistant.selection.discard(name)  # nothing left in the store under that name
        notify_documents_changed()
        return jsonify({"files": results, "total_chunks": total})

    # ---- LAN (phones): the laptop's own card; a joined phone never sees the PIN -----------------
    def lan_body() -> dict[str, Any]:
        if not lan_on:
            return {"enabled": False, "urls": [], "pin": None, "phones": 0, "locked_for": 0}
        pin = gate.pin
        return {
            "enabled": True,
            "urls": [join_url(address, settings.port, pin) for address in phone_addresses(settings.host)],
            "pin": pin,
            "phones": len(phones),
            "locked_for": math.ceil(gate.server_locked_for()),
        }

    def only_the_laptop() -> Any:
        if is_local_request(request.remote_addr):
            return None
        return jsonify({"error": "only on the laptop itself"}), 403

    def drop_phones() -> None:
        for sid in phones.take_all():
            try:
                socketio.server.disconnect(sid, namespace="/")
            except Exception:
                log.exception("could not disconnect phone %s", sid)

    @app.get("/api/lan")
    def api_lan() -> Any:
        return only_the_laptop() or jsonify(lan_body())

    @app.post("/api/lan/rotate")
    def api_lan_rotate() -> Any:
        refused = only_the_laptop()
        if refused is not None:
            return refused
        if not lan_on:
            return jsonify({**lan_body(), "error": "LAN mode is off"}), 409
        gate.rotate(revoke_tokens=True)
        drop_phones()
        return jsonify(lan_body())

    @app.post("/api/lan/disconnect")
    def api_lan_disconnect() -> Any:
        refused = only_the_laptop()
        if refused is not None:
            return refused
        if not lan_on:
            return jsonify({**lan_body(), "error": "LAN mode is off"}), 409
        gate.revoke_all()
        drop_phones()
        return jsonify(lan_body())

    @app.get("/api/status")
    def status() -> Any:
        try:
            doc_chunks = svc.store.count()
        except Exception:
            doc_chunks = 0
        try:
            documents = documents_body()
            active_documents, total_documents = documents["active_count"], len(documents["documents"])
        except Exception:
            active_documents = total_documents = 0
        return jsonify(
            {
                "fps": round(float(getattr(svc.vision, "fps", 0.0)), 1),
                "camera_running": bool(getattr(svc.vision, "is_running", False))
                and svc.vision.latest_frame() is not None,
                "calibrated": assistant.calibrated,
                "doc_chunks": doc_chunks,
                "active_documents": active_documents,
                "total_documents": total_documents,
                "vlm_backend": settings.vlm_backend,
                "vlm_model": settings.vlm_model,
                "profile": settings.profile,
                "mode": assistant.tutor.mode,
                "audio_available": bool(getattr(svc.audio, "available", False)),
                "tts_backend": settings.tts_backend,
                "listen_mode": settings.listen_mode,
                "state": assistant.state,
                "busy": assistant.busy,
                "last_latency_s": assistant.last_latency_s,
                "uptime_s": round(time.monotonic() - assistant.started_at),
                "language": assistant.language_setting,
                "current_language": assistant.current_language,
                "glare": glare_status(),
                "host_os": host_os(),
            }
        )

    def glare_status() -> dict[str, Any] | None:
        # Measured at most every few seconds on the raw camera frame; a broken camera must not break the status.
        try:
            return assistant.glare.status(svc.vision.latest_frame)
        except Exception:
            log.exception("glare status failed")
            return None

    @app.get("/api/config")
    def api_config() -> Any:
        return jsonify(
            {
                "profile": settings.profile,
                "camera_index": settings.camera_index,
                "camera_size": [settings.camera_width, settings.camera_height],
                "projector_size": list(settings.projector_size),
                "checkerboard": [settings.checker_cols, settings.checker_rows],
                "vlm_backend": settings.vlm_backend,
                "vlm_model": settings.vlm_model,
                "vlm_base_url": settings.vlm_base_url,
                "embed_model": settings.embed_model,
                "rag_top_k": settings.rag_top_k,
                "whisper_model": settings.whisper_model,
                "listen_mode": settings.listen_mode,
                "wake_word": settings.wake_word,
                "tts_backend": settings.tts_backend,
                "tts_voice": settings.tts_voice,
                "language": assistant.language_setting,
                "current_language": assistant.current_language,
            }
        )

    @app.get("/api/language")
    def api_language() -> Any:
        return jsonify(assistant.language_body())

    @app.put("/api/language")
    def api_set_language() -> Any:
        body = request.get_json(silent=True)
        setting = body.get("language") if isinstance(body, dict) else None
        if setting not in LANGUAGE_SETTINGS:
            return jsonify({"error": f"language must be one of: {', '.join(LANGUAGE_SETTINGS)}"}), 400
        return jsonify(assistant.set_language(setting))

    # Joined phones may change these too (no only_the_laptop): the teacher walks the room with one.
    @app.get("/api/accessibility")
    def api_accessibility() -> Any:
        return jsonify(assistant.accessibility_body())

    @app.put("/api/accessibility")
    def api_set_accessibility() -> Any:
        body = request.get_json(silent=True)
        try:
            saved = assistant.set_accessibility(body)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(saved)

    @app.get("/api/history")
    def api_history() -> Any:
        return jsonify({"interactions": assistant.history()})

    @app.get("/api/documents")
    def api_documents() -> Any:
        return jsonify(documents_body())

    @app.put("/api/documents/selection")
    def api_select_documents() -> Any:
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"error": 'expected {"all": true} or {"sources": [...]}'}), 400
        if body.get("all") is True:
            assistant.selection.select_all()
        elif "sources" in body:
            requested = body["sources"]
            if not isinstance(requested, list):
                return jsonify({"error": "sources must be a list of file names"}), 400
            known = {d["source"] for d in svc.store.list_sources()}
            assistant.selection.select(s for s in requested if isinstance(s, str) and s in known)
        else:
            return jsonify({"error": 'expected {"all": true} or {"sources": [...]}'}), 400
        documents = documents_body()
        notify_documents_changed(documents)
        return jsonify(documents)

    @app.delete("/api/documents/<path:source>")
    def api_delete_document(source: str) -> Any:
        deleted = svc.store.delete_source(source)
        assistant.selection.discard(source)
        assistant.quiz_bank.remove_source(source)
        unflag_answers(source)
        if not deleted:
            return jsonify({"deleted": 0, "error": "no such document"}), 404
        notify_documents_changed()
        return jsonify({"deleted": deleted})

    @app.get("/api/tutor")
    def api_tutor() -> Any:
        return jsonify(assistant.tutor.to_dict())

    @app.put("/api/tutor/mode")
    def api_tutor_mode() -> Any:
        body = request.get_json(silent=True)
        mode = body.get("mode") if isinstance(body, dict) else None
        if mode not in MODES:
            return jsonify({"error": f"mode must be one of: {', '.join(MODES)}"}), 400
        assistant.tutor.set_mode(mode)
        assistant.push_tutor()
        return jsonify(assistant.tutor.to_dict())

    @app.post("/api/tutor/quiz/start")
    def api_quiz_start() -> Any:
        assistant.start_quiz()
        return jsonify(assistant.tutor.to_dict())

    @app.post("/api/tutor/quiz/stop")
    def api_quiz_stop() -> Any:
        assistant.tutor.stop_quiz()
        assistant.push_tutor()
        return jsonify(assistant.tutor.to_dict())

    @app.get("/api/search")
    def api_search() -> Any:
        text = (request.args.get("q") or "").strip()
        try:
            k = max(1, min(int(request.args.get("k", settings.rag_top_k)), MAX_SEARCH_RESULTS))
        except ValueError:
            k = settings.rag_top_k
        sources = assistant.selection.query_sources()
        if not text or sources == []:
            return jsonify({"results": []})
        # Same files the assistant would read, so "Search your files" shows what an answer can use.
        results = svc.store.query(text, k, sources=sources)
        return jsonify({"results": [{"text": c.text, "source": c.source, "score": round(c.score, 3)} for c in results]})

    @app.get("/api/ink")
    def api_ink() -> Any:
        from biobench_sar.core.vision_loop import find_ink_regions

        frame = svc.vision.latest_warped()
        if frame is None:
            return jsonify({"regions": [], "size": list(settings.projector_size), "error": "no frame"}), 503
        height, width = frame.shape[:2]
        return jsonify({"regions": find_ink_regions(frame, **settings.ink_kwargs), "size": [width, height]})

    @app.post("/api/query")
    def api_query() -> Any:
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            body = {}
        text = str(body.get("text") or "").strip()
        if not text:
            return jsonify({"accepted": False, "error": "empty text"}), 400
        if assistant.busy:
            return jsonify({"accepted": False, "error": "busy"}), 409
        assistant.submit_text(text)
        return jsonify({"accepted": True}), 202

    @app.post("/api/listen")
    def api_listen() -> Any:
        if not getattr(svc.audio, "available", False):
            return jsonify({"accepted": False, "error": "no microphone available"}), 503
        # trigger() returns False when a recording is already running (older versions return None).
        if svc.audio.trigger() is False:
            return jsonify({"accepted": False, "error": "already recording"}), 409
        return jsonify({"accepted": True}), 202

    calibrating = threading.Lock()

    @app.post("/api/calibrate")
    def api_calibrate() -> Any:
        if svc.vision.latest_frame() is None:
            return jsonify({"ok": False, "error": "no camera frame available"}), 503
        if not calibrating.acquire(blocking=False):
            return jsonify({"ok": False, "error": "calibration already running"}), 409
        try:
            return _calibrate()
        finally:
            calibrating.release()

    def _calibrate() -> Any:
        from biobench_sar.core.homography import calibrate_from_frame

        socketio.emit("show_calibration", {"url": "/calibration.png"})
        try:
            time.sleep(CALIBRATION_SETTLE_SECONDS)
            frame = svc.vision.latest_frame()
            if frame is None:
                return jsonify({"ok": False, "error": "no camera frame available"})
            calibration = calibrate_from_frame(
                frame, settings.projector_size, settings.checker_cols, settings.checker_rows
            )
        except Exception as exc:
            log.exception("calibration failed")
            return jsonify({"ok": False, "error": str(exc)})
        finally:
            socketio.emit("hide_calibration", {})
        if calibration is None:
            return jsonify({"ok": False, "error": "checkerboard not detected; check camera view and lighting"})
        svc.vision.set_calibration(calibration)
        assistant.calibrated = True
        try:
            settings.calibration_file.parent.mkdir(parents=True, exist_ok=True)
            calibration.save(settings.calibration_file)
        except Exception as exc:
            log.warning("could not save calibration: %s", exc)
            return jsonify({"ok": True, "error": f"calibrated but not saved: {exc}"})
        return jsonify({"ok": True})

    @app.get("/calibration.png")
    def calibration_png() -> Any:
        import cv2

        from biobench_sar.core.homography import generate_checkerboard

        board = generate_checkerboard(
            settings.projector_width, settings.projector_height, settings.checker_cols, settings.checker_rows
        )
        ok, buf = cv2.imencode(".png", board)
        if not ok:
            return jsonify({"error": "encode failed"}), 500
        return app.response_class(buf.tobytes(), mimetype="image/png")

    @app.get("/api/frame.jpg")
    def frame_jpg() -> Any:
        from biobench_sar.core.vision_loop import encode_jpeg

        warped = request.args.get("warped", "0") not in ("0", "", "false")
        frame = svc.vision.latest_warped() if warped else svc.vision.latest_frame()
        if frame is None:
            return jsonify({"error": "no frame"}), 503
        response = app.response_class(encode_jpeg(frame), mimetype="image/jpeg")
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/tts/<path:name>")
    def tts_file(name: str) -> Any:
        return send_from_directory(settings.tts_dir, name)

    # ---- sockets -----------------------------------------------------------
    # SocketIOGate refuses devices without a token first; every handler checks again (defence in depth).
    @socketio.on("connect")
    def on_connect(_auth: Any = None) -> None:
        if not socket_allowed():
            raise flask_socketio.ConnectionRefusedError("join with the PIN first")
        if lan_on and not is_local_request(request.remote_addr):
            phones.add(request.sid)
        # A client that (re)connects mid-query must not show a stale state.
        socketio.emit("status", {"state": assistant.state}, to=request.sid)
        socketio.emit("tutor", assistant.tutor.to_dict(), to=request.sid)
        socketio.emit("language", assistant.language_body(), to=request.sid)
        socketio.emit("accessibility", assistant.accessibility_body(), to=request.sid)

    @socketio.on("disconnect")
    def on_disconnect(*_args: Any) -> None:
        phones.discard(request.sid)

    @socketio.on("text_query")
    def on_text_query(data: Any) -> None:
        if not socket_allowed():
            disconnect()
            return
        text = str((data or {}).get("text") or "") if isinstance(data, dict) else ""
        if text.strip():
            assistant.submit_text(text)

    @socketio.on("trigger_listen")
    def on_trigger_listen(_data: Any = None) -> None:
        if not socket_allowed():
            disconnect()
            return
        svc.audio.trigger()

    @socketio.on("ui_rects")
    def on_ui_rects(data: Any = None) -> None:
        # The projector page's own UI (avatar, bubble, quiz card, highlight) in projector px, so the camera's
        # view of the projection is not read as ink. The assistant keeps them with their time and ages them out.
        if not socket_allowed():
            disconnect()
            return
        rects = parse_ui_rects(data)
        setter = getattr(assistant, "set_ui_rects", None)
        if rects is None or not callable(setter):
            return
        try:
            setter(rects)
        except Exception:
            log.exception("set_ui_rects failed")

    app.config["SAR_SERVICES"] = svc
    app.config["SAR_ASSISTANT"] = assistant
    app.config["SAR_HANDLE_QUERY"] = assistant.handle_query
    app.config["SAR_ON_STATE"] = assistant.on_state
    app.config["SAR_SHOULD_LISTEN"] = assistant.should_listen
    app.config["SAR_NOTIFY_DOCUMENTS"] = notify_documents_changed
    app.config["SAR_FLAG_ANSWERS"] = flag_answers
    app.config["SAR_JOIN_PIN"] = gate if lan_on else None
    app.config["SAR_PHONES"] = phones
    return app, socketio


def host_os() -> str:
    """Which camera-permission steps the dashboard shows: "mac", "windows" or "other"."""
    if sys.platform == "darwin":
        return "mac"
    return "windows" if sys.platform.startswith("win") else "other"


def parse_ui_rects(data: Any) -> list[list[float]] | None:
    """[[x, y, w, h], ...] from a "ui_rects" payload ({"rects": [...]}); None when it isn't one.

    Rects that aren't four finite numbers with a positive size are dropped; [] clears the mask.
    """
    raw = data.get("rects") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return None
    rects: list[list[float]] = []
    for item in raw[:MAX_UI_RECTS]:
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            continue
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in item):
            continue
        rect = [float(v) for v in item]
        if all(math.isfinite(v) for v in rect) and rect[2] > 0 and rect[3] > 0:
            rects.append(rect)
    return rects


def phone_addresses(host: str) -> list[str]:
    """Addresses phones can reach this server at: none for a loopback bind, the address itself for one IP."""
    if is_loopback_bind(host):
        return []
    try:
        ip = ipaddress.ip_address(host.strip().strip("[]"))
    except ValueError:  # "" (every interface) or a host name
        return local_ip_addresses()
    return local_ip_addresses() if ip.is_unspecified else [str(ip)]


def laptop_url(host: str, port: int) -> str:
    """The address to click on the laptop: 0.0.0.0 and :: aren't addresses a browser can open."""
    text = host.strip().strip("[]")
    if text in ("", "0.0.0.0"):
        text = "127.0.0.1"
    elif text == "::":
        text = "::1"
    if ":" in text:
        text = f"[{text}]"
    return f"http://{text}:{port}"


def _lockout_text(seconds: int, language: str) -> str | None:
    if seconds <= 0:
        return None
    texts = JOIN_TEXTS[language]
    if seconds < 60:
        return texts["locked_seconds"].format(n=seconds)
    minutes = math.ceil(seconds / 60)
    return texts["locked_minute"] if minutes == 1 else texts["locked_minutes"].format(n=minutes)


def _fallback_join_html(language: str, message: str | None) -> str:
    """A bare PIN form for when templates/join.html is missing, so joining still works."""
    texts = JOIN_TEXTS[language]
    notice = f'<p role="alert">{escape(message)}</p>' if message else ""
    return (
        f'<!doctype html><html lang="{language}"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(texts['title'])}</title></head>"
        '<body style="font-family:system-ui,sans-serif;max-width:22rem;margin:3rem auto;padding:0 1rem">'
        f"<h1>{escape(texts['title'])}</h1><p>{escape(texts['hint'])}</p>{notice}"
        '<form method="post" action="/join"><label for="pin">PIN</label> '
        '<input id="pin" name="pin" inputmode="numeric" autocomplete="one-time-code" maxlength="9" required autofocus> '
        f'<button type="submit">{escape(texts["button"])}</button></form></body></html>'
    )


def startup_banner(settings: Settings, pin: str | None, addresses: list[str]) -> str:
    """The block main() logs once the server is ready: where to open the pages, and in LAN mode how phones join."""
    base = laptop_url(settings.host, settings.port)
    lines = [
        "",
        "",
        "  SAR Whiteboard Assistant",
        f"  Projector  {base}/          (fullscreen on the projector, press F)",
        f"  Dashboard  {base}/dashboard",
        f"  Subject    {settings.profile}",
        f"  Model      {settings.vlm_backend} · {settings.vlm_model}",
    ]
    if settings.lan and pin is not None:
        grouped = f"{pin[:3]} {pin[3:]}" if len(pin) == 6 else pin
        if addresses:
            lines += [
                f"  Phones     {join_url(a, settings.port, pin).split('?', 1)[0]}   PIN {grouped}" for a in addresses
            ]
        elif is_loopback_bind(settings.host):
            lines.append(f"  Phones     can't reach {settings.host}: unset SAR_HOST to listen on the Wi-Fi")
        else:
            lines.append("  Phones     this laptop isn't on a network: join the classroom Wi-Fi")
        lines += [
            "             Allow incoming connections if the firewall asks (Windows: Private networks).",
            "             Disconnect any VPN if phones can't connect.",
        ]
    return "\n".join([*lines, ""])


def _origin_of(url: str) -> tuple[str, str, int | None] | None:
    """(scheme, host, port) with the scheme's default port filled in; None if the URL can't be parsed."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    return scheme, (parts.hostname or "").lower(), port or DEFAULT_PORTS.get(scheme)


def _load_initial_calibration(settings: Settings) -> bool:
    """True if a saved calibration exists (build_services already applied it, else identity)."""
    try:
        from biobench_sar.core.homography import Calibration

        return Calibration.load(settings.calibration_file, expected_projector_size=settings.projector_size) is not None
    except Exception:
        log.exception("could not load calibration")
        return False


def _prune_quiz_bank(assistant: Assistant, store: Any) -> None:
    """Drop quiz problems of files the store no longer has (e.g. its folder was deleted to reset it)."""
    try:
        known = {d["source"] for d in store.list_sources()}
        for source in assistant.quiz_bank.sources():
            if source not in known:
                assistant.quiz_bank.remove_source(source)
    except Exception:
        log.exception("could not prune the quiz bank")


def _seeded_profiles(marker: Path) -> set[str]:
    try:
        return set(marker.read_text(encoding="utf-8").split())
    except OSError:
        return set()


def seed_knowledge(
    store: Any,
    directory: Path,
    *,
    marker: Path | None = None,
    quiz_bank: Any = None,
    on_done: Callable[[list[str]], None] | None = None,
    strip_answers: bool = False,
) -> int:
    """Ingest a profile's bundled documents (directory = knowledge/<profile>) so a fresh install can answer.

    With a marker file each profile is seeded once: a student who removed every file must not get the demo
    documents back on the next start. The marker is only written once every file went in, so a file that failed
    is tried again on the next start. Without a marker, only an empty store is seeded. A seed file is skipped
    when the store already has a file of that name (the student's own quiz_bank.csv is never merged into or
    overwritten). `on_done` gets the seeded file names. strip_answers (math): the store gets each file without its
    answers; the quiz bank still parses the original file, answers included.
    """
    from biobench_sar.rag.ingestor import SUPPORTED_EXTENSIONS, ingest_file
    from biobench_sar.rag.quiz_bank import parse_quiz_file

    if not directory.is_dir():
        return 0
    if marker is not None:
        if directory.name in _seeded_profiles(marker):
            return 0
    elif store.count() > 0:
        return 0
    try:
        present = {d["source"] for d in store.list_sources()}
    except Exception:
        log.exception("could not list the stored files; seeding every file")
        present = set()
    added = failed = skipped = 0
    seeded: list[str] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if path.name in present:
            log.info("not seeding %s: a file of that name is already stored", path.name)
            skipped += 1
            continue
        try:
            count = store.add_chunks(ingest_file(path, strip_answers=strip_answers))
        except Exception:
            log.exception("could not seed %s", path.name)
            failed += 1
            continue
        added += count
        if count:
            seeded.append(path.name)
            if quiz_bank is not None:
                quiz_bank.replace_source(path.name, parse_quiz_file(path, path.name))
    log.info("seeded knowledge base with %d chunks from %s", added, directory)
    if marker is not None and not failed and (added or skipped):
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("\n".join(sorted({*_seeded_profiles(marker), directory.name})) + "\n", encoding="utf-8")
        except OSError as exc:
            log.warning("could not write seed marker %s: %s", marker, exc)
    if seeded and on_done is not None:
        on_done(seeded)
    return added


def seed_app_knowledge(app: Flask, settings: Settings, directory: Path | None = None) -> int:
    """First-run seeding for a running app: new files join an explicit selection and open dashboards refresh."""
    assistant: Assistant = app.config["SAR_ASSISTANT"]
    notify: Callable[[], None] = app.config["SAR_NOTIFY_DOCUMENTS"]

    def done(names: list[str]) -> None:
        assistant.selection.activate(names)
        notify()

    return seed_knowledge(
        app.config["SAR_SERVICES"].store,
        directory or settings.knowledge_dir,
        marker=settings.seed_marker_file,
        quiz_bank=assistant.quiz_bank,
        on_done=done,
        strip_answers=settings.profile == "math",
    )


# A chunk's lines, and a CSV chunk's "Row N: ..." rows (the ingestor joins those on one line).
_SEGMENT = re.compile(r"\n|(?=\bRow \d+: )")


def _segments(texts: Iterable[str]) -> set[str]:
    return {part.strip() for text in texts for part in _SEGMENT.split(text) if part.strip()}


def _is_seed_copy(texts: list[str], seed: Path) -> bool:
    """Whether stored chunks are the bundled seed file as an older version stored it (most of their lines are lines
    of the seed file ingested with its answers), not the student's own file under the same name."""
    from biobench_sar.rag.ingestor import ingest_file

    stored = _segments(texts)
    return bool(stored) and 2 * len(stored & _segments(c.text for c in ingest_file(seed))) >= len(stored)


def _stored_texts(store: Any, source: str, count: int) -> list[str]:
    """Every chunk text the store holds for `source` (`count` of them), through the store's public query."""
    if count <= 0:
        return []
    return [chunk.text for chunk in store.query(ALL_CHUNKS_PROBE, count, sources=[source])]


def migrate_answer_keys(store: Any, directory: Path | None, *, quiz_bank: Any = None) -> tuple[list[str], list[str]]:
    """For a math store filled before answers were stripped at upload: (re-ingested, flagged) file names.

    A file whose stored chunks still hold answers (an "Answer:" line, an answer key, a CSV answer column) is
    re-ingested without them when it is a bundled seed file as an older version stored it; its quiz problems are
    re-read from that file. Any other such file (an upload, whose
    original is gone, or the student's own file under a seed file's name) is flagged for a re-upload; the
    per-chunk stripping at prompt time still covers it meanwhile. Never raises for one bad file.
    """
    from biobench_sar.rag.ingestor import ingest_file
    from biobench_sar.rag.quiz_bank import chunks_with_answers, parse_quiz_file

    reingested: list[str] = []
    flagged: list[str] = []
    for document in store.list_sources():
        source = str(document.get("source") or "")
        try:
            texts = _stored_texts(store, source, int(document.get("chunks") or 0))
            if not any(chunks_with_answers([(source, text) for text in texts])):
                continue
            seed = directory / source if directory is not None and source else None
            if seed is not None and seed.is_file() and _is_seed_copy(texts, seed):
                store.replace_source(source, ingest_file(seed, strip_answers=True))
                if quiz_bank is not None:
                    quiz_bank.replace_source(source, parse_quiz_file(seed, source))
                reingested.append(source)
            else:
                flagged.append(source)
        except Exception:
            log.exception("could not check %s for stored answers", source)
    if reingested:
        log.info("re-ingested %s without their answers", ", ".join(reingested))
    if flagged:
        log.warning(
            "these files were stored with their answers before answers were stripped at upload; re-upload them: %s",
            ", ".join(flagged),
        )
    return reingested, flagged


def migrate_app_knowledge(app: Flask, settings: Settings) -> tuple[list[str], list[str]]:
    """migrate_answer_keys for a running math app: flagged files show `has_answers` in the document list."""
    if settings.profile != "math":
        return [], []
    assistant: Assistant = app.config["SAR_ASSISTANT"]
    reingested, flagged = migrate_answer_keys(
        app.config["SAR_SERVICES"].store, settings.knowledge_dir, quiz_bank=assistant.quiz_bank
    )
    app.config["SAR_FLAG_ANSWERS"](flagged)
    if reingested or flagged:
        app.config["SAR_NOTIFY_DOCUMENTS"]()
    return reingested, flagged


def prepare_knowledge(app: Flask, settings: Settings) -> None:
    """Startup work on the knowledge base, off the request threads: strip stored answers, then seed a fresh store."""
    try:
        migrate_app_knowledge(app, settings)
    except Exception:
        log.exception("could not check the stored files for answers")
    if settings.seed_knowledge:
        seed_app_knowledge(app, settings)


def _port_free(host: str, port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _warm_up(svc: Services) -> None:
    """Load the VLM and Whisper in the background so the first question isn't the slow one."""
    import numpy as np

    started = time.monotonic()
    warmup = getattr(svc.vlm, "warmup", None)
    if callable(warmup):
        warmup()
    if getattr(svc.audio, "available", False):
        try:
            svc.audio.transcribe(np.zeros(16000, dtype=np.float32))
        except Exception:
            log.exception("whisper warm-up failed")
    log.info("models warmed up in %.1fs", time.monotonic() - started)


def _request_camera_access(camera_index: int) -> None:
    # macOS only shows the camera permission prompt when the first capture opens on the main thread;
    # VisionLoop opens it from a worker thread, where AVFoundation refuses to ask.
    if sys.platform != "darwin":
        return
    import cv2

    cap = cv2.VideoCapture(camera_index)
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    if not ok:
        log.warning(
            "Camera %d not readable. Allow camera access for your terminal/app in "
            "System Settings > Privacy & Security > Camera, then restart.",
            camera_index,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "httpcore", "sentence_transformers", "werkzeug", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    settings = Settings()
    if settings.port in BROWSER_BLOCKED_PORTS:
        log.error(
            "Browsers refuse to open port %d, so no page could load. Set SAR_PORT to another port (5050 is the default).",
            settings.port,
        )
        raise SystemExit(1)
    if not _port_free(settings.host, settings.port):
        log.error("Port %d is already in use. Stop the other server or set SAR_PORT to a free port.", settings.port)
        raise SystemExit(1)
    app, socketio = create_app(settings)
    svc: Services = app.config["SAR_SERVICES"]
    _request_camera_access(settings.camera_index)
    svc.vision.start()
    threading.Thread(target=prepare_knowledge, args=(app, settings), daemon=True, name="knowledge").start()
    threading.Thread(target=_warm_up, args=(svc,), daemon=True, name="warm-up").start()
    try:
        svc.audio.start()
    except Exception:
        log.exception("audio failed to start; use typed queries")
    gate: JoinPin | None = app.config["SAR_JOIN_PIN"]
    if settings.lan and not lan_requested():
        log.info("SAR_HOST=%r reaches the network, so the phone PIN gate is on", settings.host)
    if settings.lan and is_loopback_bind(settings.host):
        log.warning("SAR_LAN=1 with SAR_HOST=%s: phones can't reach this server; unset SAR_HOST", settings.host)
    addresses = phone_addresses(settings.host) if gate is not None else []
    log.info("%s", startup_banner(settings, gate.pin if gate is not None else None, addresses))
    try:
        socketio.run(app, host=settings.host, port=settings.port, allow_unsafe_werkzeug=True)
    finally:
        svc.vision.stop()
        with contextlib.suppress(Exception):
            svc.audio.stop()


if __name__ == "__main__":
    main()
