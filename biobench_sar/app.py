"""Flask + Socket.IO server wiring vision, RAG, VLM, audio and TTS together."""

from __future__ import annotations

import contextlib
import logging
import sys
import threading
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename

from biobench_sar.assistant import Assistant
from biobench_sar.config import PACKAGE_DIR, Settings

log = logging.getLogger("biobench_sar.app")
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
CALIBRATION_SETTLE_SECONDS = 1.5
MAX_SEARCH_RESULTS = 10
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
DEFAULT_PORTS = {"http": 80, "https": 443}


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

    assistant = Assistant(settings, socketio.emit, socketio.start_background_task)
    svc = services or build_services(settings, assistant.handle_query, assistant.on_state, assistant.should_listen)
    assistant.services = svc
    if services is None:
        assistant.calibrated = _load_initial_calibration(settings)

    def documents_body() -> dict[str, Any]:
        documents = svc.store.list_sources()
        active = assistant.active_sources
        listed = [{**d, "active": active is None or d["source"] in active} for d in documents]
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

    # ---- pages -------------------------------------------------------------
    def page(name: str) -> str:
        return render_template(
            name, projector_width=settings.projector_width, projector_height=settings.projector_height
        )

    @app.get("/")
    def index() -> str:
        return page("index.html")

    @app.get("/precharge")
    @app.get("/dashboard")
    def precharge() -> str:
        return page("precharge.html")

    # ---- API ---------------------------------------------------------------
    @app.post("/api/upload")
    def upload() -> Any:
        from biobench_sar.rag.ingestor import SUPPORTED_EXTENSIONS, ingest_file

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
                chunks = [replace(c, source=name, metadata={**c.metadata, "source": name}) for c in ingest_file(path)]
                # Replace, not add: ids are content hashes, so an edited file would keep its old chunks.
                added, removed = svc.store.replace_source(name, chunks)
                total += added
                count = svc.store.source_count(name)
                (stored if count else emptied).append(name)
                results.append({"name": name, "chunks": count, "replaced": removed})
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
                "audio_available": bool(getattr(svc.audio, "available", False)),
                "tts_backend": settings.tts_backend,
                "listen_mode": settings.listen_mode,
                "state": assistant.state,
                "busy": assistant.busy,
                "last_latency_s": assistant.last_latency_s,
                "uptime_s": round(time.monotonic() - assistant.started_at),
            }
        )

    @app.get("/api/config")
    def api_config() -> Any:
        return jsonify(
            {
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
            }
        )

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
        if not deleted:
            return jsonify({"deleted": 0, "error": "no such document"}), 404
        notify_documents_changed()
        return jsonify({"deleted": deleted})

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
        svc.audio.submit_text(text)
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
    @socketio.on("connect")
    def on_connect(_auth: Any = None) -> None:
        # A client that (re)connects mid-query must not show a stale state.
        socketio.emit("status", {"state": assistant.state}, to=request.sid)

    @socketio.on("text_query")
    def on_text_query(data: Any) -> None:
        text = str((data or {}).get("text") or "") if isinstance(data, dict) else ""
        if text.strip():
            svc.audio.submit_text(text)

    @socketio.on("trigger_listen")
    def on_trigger_listen(_data: Any = None) -> None:
        svc.audio.trigger()

    app.config["SAR_SERVICES"] = svc
    app.config["SAR_ASSISTANT"] = assistant
    app.config["SAR_HANDLE_QUERY"] = assistant.handle_query
    app.config["SAR_ON_STATE"] = assistant.on_state
    app.config["SAR_SHOULD_LISTEN"] = assistant.should_listen
    return app, socketio


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


def seed_knowledge(store: Any, directory: Path) -> int:
    """Ingest the bundled scenario documents into an empty store so a fresh install can answer right away."""
    from biobench_sar.rag.ingestor import SUPPORTED_EXTENSIONS, ingest_file

    if store.count() > 0 or not directory.is_dir():
        return 0
    added = 0
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            added += store.add_chunks(ingest_file(path))
        except Exception:
            log.exception("could not seed %s", path.name)
    log.info("seeded knowledge base with %d chunks from %s", added, directory)
    return added


def _port_free(host: str, port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
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
    if not _port_free(settings.host, settings.port):
        log.error("Port %d is already in use. Stop the other server or set SAR_PORT to a free port.", settings.port)
        raise SystemExit(1)
    app, socketio = create_app(settings)
    svc: Services = app.config["SAR_SERVICES"]
    _request_camera_access(settings.camera_index)
    svc.vision.start()
    if settings.seed_knowledge:
        threading.Thread(
            target=seed_knowledge, args=(svc.store, PACKAGE_DIR / "knowledge"), daemon=True, name="seed"
        ).start()
    threading.Thread(target=_warm_up, args=(svc,), daemon=True, name="warm-up").start()
    try:
        svc.audio.start()
    except Exception:
        log.exception("audio failed to start; use typed queries")
    base = f"http://{settings.host}:{settings.port}"
    log.info(
        "\n\n  SAR Whiteboard Assistant\n  Projector  %s/          (fullscreen on the projector, press F)\n"
        "  Dashboard  %s/dashboard\n  Model      %s · %s\n",
        base,
        base,
        settings.vlm_backend,
        settings.vlm_model,
    )
    try:
        socketio.run(app, host=settings.host, port=settings.port, allow_unsafe_werkzeug=True)
    finally:
        svc.vision.stop()
        with contextlib.suppress(Exception):
            svc.audio.stop()


if __name__ == "__main__":
    main()
