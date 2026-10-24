"""Flask + Socket.IO server wiring vision, RAG, VLM, audio and TTS together."""

from __future__ import annotations

import logging
import sys
import threading
import uuid
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename

from biobench_sar.config import Settings
from biobench_sar.schemas import error_action

log = logging.getLogger("biobench_sar.app")
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

CALIBRATION_SETTLE_SECONDS = 1.5


@dataclass
class Services:
    vision: Any  # VisionLoop
    store: Any  # VectorStore
    vlm: Any  # VLMClient
    audio: Any  # AudioTrigger (on_query/on_state are wired by create_app)
    tts: Any  # TTSEngine


def build_services(settings: Settings, on_query: Any, on_state: Any) -> Services:
    """Construct real services. Imports are lazy so importing app.py stays light."""
    from biobench_sar.core.audio_trigger import AudioTrigger
    from biobench_sar.core.homography import Calibration
    from biobench_sar.core.vision_loop import VisionLoop
    from biobench_sar.llm.tts_engine import TTSEngine
    from biobench_sar.llm.vlm_client import VLMClient
    from biobench_sar.rag.vector_store import VectorStore

    calibration = Calibration.load(settings.calibration_file)
    if calibration is None:
        calibration = Calibration.identity(
            (settings.camera_width, settings.camera_height), settings.projector_size
        )
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
    socketio = SocketIO(app, async_mode="threading", cors_allowed_origins=None)

    state: dict[str, Any] = {"value": "idle", "calibrated": False}
    busy = threading.Lock()

    def emit_status(value: str, detail: str | None = None) -> None:
        state["value"] = value
        payload: dict[str, Any] = {"state": value}
        if detail:
            payload["detail"] = detail
        socketio.emit("status", payload)

    def run_pipeline(text: str) -> None:
        try:
            socketio.emit("transcript", {"text": text})
            emit_status("thinking")
            from biobench_sar.core.vision_loop import encode_jpeg, find_ink_regions

            warped = svc.vision.latest_warped()
            have_frame = warped is not None
            ink = find_ink_regions(warped) if have_frame else []
            try:
                context = svc.store.query(text, settings.rag_top_k)
            except Exception:
                log.exception("RAG query failed; continuing without context")
                context = []
            image = encode_jpeg(warped) if have_frame else None
            action = svc.vlm.evaluate(text, image, context, ink, settings.projector_size)
            socketio.emit("avatar_update", action.to_dict())
            emit_status("speaking")
            url_name = svc.tts.speak(action.spoken_text)
            socketio.emit(
                "speak",
                {"url": f"/tts/{url_name}" if url_name else None, "text": action.spoken_text},
            )
            emit_status("idle")
        except Exception:
            log.exception("query pipeline failed")
            socketio.emit(
                "avatar_update",
                error_action("Sorry, something went wrong. Please try again.", settings.projector_size).to_dict(),
            )
            emit_status("idle")
        finally:
            busy.release()

    def handle_query(text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        if not busy.acquire(blocking=False):
            emit_status(state["value"], "busy")
            return
        try:
            socketio.start_background_task(run_pipeline, text)
        except Exception:
            busy.release()
            raise

    def on_state(value: str) -> None:
        # While the pipeline is running, it owns the state machine.
        if value == "idle" and busy.locked():
            return
        emit_status(value)

    svc = services if services is not None else build_services(settings, handle_query, on_state)
    if services is None:
        state["calibrated"] = _load_initial_calibration(settings, svc)

    # ---- pages -------------------------------------------------------------
    def page(name: str) -> str:
        return render_template(
            name, projector_width=settings.projector_width, projector_height=settings.projector_height
        )

    @app.get("/")
    def index() -> str:
        return page("index.html")

    @app.get("/precharge")
    def precharge() -> str:
        return page("precharge.html")

    # ---- API ---------------------------------------------------------------
    @app.post("/api/upload")
    def upload() -> Any:
        from biobench_sar.rag.ingestor import SUPPORTED_EXTENSIONS, ingest_file

        results: list[dict[str, Any]] = []
        total = 0
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
                added = svc.store.add_chunks(chunks) if chunks else 0
                total += added
                results.append({"name": name, "chunks": added})
            except Exception as exc:  # UnsupportedFileType, parse errors, embed errors
                log.warning("upload %s failed: %s", name, exc)
                results.append({"name": name, "chunks": 0, "error": str(exc) or type(exc).__name__})
            finally:
                path.unlink(missing_ok=True)
        return jsonify({"files": results, "total_chunks": total})

    @app.get("/api/status")
    def status() -> Any:
        try:
            doc_chunks = svc.store.count()
        except Exception:
            doc_chunks = 0
        calibrated = bool(state.get("calibrated", False))
        return jsonify(
            {
                "fps": float(getattr(svc.vision, "fps", 0.0)),
                "camera_running": bool(getattr(svc.vision, "is_running", False)) and svc.vision.latest_frame() is not None,
                "calibrated": calibrated,
                "doc_chunks": doc_chunks,
                "vlm_backend": settings.vlm_backend,
                "vlm_model": settings.vlm_model,
                "audio_available": bool(getattr(svc.audio, "available", False)),
                "state": state["value"],
            }
        )

    @app.post("/api/query")
    def api_query() -> Any:
        body = request.get_json(silent=True) or {}
        text = str(body.get("text") or "").strip()
        if not text:
            return jsonify({"accepted": False, "error": "empty text"}), 400
        svc.audio.submit_text(text)
        return jsonify({"accepted": True}), 202

    @app.post("/api/listen")
    def api_listen() -> Any:
        svc.audio.trigger()
        return jsonify({"accepted": True}), 202

    @app.post("/api/calibrate")
    def api_calibrate() -> Any:
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
        state["calibrated"] = True
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
    @socketio.on("text_query")
    def on_text_query(data: Any) -> None:
        text = str((data or {}).get("text") or "") if isinstance(data, dict) else ""
        if text.strip():
            svc.audio.submit_text(text)

    @socketio.on("trigger_listen")
    def on_trigger_listen(_data: Any = None) -> None:
        svc.audio.trigger()

    app.config["SAR_SERVICES"] = svc
    app.config["SAR_HANDLE_QUERY"] = handle_query
    app.config["SAR_ON_STATE"] = on_state
    return app, socketio


def _load_initial_calibration(settings: Settings, svc: Services) -> bool:
    """True if a saved calibration exists (build_services already applied it, else identity)."""
    try:
        from biobench_sar.core.homography import Calibration

        return Calibration.load(settings.calibration_file) is not None
    except Exception:
        log.exception("could not load calibration")
        return False


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
            "System Settings > Privacy & Security > Camera, then restart.", camera_index
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings()
    app, socketio = create_app(settings)
    svc: Services = app.config["SAR_SERVICES"]
    _request_camera_access(settings.camera_index)
    svc.vision.start()
    try:
        svc.audio.start()
    except Exception:
        log.exception("audio failed to start; use typed queries")
    log.info("Open http://%s:%d/ on the projector and /precharge on your laptop", settings.host, settings.port)
    try:
        socketio.run(app, host=settings.host, port=settings.port, allow_unsafe_werkzeug=True)
    finally:
        svc.vision.stop()
        try:
            svc.audio.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
