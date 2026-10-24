from __future__ import annotations

import dataclasses
import io
import sys
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from biobench_sar import app as app_module
from biobench_sar.config import Settings
from biobench_sar.schemas import RetrievedChunk


@pytest.fixture(autouse=True)
def fake_cv_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    vision = types.ModuleType("biobench_sar.core.vision_loop")
    vision.encode_jpeg = lambda frame, quality=80: b"\xff\xd8fakejpeg"  # type: ignore[attr-defined]
    vision.find_ink_regions = lambda frame, **kw: []  # type: ignore[attr-defined]
    homography = types.ModuleType("biobench_sar.core.homography")
    homography.generate_checkerboard = lambda w, h, cols=9, rows=6: np.full((h, w, 3), 255, np.uint8)  # type: ignore[attr-defined]
    homography.calibrate_from_frame = lambda *a, **k: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "biobench_sar.core.vision_loop", vision)
    monkeypatch.setitem(sys.modules, "biobench_sar.core.homography", homography)
    monkeypatch.setattr("biobench_sar.app.CALIBRATION_SETTLE_SECONDS", 0)


def collect(client: Any, names: list[str], timeout: float = 3.0) -> list[dict[str, Any]]:
    got: list[dict[str, Any]] = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        got.extend(client.get_received())
        if all(any(m["name"] == n for m in got) for n in names):
            time.sleep(0.05)
            got.extend(client.get_received())
            return got
        time.sleep(0.02)
    return got


def test_pages_render(built: Any) -> None:
    app, _ = built
    client = app.test_client()
    r = client.get("/")
    assert r.status_code == 200 and b"640x360" in r.data
    assert client.get("/precharge").status_code == 200


def test_status_shape(built: Any) -> None:
    app, _ = built
    data = app.test_client().get("/api/status").get_json()
    assert {
        "fps",
        "camera_running",
        "calibrated",
        "doc_chunks",
        "vlm_backend",
        "vlm_model",
        "audio_available",
        "state",
        "busy",
        "tts_backend",
        "listen_mode",
        "last_latency_s",
        "uptime_s",
    } <= set(data)
    assert data["state"] == "idle" and data["audio_available"] is True


def test_upload_good_and_bad(built: Any, fakes: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    ingestor = types.ModuleType("biobench_sar.rag.ingestor")

    @dataclass(frozen=True)
    class Chunk:
        text: str
        source: str
        metadata: dict[str, str | int]

    def ingest_file(path: Any) -> list[Chunk]:
        return [Chunk(t, str(path), {"source": str(path)}) for t in ("a", "b")]

    ingestor.ingest_file = ingest_file  # type: ignore[attr-defined]
    ingestor.SUPPORTED_EXTENSIONS = frozenset({".txt"})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "biobench_sar.rag.ingestor", ingestor)
    app, _ = built
    r = app.test_client().post(
        "/api/upload",
        data={"files[]": [(io.BytesIO(b"hello"), "notes.txt"), (io.BytesIO(b"MZ"), "bad.exe")]},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["total_chunks"] == 2
    by_name = {f["name"]: f for f in body["files"]}
    assert by_name["notes.txt"]["chunks"] == 2 and "error" not in by_name["notes.txt"]
    assert "error" in by_name["bad.exe"]
    assert fakes.store.count() == 2
    assert not any((tmp_path / "uploads").iterdir()), "uploaded files are deleted after ingest"


def test_query_event_order(built: Any) -> None:
    app, socketio = built
    sio = socketio.test_client(app)
    sio.get_received()
    r = app.test_client().post("/api/query", json={"text": "what is this?"})
    assert r.status_code == 202 and r.get_json() == {"accepted": True}
    got = collect(sio, ["speak"])
    names = [m["name"] for m in got]
    assert names[0] == "transcript"
    assert names.index("status") < names.index("avatar_update") < names.index("speak")
    assert names[-1] == "status"
    avatar = next(m for m in got if m["name"] == "avatar_update")["args"][0]
    # No camera frame -> no ink, so the server downgrades the fake VLM's "pointing" to "idle".
    assert avatar["animation_state"] == "idle" and avatar["target_x"] == 100
    speak = next(m for m in got if m["name"] == "speak")["args"][0]
    assert speak == {"url": "/tts/fake.mp3", "text": "Look here"}
    assert got[-1]["args"][0]["state"] == "idle"


def test_socket_text_query(built: Any) -> None:
    app, socketio = built
    sio = socketio.test_client(app)
    sio.get_received()
    sio.emit("text_query", {"text": "hi"})
    names = [m["name"] for m in collect(sio, ["speak"])]
    assert "speak" in names


def test_busy_single_flight(built: Any, fakes: Any) -> None:
    app, socketio = built
    sio = socketio.test_client(app)
    gate = threading.Event()
    fakes.vlm.gate = gate
    handle = app.config["SAR_HANDLE_QUERY"]
    handle("first")
    time.sleep(0.2)
    sio.get_received()
    handle("second")
    got = sio.get_received()
    assert any(m["name"] == "status" and m["args"][0].get("detail") == "busy" for m in got)
    assert app.config["SAR_SHOULD_LISTEN"]() is False, "no wake listening while a query is in flight"
    gate.set()
    collect(sio, ["speak"])
    assert fakes.vlm.calls == ["first"]
    time.sleep(0.1)
    assert app.config["SAR_SHOULD_LISTEN"]() is False, "still False while the answer is being spoken"


def test_pipeline_error_emits_error_avatar(built: Any, fakes: Any) -> None:
    app, socketio = built
    sio = socketio.test_client(app)
    sio.get_received()
    fakes.vlm.error = RuntimeError("boom")
    app.config["SAR_HANDLE_QUERY"]("q")
    got = collect(sio, ["avatar_update"])
    avatar = next(m for m in got if m["name"] == "avatar_update")["args"][0]
    assert avatar["animation_state"] == "error"
    time.sleep(0.1)
    got += sio.get_received()
    assert got[-1]["name"] == "status" and got[-1]["args"][0]["state"] == "idle"
    fakes.vlm.error = None
    app.config["SAR_HANDLE_QUERY"]("again")  # lock released after failure
    assert any(m["name"] == "speak" for m in collect(sio, ["speak"]))


def test_rag_failure_is_tolerated(built: Any, fakes: Any) -> None:
    app, socketio = built
    sio = socketio.test_client(app)
    fakes.store.fail_query = True
    app.config["SAR_HANDLE_QUERY"]("q")
    got = collect(sio, ["speak"])
    assert any(m["name"] == "speak" for m in got)


def test_frame_jpg(built: Any, fakes: Any) -> None:
    app, _ = built
    client = app.test_client()
    assert client.get("/api/frame.jpg").status_code == 503
    fakes.vision.frame = np.zeros((10, 10, 3), np.uint8)
    r = client.get("/api/frame.jpg?warped=1")
    assert r.status_code == 200 and r.mimetype == "image/jpeg"


def test_calibration_png(built: Any) -> None:
    app, _ = built
    r = app.test_client().get("/calibration.png")
    assert r.status_code == 200 and r.data[:8] == b"\x89PNG\r\n\x1a\n"


def test_calibrate_failure_and_success(built: Any, fakes: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    app, socketio = built
    sio = socketio.test_client(app)
    client = app.test_client()
    assert client.post("/api/calibrate").get_json()["ok"] is False  # no frame
    fakes.vision.frame = np.zeros((10, 10, 3), np.uint8)
    assert client.post("/api/calibrate").get_json()["ok"] is False  # board not found

    class FakeCal:
        def save(self, path: Any) -> None:
            path.write_text("{}")

    monkeypatch.setattr(sys.modules["biobench_sar.core.homography"], "calibrate_from_frame", lambda *a, **k: FakeCal())
    sio.get_received()
    assert client.post("/api/calibrate").get_json() == {"ok": True}
    names = [m["name"] for m in sio.get_received()]
    assert names == ["show_calibration", "hide_calibration"]
    assert isinstance(fakes.vision.calibration, FakeCal)
    assert client.get("/api/status").get_json()["calibrated"] is True


def test_listen_and_tts_file(built: Any, fakes: Any, settings: Any) -> None:
    app, _ = built
    client = app.test_client()
    assert client.post("/api/listen").status_code == 202 and fakes.audio.triggered == 1
    (settings.tts_dir / "x.mp3").write_bytes(b"abc")
    assert client.get("/tts/x.mp3").data == b"abc"


# ---- dashboard / admin endpoints ------------------------------------------------


def _chunk(text: str, source: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(text=text, source=source, score=score, metadata={})


def test_dashboard_renders_like_precharge(built: Any) -> None:
    client = built[0].test_client()
    dashboard, precharge = client.get("/dashboard"), client.get("/precharge")
    assert dashboard.status_code == 200
    assert dashboard.data == precharge.data and b"precharge 640" in dashboard.data


def test_config_keys_and_no_secret(built: Any, settings: Settings) -> None:
    r = built[0].test_client().get("/api/config")
    assert r.status_code == 200
    data = r.get_json()
    assert {
        "camera_index",
        "camera_size",
        "projector_size",
        "checkerboard",
        "vlm_backend",
        "vlm_model",
        "vlm_base_url",
        "embed_model",
        "rag_top_k",
        "whisper_model",
        "listen_mode",
        "wake_word",
        "tts_backend",
        "tts_voice",
    } <= set(data)
    assert data["projector_size"] == [640, 360]
    assert "vlm_api_key" not in data and "api_key" not in r.get_data(as_text=True).lower()


def test_config_does_not_leak_configured_api_key(
    settings: Settings, fakes: app_module.Services, templates_dir: Path
) -> None:
    secret = "sk-super-secret-123"
    flask_app, _ = app_module.create_app(
        dataclasses.replace(settings, vlm_api_key=secret), fakes, template_folder=str(templates_dir)
    )
    assert secret not in flask_app.test_client().get("/api/config").get_data(as_text=True)


def test_history_empty_then_after_query(built: Any) -> None:
    app, socketio = built
    client = app.test_client()
    assert client.get("/api/history").get_json() == {"interactions": []}
    sio = socketio.test_client(app)
    client.post("/api/query", json={"text": "what is this?"})
    collect(sio, ["interaction"])
    items = client.get("/api/history").get_json()["interactions"]
    assert len(items) == 1
    assert items[0]["query"] == "what is this?" and items[0]["animation_state"] == "idle"
    assert items[0]["error"] is False


def test_documents_lists_sources_with_total(built: Any, fakes: Any) -> None:
    client = built[0].test_client()
    assert client.get("/api/documents").get_json() == {"documents": [], "total_chunks": 0}
    fakes.store.sources = [
        {"source": "a.pdf", "type": "pdf", "chunks": 3},
        {"source": "b.txt", "type": "txt", "chunks": 2},
    ]
    data = client.get("/api/documents").get_json()
    assert [d["source"] for d in data["documents"]] == ["a.pdf", "b.txt"]
    assert data["total_chunks"] == 5


def test_delete_document_found_and_missing(built: Any, fakes: Any) -> None:
    client = built[0].test_client()
    fakes.store.sources = [{"source": "a.pdf", "type": "pdf", "chunks": 3}]
    r = client.delete("/api/documents/a.pdf")
    assert r.status_code == 200 and r.get_json() == {"deleted": 3}
    missing = client.delete("/api/documents/a.pdf")
    assert missing.status_code == 404 and missing.get_json()["deleted"] == 0


def test_delete_document_name_with_slash(built: Any, fakes: Any) -> None:
    fakes.store.sources = [{"source": "dir/a.pdf", "type": "pdf", "chunks": 1}]
    assert built[0].test_client().delete("/api/documents/dir/a.pdf").get_json() == {"deleted": 1}


def test_search_returns_rounded_results(built: Any, fakes: Any) -> None:
    fakes.store.query_results = [_chunk("hello", "a.pdf", 0.123456)]
    r = built[0].test_client().get("/api/search?q=hello&k=2")
    assert r.get_json() == {"results": [{"text": "hello", "source": "a.pdf", "score": 0.123}]}
    assert fakes.store.query_calls == [("hello", 2)]


@pytest.mark.parametrize(
    ("k_arg", "expected"),
    [("0", 1), ("-5", 1), ("1", 1), ("10", 10), ("11", 10), ("999", 10), ("abc", None), ("", None)],
)
def test_search_k_clamped(built: Any, fakes: Any, settings: Settings, k_arg: str, expected: int | None) -> None:
    built[0].test_client().get(f"/api/search?q=x&k={k_arg}")
    assert fakes.store.query_calls == [("x", settings.rag_top_k if expected is None else expected)]


def test_search_default_k_is_rag_top_k(built: Any, fakes: Any, settings: Settings) -> None:
    built[0].test_client().get("/api/search?q=x")
    assert fakes.store.query_calls == [("x", settings.rag_top_k)]


@pytest.mark.parametrize("url", ["/api/search", "/api/search?q=", "/api/search?q=%20%20"])
def test_search_empty_query_returns_empty_without_querying(built: Any, fakes: Any, url: str) -> None:
    r = built[0].test_client().get(url)
    assert r.status_code == 200 and r.get_json() == {"results": []}
    assert fakes.store.query_calls == []


def test_ink_without_frame_is_503(built: Any) -> None:
    r = built[0].test_client().get("/api/ink")
    assert r.status_code == 503
    body = r.get_json()
    assert body["regions"] == [] and body["error"] == "no frame" and body["size"] == [640, 360]


def test_ink_with_frame_returns_regions_and_size(built: Any, fakes: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    region = {"x": 1, "y": 2, "w": 3, "h": 4, "cx": 2, "cy": 4}
    monkeypatch.setattr(sys.modules["biobench_sar.core.vision_loop"], "find_ink_regions", lambda frame, **kw: [region])
    fakes.vision.frame = np.zeros((20, 30, 3), np.uint8)
    r = built[0].test_client().get("/api/ink")
    assert r.status_code == 200
    assert r.get_json() == {"regions": [region], "size": [30, 20]}


def test_query_empty_text_is_400(built: Any) -> None:
    client = built[0].test_client()
    assert client.post("/api/query", json={"text": "   "}).status_code == 400
    assert client.post("/api/query", json={}).status_code == 400


def test_query_returns_409_while_busy(built: Any, fakes: Any) -> None:
    app, _ = built
    gate = threading.Event()
    fakes.vlm.gate = gate
    client = app.test_client()
    try:
        assert client.post("/api/query", json={"text": "first"}).status_code == 202
        deadline = time.time() + 2
        while not app.config["SAR_ASSISTANT"].busy and time.time() < deadline:
            time.sleep(0.01)
        r = client.post("/api/query", json={"text": "second"})
        assert r.status_code == 409
        assert r.get_json() == {"accepted": False, "error": "busy"}
    finally:
        gate.set()
    deadline = time.time() + 3
    while app.config["SAR_ASSISTANT"].busy and time.time() < deadline:
        time.sleep(0.01)
    assert fakes.vlm.calls == ["first"]


def test_calibrate_503_immediately_without_frame(built: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("biobench_sar.app.CALIBRATION_SETTLE_SECONDS", 30)
    started = time.monotonic()
    r = built[0].test_client().post("/api/calibrate")
    assert r.status_code == 503 and r.get_json()["ok"] is False
    assert time.monotonic() - started < 1


def test_calibrate_409_when_already_running(built: Any, fakes: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    app, _ = built
    fakes.vision.frame = np.zeros((10, 10, 3), np.uint8)
    entered, release = threading.Event(), threading.Event()

    def blocking_calibrate(*args: Any, **kwargs: Any) -> None:
        entered.set()
        release.wait(5)
        return None

    monkeypatch.setattr(sys.modules["biobench_sar.core.homography"], "calibrate_from_frame", blocking_calibrate)
    results: list[Any] = []
    first = threading.Thread(target=lambda: results.append(app.test_client().post("/api/calibrate")))
    first.start()
    try:
        assert entered.wait(2)
        second = app.test_client().post("/api/calibrate")
        assert second.status_code == 409
        assert second.get_json() == {"ok": False, "error": "calibration already running"}
    finally:
        release.set()
        first.join(3)
    assert results[0].get_json()["ok"] is False  # board not detected
    # lock released: a new run is accepted (not 409)
    assert app.test_client().post("/api/calibrate").status_code == 200


def test_socket_connect_receives_current_status(built: Any, fakes: Any) -> None:
    app, socketio = built
    sio = socketio.test_client(app)
    got = sio.get_received()
    assert [m["name"] for m in got] == ["status"]
    assert got[0]["args"][0] == {"state": "idle"}


def test_socket_connect_mid_query_gets_current_state(built: Any, fakes: Any) -> None:
    app, socketio = built
    gate = threading.Event()
    fakes.vlm.gate = gate
    app.config["SAR_HANDLE_QUERY"]("slow")
    try:
        assistant = app.config["SAR_ASSISTANT"]
        deadline = time.time() + 2
        while assistant.state != "thinking" and time.time() < deadline:
            time.sleep(0.01)
        late = socketio.test_client(app)
        statuses = [m["args"][0] for m in late.get_received() if m["name"] == "status"]
        assert {"state": "thinking"} in statuses
    finally:
        gate.set()
        collect(socketio.test_client(app), ["speak"])
