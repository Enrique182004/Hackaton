from __future__ import annotations

import io
import sys
import threading
import time
import types
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest


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
    assert set(data) == {
        "fps", "camera_running", "calibrated", "doc_chunks",
        "vlm_backend", "vlm_model", "audio_available", "state",
    }
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
    assert avatar["animation_state"] == "pointing" and avatar["target_x"] == 100
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
