from __future__ import annotations

import dataclasses
import io
import json
import sys
import threading
import time
import types
import urllib.request
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
        "active_documents",
        "total_documents",
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
    assert data["active_documents"] == 0 and data["total_documents"] == 0


@pytest.fixture
def ingest_texts(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The chunk texts the fake ingestor returns for any .txt file; mutate to simulate an edited file."""
    texts = ["a", "b"]
    ingestor = types.ModuleType("biobench_sar.rag.ingestor")

    @dataclass(frozen=True)
    class Chunk:
        text: str
        source: str
        metadata: dict[str, str | int]

    def ingest_file(path: Any) -> list[Chunk]:
        return [Chunk(t, str(path), {"source": str(path)}) for t in texts]

    ingestor.ingest_file = ingest_file  # type: ignore[attr-defined]
    ingestor.SUPPORTED_EXTENSIONS = frozenset({".txt"})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "biobench_sar.rag.ingestor", ingestor)
    return texts


def _upload(client: Any, name: str = "notes.txt", **kwargs: Any) -> Any:
    return client.post(
        "/api/upload", data={"files[]": [(io.BytesIO(b"hello"), name)]}, content_type="multipart/form-data", **kwargs
    )


def test_upload_good_and_bad(built: Any, fakes: Any, ingest_texts: list[str], tmp_path: Any) -> None:
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


def test_reupload_replaces_edited_file_and_reports_counts(built: Any, fakes: Any, ingest_texts: list[str]) -> None:
    client = built[0].test_client()
    first = _upload(client).get_json()
    assert first == {"files": [{"name": "notes.txt", "chunks": 2, "replaced": 0}], "total_chunks": 2}
    ingest_texts[:] = ["a", "c"]
    edited = _upload(client).get_json()
    assert edited == {"files": [{"name": "notes.txt", "chunks": 2, "replaced": 1}], "total_chunks": 1}
    assert sorted(c.text for c in fakes.store.chunks) == ["a", "c"], "stale chunk 'b' is gone"
    unchanged = _upload(client).get_json()
    assert unchanged == {"files": [{"name": "notes.txt", "chunks": 2, "replaced": 0}], "total_chunks": 0}


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
    assert client.get("/api/documents").get_json() == {
        "documents": [],
        "total_chunks": 0,
        "active_count": 0,
        "all_active": True,
    }
    fakes.store.sources = [
        {"source": "a.pdf", "type": "pdf", "chunks": 3},
        {"source": "b.txt", "type": "txt", "chunks": 2},
    ]
    data = client.get("/api/documents").get_json()
    assert data == {
        "documents": [
            {"source": "a.pdf", "type": "pdf", "chunks": 3, "active": True},
            {"source": "b.txt", "type": "txt", "chunks": 2, "active": True},
        ],
        "total_chunks": 5,
        "active_count": 2,
        "all_active": True,
    }


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


@pytest.mark.parametrize("payload", ["[1]", '"abc"', "5", "true", "null"])
def test_query_non_object_json_is_400(built: Any, fakes: Any, payload: str) -> None:
    r = built[0].test_client().post("/api/query", data=payload, content_type="application/json")
    assert r.status_code == 400 and r.get_json() == {"accepted": False, "error": "empty text"}
    assert fakes.vlm.calls == []


# ---- cross-origin guard ---------------------------------------------------------

EVIL = {"Origin": "http://evil.example"}


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Origin": "http://localhost"},
        {"Origin": "HTTP://LOCALHOST:80"},
        {"Referer": "http://localhost/dashboard"},
        {"Origin": "http://localhost", "Referer": "http://evil.example/"},
    ],
)
def test_post_allowed_same_origin_or_without_headers(built: Any, fakes: Any, headers: dict[str, str]) -> None:
    assert built[0].test_client().post("/api/listen", headers=headers).status_code == 202
    assert fakes.audio.triggered == 1


@pytest.mark.parametrize(
    "headers",
    [
        EVIL,
        {"Origin": "https://localhost"},
        {"Origin": "http://localhost:8080"},
        {"Origin": "null"},
        {"Origin": "http://localhost:bad"},
        {"Referer": "http://evil.example/page"},
    ],
)
def test_cross_origin_post_blocked(built: Any, fakes: Any, headers: dict[str, str]) -> None:
    r = built[0].test_client().post("/api/listen", headers=headers)
    assert r.status_code == 403 and r.get_json() == {"error": "cross-origin request blocked"}
    assert fakes.audio.triggered == 0


def test_cross_origin_blocks_upload_calibrate_and_delete(built: Any, fakes: Any, ingest_texts: list[str]) -> None:
    client = built[0].test_client()
    fakes.vision.frame = np.zeros((10, 10, 3), np.uint8)
    fakes.store.sources = [{"source": "a.pdf", "type": "pdf", "chunks": 1}]
    assert _upload(client, headers=EVIL).status_code == 403
    assert client.post("/api/calibrate", headers=EVIL).status_code == 403
    assert client.delete("/api/documents/a.pdf", headers=EVIL).status_code == 403
    assert fakes.store.chunks == [] and fakes.store.sources != []


def test_cross_origin_get_still_allowed(built: Any) -> None:
    assert built[0].test_client().get("/api/status", headers=EVIL).status_code == 200


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
    assert [m["name"] for m in got] == ["status", "tutor"]
    assert got[0]["args"][0] == {"state": "idle"}
    assert got[1]["args"][0] == {"mode": "help", "learn_topic": None, "quiz": None}, "a reloaded projector catches up"


def _poll(base: str, sid: str | None = None, body: bytes | None = None) -> str:
    url = f"{base}/socket.io/?EIO=4&transport=polling" + (f"&sid={sid}" if sid else "")
    with urllib.request.urlopen(urllib.request.Request(url, data=body), timeout=5) as response:
        return response.read().decode()


def test_socket_connect_packet_precedes_status_push(built: Any) -> None:
    # socket.io-client replays events that arrive before CONNECT ahead of its "connect" event, where the
    # projector resets to idle; so on the wire the status push must follow CONNECT. The test client
    # can't see this, so speak raw Engine.IO long-polling to a real server.
    from werkzeug.serving import make_server

    server = make_server("127.0.0.1", 0, built[0], threaded=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        sid = json.loads(_poll(base)[1:])["sid"]
        assert _poll(base, sid, b"40") == "OK"
        packets: list[str] = []
        deadline = time.time() + 5
        while time.time() < deadline and not (
            any(p.startswith("40") for p in packets) and any(p.startswith('42["tutor"') for p in packets)
        ):
            packets += _poll(base, sid).split("\x1e")
    finally:
        server.shutdown()
        server.server_close()
    messages = [p for p in packets if p[:2] in ("40", "42")]
    assert [p[:2] for p in messages] == ["40", "42", "42"], packets
    assert json.loads(messages[1][2:]) == ["status", {"state": "idle"}]
    assert json.loads(messages[2][2:])[0] == "tutor"


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


# ---- document selection ---------------------------------------------------------

DOCS = [
    {"source": "a.pdf", "type": "pdf", "chunks": 3},
    {"source": "b.txt", "type": "txt", "chunks": 2},
    {"source": "c.md", "type": "md", "chunks": 1},
]
SELECTION = "/api/documents/selection"


def _with_docs(fakes: Any) -> None:
    fakes.store.sources = [dict(d) for d in DOCS]


def _active(body: dict[str, Any]) -> dict[str, bool]:
    return {d["source"]: d["active"] for d in body["documents"]}


def _pushed(sio: Any, wait: bool = True) -> list[dict[str, Any]]:
    got = collect(sio, ["documents_changed"], timeout=1.0) if wait else sio.get_received()
    return [m["args"][0] for m in got if m["name"] == "documents_changed"]


def _ask(app: Any, socketio: Any, text: str = "q") -> dict[str, Any]:
    sio = socketio.test_client(app)
    assert app.test_client().post("/api/query", json={"text": text}).status_code == 202
    got = collect(sio, ["interaction"])
    return next(m for m in got if m["name"] == "interaction")["args"][0]


def test_select_sources_marks_exactly_those_active(built: Any, fakes: Any, settings: Settings) -> None:
    _with_docs(fakes)
    client = built[0].test_client()
    r = client.put(SELECTION, json={"sources": ["c.md", "a.pdf", "nope.pdf", 7]})
    assert r.status_code == 200
    body = r.get_json()
    assert body == {
        "documents": [
            {"source": "a.pdf", "type": "pdf", "chunks": 3, "active": True},
            {"source": "b.txt", "type": "txt", "chunks": 2, "active": False},
            {"source": "c.md", "type": "md", "chunks": 1, "active": True},
        ],
        "total_chunks": 6,
        "active_count": 2,
        "all_active": False,
    }
    assert client.get("/api/documents").get_json() == body, "PUT returns the GET body"
    assert json.loads(settings.selection_file.read_text()) == {"sources": ["a.pdf", "c.md"]}, "unknown names ignored"


def test_select_none_switches_every_file_off(built: Any, fakes: Any) -> None:
    _with_docs(fakes)
    body = built[0].test_client().put(SELECTION, json={"sources": []}).get_json()
    assert body["active_count"] == 0 and body["all_active"] is False
    assert not any(_active(body).values())


def test_select_all_switches_everything_back_on(built: Any, fakes: Any, settings: Settings) -> None:
    _with_docs(fakes)
    client = built[0].test_client()
    client.put(SELECTION, json={"sources": []})
    body = client.put(SELECTION, json={"all": True}).get_json()
    assert body["all_active"] is True and body["active_count"] == 3 and all(_active(body).values())
    assert json.loads(settings.selection_file.read_text()) == {"all": True}


@pytest.mark.parametrize("sources", ["a.pdf", None, {"a.pdf": True}, 3])
def test_select_non_list_sources_is_400(built: Any, fakes: Any, settings: Settings, sources: Any) -> None:
    _with_docs(fakes)
    r = built[0].test_client().put(SELECTION, json={"sources": sources})
    assert r.status_code == 400 and "error" in r.get_json()
    assert not settings.selection_file.exists()
    assert built[0].config["SAR_ASSISTANT"].active_sources is None


@pytest.mark.parametrize("payload", ["[1]", '"abc"', "null", "{}", '{"all": false}', "not json"])
def test_select_malformed_body_is_400(built: Any, settings: Settings, payload: str) -> None:
    r = built[0].test_client().put(SELECTION, data=payload, content_type="application/json")
    assert r.status_code == 400 and "error" in r.get_json()
    assert not settings.selection_file.exists()


def test_cross_origin_selection_change_blocked(built: Any, fakes: Any, settings: Settings) -> None:
    _with_docs(fakes)
    r = built[0].test_client().put(SELECTION, json={"sources": []}, headers=EVIL)
    assert r.status_code == 403 and not settings.selection_file.exists()


def test_selection_change_pushes_documents_changed(built: Any, fakes: Any) -> None:
    _with_docs(fakes)
    app, socketio = built
    sio = socketio.test_client(app)
    sio.get_received()
    body = app.test_client().put(SELECTION, json={"sources": ["b.txt"]}).get_json()
    assert _pushed(sio) == [body]


def test_rejected_selection_does_not_push(built: Any, fakes: Any) -> None:
    app, socketio = built
    sio = socketio.test_client(app)
    sio.get_received()
    assert app.test_client().put(SELECTION, json={"sources": "x"}).status_code == 400
    time.sleep(0.05)
    assert _pushed(sio, wait=False) == []


def test_upload_pushes_documents_changed(built: Any, ingest_texts: list[str]) -> None:
    app, socketio = built
    sio = socketio.test_client(app)
    sio.get_received()
    client = app.test_client()
    _upload(client)
    pushed = _pushed(sio)
    assert pushed == [client.get("/api/documents").get_json()]
    assert pushed[0]["documents"] == [{"source": "notes.txt", "type": "", "chunks": 2, "active": True}]


def test_upload_keeps_all_active_mode(built: Any, settings: Settings, ingest_texts: list[str]) -> None:
    client = built[0].test_client()
    _upload(client)
    assert client.get("/api/documents").get_json()["all_active"] is True
    assert not settings.selection_file.exists(), "no explicit selection is created"


def test_upload_activates_new_file_when_selection_is_explicit(
    built: Any, fakes: Any, settings: Settings, ingest_texts: list[str]
) -> None:
    _with_docs(fakes)
    client = built[0].test_client()
    client.put(SELECTION, json={"sources": ["a.pdf"]})
    _upload(client, "new.txt")
    body = client.get("/api/documents").get_json()
    assert _active(body) == {"a.pdf": True, "b.txt": False, "c.md": False, "new.txt": True}
    assert body["all_active"] is False and body["active_count"] == 2
    assert json.loads(settings.selection_file.read_text()) == {"sources": ["a.pdf", "new.txt"]}


def test_reupload_switches_a_deselected_file_back_on(built: Any, ingest_texts: list[str]) -> None:
    client = built[0].test_client()
    _upload(client, "notes.txt")
    _upload(client, "other.txt")
    client.put(SELECTION, json={"sources": ["other.txt"]})
    _upload(client, "notes.txt")
    assert _active(client.get("/api/documents").get_json()) == {"notes.txt": True, "other.txt": True}


def test_failed_or_empty_upload_is_not_activated(built: Any, ingest_texts: list[str]) -> None:
    app, _ = built
    client = app.test_client()
    client.put(SELECTION, json={"sources": []})
    ingest_texts[:] = []
    assert _upload(client, "empty.txt").get_json()["files"][0]["chunks"] == 0
    assert _upload(client, "bad.exe").get_json()["files"][0]["error"]
    assert app.config["SAR_ASSISTANT"].active_sources == set()


def test_delete_removes_document_from_selection_and_pushes(built: Any, fakes: Any, settings: Settings) -> None:
    _with_docs(fakes)
    app, socketio = built
    client = app.test_client()
    client.put(SELECTION, json={"sources": ["a.pdf", "b.txt"]})
    sio = socketio.test_client(app)
    sio.get_received()
    assert client.delete("/api/documents/a.pdf").get_json() == {"deleted": 3}
    assert json.loads(settings.selection_file.read_text()) == {"sources": ["b.txt"]}
    pushed = _pushed(sio)
    assert pushed == [client.get("/api/documents").get_json()]
    assert _active(pushed[0]) == {"b.txt": True, "c.md": False}


def test_delete_missing_document_does_not_push(built: Any) -> None:
    app, socketio = built
    sio = socketio.test_client(app)
    sio.get_received()
    assert app.test_client().delete("/api/documents/nope.pdf").status_code == 404
    time.sleep(0.05)
    assert _pushed(sio, wait=False) == []


def test_document_named_selection_can_still_be_deleted(built: Any, fakes: Any) -> None:
    fakes.store.sources = [{"source": "selection", "type": "", "chunks": 1}]
    assert built[0].test_client().delete(SELECTION).get_json() == {"deleted": 1}


def test_status_counts_active_and_total_documents(built: Any, fakes: Any) -> None:
    _with_docs(fakes)
    client = built[0].test_client()
    status = client.get("/api/status").get_json()
    assert (status["active_documents"], status["total_documents"]) == (3, 3)
    client.put(SELECTION, json={"sources": ["b.txt"]})
    status = client.get("/api/status").get_json()
    assert (status["active_documents"], status["total_documents"]) == (1, 3)


def test_selection_survives_restart(settings: Settings, fakes: app_module.Services, templates_dir: Path) -> None:
    _with_docs(fakes)
    first, _ = app_module.create_app(settings, fakes, template_folder=str(templates_dir))
    first.test_client().put(SELECTION, json={"sources": ["b.txt"]})
    second, _ = app_module.create_app(settings, fakes, template_folder=str(templates_dir))
    assert _active(second.test_client().get("/api/documents").get_json()) == {
        "a.pdf": False,
        "b.txt": True,
        "c.md": False,
    }


def test_corrupt_selection_file_means_all_active(
    settings: Settings, fakes: app_module.Services, templates_dir: Path
) -> None:
    _with_docs(fakes)
    settings.selection_file.write_text("{garbage")
    flask_app, _ = app_module.create_app(settings, fakes, template_folder=str(templates_dir))
    body = flask_app.test_client().get("/api/documents").get_json()
    assert body["all_active"] is True and body["active_count"] == 3


def test_query_searches_every_file_when_all_active(built: Any, fakes: Any) -> None:
    fakes.store.query_results = [_chunk("x", "a.pdf", 0.9), _chunk("y", "b.txt", 0.8)]
    item = _ask(*built)
    assert fakes.store.query_sources == [None]
    assert item["sources"] == ["a.pdf", "b.txt"]


def test_query_uses_only_selected_files(built: Any, fakes: Any) -> None:
    _with_docs(fakes)
    fakes.store.query_results = [_chunk("x", "a.pdf", 0.9), _chunk("y", "b.txt", 0.8)]
    built[0].test_client().put(SELECTION, json={"sources": ["c.md", "b.txt"]})
    item = _ask(*built)
    assert fakes.store.query_sources == [["b.txt", "c.md"]], "sorted selection passed to the store"
    assert item["sources"] == ["b.txt"]
    assert built[0].test_client().get("/api/history").get_json()["interactions"][0]["sources"] == ["b.txt"]


def test_query_with_every_file_off_skips_rag(built: Any, fakes: Any) -> None:
    _with_docs(fakes)
    fakes.store.query_results = [_chunk("x", "a.pdf", 0.9)]
    built[0].test_client().put(SELECTION, json={"sources": []})
    item = _ask(*built)
    assert fakes.store.query_calls == [], "store never queried"
    assert item["sources"] == [] and item["error"] is False


def test_search_reads_only_selected_files(built: Any, fakes: Any) -> None:
    _with_docs(fakes)
    fakes.store.query_results = [_chunk("y", "b.txt", 0.8)]
    client = built[0].test_client()
    client.put(SELECTION, json={"sources": ["b.txt"]})
    body = client.get("/api/search?q=runoff").get_json()
    assert fakes.store.query_sources == [["b.txt"]]
    assert [r["source"] for r in body["results"]] == ["b.txt"]


def test_search_with_every_file_off_returns_nothing(built: Any, fakes: Any) -> None:
    _with_docs(fakes)
    client = built[0].test_client()
    client.put(SELECTION, json={"sources": []})
    assert client.get("/api/search?q=runoff").get_json() == {"results": []}
    assert fakes.store.query_sources == []
