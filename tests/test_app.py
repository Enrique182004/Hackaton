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
        "profile",
        "mode",
    } <= set(data)
    assert data["state"] == "idle" and data["audio_available"] is True
    assert (data["profile"], data["mode"]) == ("math", "help")
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

    def ingest_file(path: Any, *, strip_answers: bool = False) -> list[Chunk]:
        return [Chunk(t, str(path), {"source": str(path)}) for t in texts]

    ingestor.ingest_file = ingest_file  # type: ignore[attr-defined]
    ingestor.SUPPORTED_EXTENSIONS = frozenset({".txt"})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "biobench_sar.rag.ingestor", ingestor)
    return texts


def _upload(client: Any, name: str = "notes.txt", content: bytes = b"hello", **kwargs: Any) -> Any:
    return client.post(
        "/api/upload", data={"files[]": [(io.BytesIO(content), name)]}, content_type="multipart/form-data", **kwargs
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
    said = {"display_text": "Look here", "spoken_text": "Look here", "language": "en"}
    # No ink to point at, so nothing to outline either.
    assert speak == {"url": "/tts/fake.mp3", "text": "Look here", "highlight": None, **said}
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
        "profile",
    } <= set(data)
    assert data["projector_size"] == [640, 360]
    assert data["profile"] == "math"
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
    assert [m["name"] for m in got] == ["status", "tutor", "language"]
    assert got[0]["args"][0] == {"state": "idle"}
    assert got[1]["args"][0] == {"mode": "help", "learn_topic": None, "quiz": None, "confirm_stop": False}, (
        "a reloaded projector catches up"
    )
    assert got[2]["args"][0] == {"setting": "auto", "current": "en"}


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
            any(p.startswith("40") for p in packets) and any(p.startswith('42["language"') for p in packets)
        ):
            packets += _poll(base, sid).split("\x1e")
    finally:
        server.shutdown()
        server.server_close()
    messages = [p for p in packets if p[:2] in ("40", "42")]
    assert [p[:2] for p in messages] == ["40", "42", "42", "42"], packets
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
    assert _upload(client, "empty.txt").get_json()["files"][0] == {
        "name": "empty.txt",
        "chunks": 0,
        "error": app_module.NO_TEXT_ERROR,
    }
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


# ---- tutor ----------------------------------------------------------------------

WORKSHEET = b"Fractions quiz\n1. 2 + 2 = ?\nAnswer: 4\n2. 1/2 + 1/4 = ?\n"
TUTOR = "/api/tutor"


def test_upload_without_text_keeps_the_existing_file(built: Any, fakes: Any, ingest_texts: list[str]) -> None:
    client = built[0].test_client()
    _upload(client, "w.txt", WORKSHEET)
    ingest_texts[:] = ["  ", ""]  # a blank rescan of the same file
    body = _upload(client, "w.txt", b"").get_json()
    assert body == {"files": [{"name": "w.txt", "chunks": 0, "error": app_module.NO_TEXT_ERROR}], "total_chunks": 0}
    assert sorted(c.text for c in fakes.store.chunks) == ["a", "b"], "the earlier upload is not wiped"
    assert len(built[0].config["SAR_ASSISTANT"].quiz_bank.problems()) == 2, "its quiz problems too"


def _tutor_pushes(sio: Any, wait: bool = True) -> list[dict[str, Any]]:
    got = collect(sio, ["tutor"], timeout=1.0) if wait else sio.get_received()
    return [m["args"][0] for m in got if m["name"] == "tutor"]


def test_tutor_state_starts_in_help(built: Any) -> None:
    assert built[0].test_client().get(TUTOR).get_json() == {
        "mode": "help",
        "learn_topic": None,
        "quiz": None,
        "confirm_stop": False,
    }


@pytest.mark.parametrize("mode", ["help", "practice", "quiz"])
def test_put_mode(built: Any, mode: str) -> None:
    app, socketio = built
    sio = socketio.test_client(app)
    sio.get_received()
    client = app.test_client()
    r = client.put(f"{TUTOR}/mode", json={"mode": mode})
    assert r.status_code == 200 and r.get_json() == {
        "mode": mode,
        "learn_topic": None,
        "quiz": None,
        "confirm_stop": False,
    }
    assert client.get(TUTOR).get_json()["mode"] == mode
    assert client.get("/api/status").get_json()["mode"] == mode
    assert _tutor_pushes(sio, wait=mode != "help") == ([] if mode == "help" else [r.get_json()])


@pytest.mark.parametrize("payload", ['{"mode": "chat"}', "{}", '{"mode": 3}', "[1]", "null", "not json"])
def test_put_bad_mode_is_400(built: Any, payload: str) -> None:
    client = built[0].test_client()
    r = client.put(f"{TUTOR}/mode", data=payload, content_type="application/json")
    assert r.status_code == 400 and "error" in r.get_json()
    assert client.get(TUTOR).get_json()["mode"] == "help"


def test_cross_origin_tutor_changes_blocked(built: Any) -> None:
    client = built[0].test_client()
    assert client.put(f"{TUTOR}/mode", json={"mode": "quiz"}, headers=EVIL).status_code == 403
    assert client.post(f"{TUTOR}/quiz/start", headers=EVIL).status_code == 403
    assert client.get(TUTOR).get_json()["mode"] == "help"


def test_quiz_start_without_problems_invents_them(built: Any) -> None:
    from biobench_sar.rag.quiz_bank import solve_problem

    body = built[0].test_client().post(f"{TUTOR}/quiz/start").get_json()
    problem = body["quiz"].pop("problem")
    assert body == {
        "mode": "quiz",
        "learn_topic": None,
        "quiz": {"active": True, "number": 1, "total": 10, "score": 0},
        "confirm_stop": False,
    }
    assert solve_problem(problem), "the projector card shows an invented problem the code can grade"


def test_quiz_start_uses_ticked_uploads_and_stop_ends_it(built: Any, ingest_texts: list[str]) -> None:
    app, socketio = built
    client = app.test_client()
    _upload(client, "w.txt", WORKSHEET)
    _upload(client, "other.txt", b"Practice\n1. 9 - 4 = ?\n")
    client.put(SELECTION, json={"sources": ["w.txt"]})
    sio = socketio.test_client(app)
    sio.get_received()
    started = client.post(f"{TUTOR}/quiz/start").get_json()
    assert started["quiz"] == {"active": True, "number": 1, "total": 2, "score": 0, "problem": "2 + 2 = ?"}
    assert _tutor_pushes(sio) == [started]
    stopped = client.post(f"{TUTOR}/quiz/stop").get_json()
    assert stopped == {"mode": "help", "learn_topic": None, "quiz": None, "confirm_stop": False}
    assert _tutor_pushes(sio) == [stopped]


def test_quiz_question_reaches_the_model_and_history(built: Any, fakes: Any, ingest_texts: list[str]) -> None:
    app, socketio = built
    client = app.test_client()
    _upload(client, "w.txt", WORKSHEET)
    sio = socketio.test_client(app)
    sio.get_received()
    item = _ask(app, socketio, "quiz me")
    assert fakes.vlm.kwargs[-1]["task"] == {
        "kind": "quiz",
        "stage": "dictate",
        "number": 1,
        "total": 2,
        "problem": "2 + 2 = ?",
        "answer": "4",
        "next_problem": None,
    }
    assert fakes.vlm.kwargs[-1]["mode"] == "quiz" and fakes.vlm.kwargs[-1]["recent_turns"] == []
    assert (item["mode"], item["task_kind"]) == ("quiz", "quiz")
    assert client.get("/api/history").get_json()["interactions"][0]["task_kind"] == "quiz"
    assert _tutor_pushes(sio)[-1]["quiz"]["problem"] == "2 + 2 = ?"


def test_learn_request_through_the_api(built: Any, fakes: Any) -> None:
    app, socketio = built
    item = _ask(app, socketio, "I want to learn how to solve fractions")
    assert fakes.vlm.kwargs[-1]["task"] == {"kind": "learn", "topic": "solving fractions", "stage": "offer"}
    assert item["task_kind"] == "learn"
    assert app.test_client().get(TUTOR).get_json()["learn_topic"] == "solving fractions"
    _ask(app, socketio, "make one up")
    assert fakes.vlm.kwargs[-1]["task"]["stage"] == "example"
    assert fakes.vlm.kwargs[-1]["recent_turns"][0]["q"] == "I want to learn how to solve fractions"


def test_upload_and_delete_keep_the_quiz_bank_in_sync(built: Any, settings: Settings, ingest_texts: list[str]) -> None:
    app, socketio = built
    client = app.test_client()
    bank = app.config["SAR_ASSISTANT"].quiz_bank
    sio = socketio.test_client(app)
    sio.get_received()
    _upload(client, "w.txt", WORKSHEET)
    assert [(p.problem, p.answer, p.source) for p in bank.problems()] == [
        ("2 + 2 = ?", "4", "w.txt"),
        ("1/2 + 1/4 = ?", None, "w.txt"),
    ]
    assert not any((settings.upload_dir).iterdir()), "parsed before the saved copy is deleted"
    assert json.loads(settings.quiz_bank_file.read_text())["sources"]["w.txt"][0]["problem"] == "2 + 2 = ?"
    _upload(client, "w.txt", b"just notes now")
    assert bank.problems() == [], "an edited file without problems drops its old ones"
    _upload(client, "w.txt", WORKSHEET)
    assert client.delete("/api/documents/w.txt").status_code == 200
    assert bank.problems() == []


def test_startup_drops_quiz_problems_of_missing_files(
    settings: Settings, fakes: app_module.Services, templates_dir: Path
) -> None:
    from biobench_sar.rag.quiz_bank import QuizBank, QuizProblem

    _with_docs(fakes)
    stale = QuizBank(settings.quiz_bank_file)
    stale.replace_source("a.pdf", [QuizProblem("1 + 1", "2", "", 1, "a.pdf")])
    stale.replace_source("gone.md", [QuizProblem("2 + 2", "4", "", 1, "gone.md")])
    flask_app, _ = app_module.create_app(settings, fakes, template_folder=str(templates_dir))
    assert flask_app.config["SAR_ASSISTANT"].quiz_bank.sources() == ["a.pdf"]


def _seed_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "seed" / "math"
    folder.mkdir(parents=True)
    (folder / "quiz_bank.csv").write_text("problem,answer\n3x + 5 = 20,x = 5\n")
    (folder / "tutoring_guide.md").write_text("# Guide\nNever give the answer.\n")
    return folder


def test_seed_never_merges_into_a_file_the_student_uploaded(tmp_path: Path, fakes: Any) -> None:
    from biobench_sar.rag.quiz_bank import QuizBank, QuizProblem

    mine = QuizProblem("2 + 2", "4", "", 1, "quiz_bank.csv")
    bank = QuizBank(tmp_path / "bank.json")
    bank.replace_source("quiz_bank.csv", [mine])
    fakes.store.sources = [{"source": "quiz_bank.csv", "type": "csv", "chunks": 1}]
    marker = tmp_path / ".seeded"
    added = app_module.seed_knowledge(fakes.store, _seed_dir(tmp_path), marker=marker, quiz_bank=bank)
    assert added > 0 and {c.source for c in fakes.store.chunks} == {"tutoring_guide.md"}
    assert bank.problems() == [mine], "the student's own quiz_bank.csv keeps its problems"
    assert marker.read_text().split() == ["math"]


def test_seed_marker_waits_until_every_file_went_in(
    tmp_path: Path, fakes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from biobench_sar.rag import ingestor

    real = ingestor.ingest_file
    broken = {"tutoring_guide.md"}

    def flaky(path: Any, **kwargs: Any) -> Any:
        if Path(path).name in broken:
            raise RuntimeError("embedder not ready")
        return real(path, **kwargs)

    monkeypatch.setattr(ingestor, "ingest_file", flaky)
    marker, folder = tmp_path / ".seeded", _seed_dir(tmp_path)

    def stored() -> list[str]:
        return sorted({c.source for c in fakes.store.chunks})

    app_module.seed_knowledge(fakes.store, folder, marker=marker)
    assert stored() == ["quiz_bank.csv"] and not marker.exists(), "a failed file is tried again next start"
    broken.clear()
    app_module.seed_knowledge(fakes.store, folder, marker=marker)
    assert stored() == ["quiz_bank.csv", "tutoring_guide.md"] and marker.read_text().split() == ["math"]
    assert len(fakes.store.chunks) == 2, "the file that went in the first time is not added twice"


def test_invented_quiz_runs_end_to_end_on_the_mock_model(
    settings: Settings, fakes: Any, templates_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real VLMClient (mock backend) + assistant with no quiz problems: dictate, check, the next one is checked."""
    from biobench_sar.llm.vlm_client import VLMClient

    fakes.vlm = VLMClient("mock", "m", "u", profile="math")
    fakes.vision.frame = np.full((40, 40, 3), 255, np.uint8)
    vision = types.ModuleType("biobench_sar.core.vision_loop")
    vision.encode_jpeg = lambda frame, quality=80: b"jpeg"  # type: ignore[attr-defined]
    ink = [{"x": 1, "y": 1, "w": 5, "h": 5, "cx": 3, "cy": 3}]
    vision.find_ink_regions = lambda frame, **kw: ink  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "biobench_sar.core.vision_loop", vision)
    flask_app, socketio = app_module.create_app(settings, fakes, template_folder=str(templates_dir))
    fakes.audio.on_query = flask_app.config["SAR_HANDLE_QUERY"]
    tutor = flask_app.config["SAR_ASSISTANT"].tutor
    first = _ask(flask_app, socketio, "start a quiz")
    problem = tutor.current().problem
    assert first["animation_state"] == "idle" and tutor.dictated and tutor.invented
    checked = _ask(flask_app, socketio, "check my answer")
    assert checked["animation_state"] == "celebrating" and tutor.quiz_number == 2 and tutor.score == 1
    assert tutor.dictated and tutor.current().problem != problem, "problem two was read out with the praise"
    _ask(flask_app, socketio, "check my answer")
    assert tutor.quiz_number == 3 and tutor.score == 2, "the answer to problem two was checked, not re-dictated"
    hint = _ask(flask_app, socketio, "I'm stuck")
    assert hint["animation_state"] == "pointing" and tutor.quiz_number == 3


def test_notify_documents_is_exposed_for_startup_seeding(built: Any) -> None:
    app, socketio = built
    sio = socketio.test_client(app)
    sio.get_received()
    app.config["SAR_NOTIFY_DOCUMENTS"]()
    assert _pushed(sio) == [app.test_client().get("/api/documents").get_json()]


def test_biology_profile_reports_itself(settings: Settings, fakes: app_module.Services, templates_dir: Path) -> None:
    flask_app, socketio = app_module.create_app(
        dataclasses.replace(settings, profile="biology"), fakes, template_folder=str(templates_dir)
    )
    fakes.audio.on_query = flask_app.config["SAR_HANDLE_QUERY"]
    client = flask_app.test_client()
    assert client.get("/api/config").get_json()["profile"] == "biology"
    assert client.get("/api/status").get_json()["profile"] == "biology"
    item = _ask(flask_app, socketio, "quiz me")
    assert fakes.vlm.kwargs[-1]["task"] is None and item["task_kind"] is None and item["mode"] == "help"


# ---- language ---------------------------------------------------------------------------------------------------


def test_language_api_reads_and_switches(built: Any, fakes: Any) -> None:
    app, socketio = built
    client = app.test_client()
    assert client.get("/api/language").get_json() == {"setting": "auto", "current": "en"}
    sio = socketio.test_client(app)
    sio.get_received()
    r = client.put("/api/language", json={"language": "es"})
    assert r.status_code == 200 and r.get_json() == {"setting": "es", "current": "es"}
    pushed = [m["args"][0] for m in collect(sio, ["language"]) if m["name"] == "language"]
    assert pushed == [{"setting": "es", "current": "es"}], "every open page hears about it"
    assert fakes.audio.languages == ["es"], "Whisper decodes in the forced language"
    assert client.get("/api/language").get_json() == {"setting": "es", "current": "es"}
    assert client.put("/api/language", json={"language": "auto"}).get_json() == {"setting": "auto", "current": "en"}


@pytest.mark.parametrize("body", [{"language": "fr"}, {"language": "ES"}, {"language": None}, {}, ["es"], "es"])
def test_language_api_rejects_anything_else(built: Any, fakes: Any, body: Any) -> None:
    client = built[0].test_client()
    r = client.put("/api/language", json=body)
    assert r.status_code == 400 and "auto, en, es" in r.get_json()["error"]
    assert client.get("/api/language").get_json()["setting"] == "auto" and fakes.audio.languages == []


def test_language_setting_from_the_environment(settings: Settings, fakes: Any, templates_dir: Path) -> None:
    app, _ = app_module.create_app(
        dataclasses.replace(settings, language="es"), fakes, template_folder=str(templates_dir)
    )
    client = app.test_client()
    assert client.get("/api/language").get_json() == {"setting": "es", "current": "es"}
    status, config = client.get("/api/status").get_json(), client.get("/api/config").get_json()
    assert (status["language"], status["current_language"]) == ("es", "es")
    assert (config["language"], config["current_language"]) == ("es", "es")


def test_typed_query_replies_in_the_question_language(built: Any, fakes: Any) -> None:
    app, socketio = built
    item = _ask(app, socketio, "¿Me ayudas con esta ecuación, por favor?")
    assert item["language"] == "es" and fakes.vlm.kwargs[-1]["language"] == "es"
    assert fakes.tts.spoken[-1][1] == "es"
    status = app.test_client().get("/api/status").get_json()
    assert (status["language"], status["current_language"]) == ("auto", "es")
    assert _ask(app, socketio, "Can you help me with this equation?")["language"] == "en"


def test_socket_text_query_replies_in_the_question_language(built: Any, fakes: Any) -> None:
    app, socketio = built
    sio = socketio.test_client(app)
    sio.emit("text_query", {"text": "Dame una pista, por favor"})
    got = collect(sio, ["interaction"])
    assert next(m for m in got if m["name"] == "interaction")["args"][0]["language"] == "es"


def test_answers_carry_display_and_spoken_text(built: Any, fakes: Any) -> None:
    from biobench_sar.schemas import AvatarAction

    @dataclass(frozen=True)
    class Shown(AvatarAction):
        display_text: str | None = None

    fakes.vlm.action = Shown("x plus two equals five", "idle", 1, 2, display_text="x + 2 = 5")
    app, socketio = built
    sio = socketio.test_client(app)
    assert app.test_client().post("/api/query", json={"text": "read it"}).status_code == 202
    got = collect(sio, ["interaction", "speak", "avatar_update"])
    by_name = {m["name"]: m["args"][0] for m in got}
    for name in ("avatar_update", "speak", "interaction"):
        assert by_name[name]["display_text"] == "x + 2 = 5", name
        assert by_name[name]["spoken_text"] == "x plus two equals five", name
    history = app.test_client().get("/api/history").get_json()["interactions"][0]
    assert (history["display_text"], history["language"]) == ("x + 2 = 5", "en")


# ---- glare ------------------------------------------------------------------------------------------------------


def test_status_says_which_computer_it_runs_on(built: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    app, _ = built
    for platform, expected in (("darwin", "mac"), ("win32", "windows"), ("linux", "other")):
        monkeypatch.setattr(sys, "platform", platform)
        assert app.test_client().get("/api/status").get_json()["host_os"] == expected


def test_status_glare_is_null_without_a_camera(built: Any) -> None:
    assert built[0].test_client().get("/api/status").get_json()["glare"] is None


def test_status_reports_glare_on_the_raw_frame(built: Any, fakes: Any) -> None:
    frame = np.full((120, 160, 3), 120, np.uint8)
    frame[20:60, 30:90] = 255  # the bulb's mirror image: a flat clipped patch
    fakes.vision.frame = frame
    glare = built[0].test_client().get("/api/status").get_json()["glare"]
    assert set(glare) == {"fraction", "severe", "hotspots", "size"}
    assert glare["severe"] is True and glare["size"] == [160, 120]
    x, y, w, h = glare["hotspots"][0]
    assert x <= 30 and y <= 20 and x + w >= 89 and y + h >= 59, "camera pixels of the raw frame"
    assert 0.1 < glare["fraction"] < 0.2


def test_status_glare_is_measured_at_most_every_two_seconds(
    built: Any, fakes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from biobench_sar.core import glare

    calls: list[int] = []
    real = glare.measure_glare

    def counted(frame: Any, **kw: Any) -> Any:
        calls.append(1)
        return real(frame, **kw)

    monkeypatch.setattr(glare, "measure_glare", counted)
    fakes.vision.frame = np.full((40, 40, 3), 120, np.uint8)
    client = built[0].test_client()
    for _ in range(5):
        assert client.get("/api/status").get_json()["glare"]["severe"] is False
    assert calls == [1], "cached between status polls"


def test_status_survives_a_glare_failure(built: Any, fakes: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from biobench_sar.core import glare

    def broken(frame: Any, **kw: Any) -> None:
        raise RuntimeError("bad frame")

    monkeypatch.setattr(glare, "measure_glare", broken)
    fakes.vision.frame = np.full((40, 40, 3), 120, np.uint8)
    r = built[0].test_client().get("/api/status")
    assert r.status_code == 200 and r.get_json()["glare"] is None


# ---- LAN mode off (the default): nothing about phones changes the laptop's server ----------------------
def test_lan_mode_off_needs_no_pin_and_marks_nothing(built: Any) -> None:
    from biobench_sar import lan

    app, socketio = built
    client = app.test_client()
    for path in ("/", "/dashboard", "/api/status"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert lan.LAN_MODE_HEADER not in response.headers, path
    assert b"lan=False" in client.get("/dashboard").data
    assert client.get("/sw.js").status_code == 404, "nothing can register a service worker outside LAN mode"
    assert client.get("/socket.io/?EIO=4&transport=polling").get_data().startswith(b"0{")
    assert socketio.test_client(app, flask_test_client=client).is_connected()
    assert app.config["SAR_JOIN_PIN"] is None


def test_api_responses_are_never_cached(built: Any) -> None:
    app, _ = built
    client = app.test_client()
    for response in (
        client.get("/api/status"),
        client.get("/api/documents"),
        client.get("/api/lan"),
        client.post("/api/query", json={}),
        client.get("/api/missing"),
    ):
        assert response.headers["Cache-Control"] == "no-store", response.request.path


def test_ui_rects_socket_event_is_forwarded(built: Any) -> None:
    app, socketio = built
    seen: list[Any] = []
    app.config["SAR_ASSISTANT"].set_ui_rects = seen.append
    socketio.test_client(app).emit("ui_rects", {"rects": [[0, 0, 200, 300]]})
    assert seen == [[[0.0, 0.0, 200.0, 300.0]]]
