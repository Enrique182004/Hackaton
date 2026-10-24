"""Phone/LAN mode wired into the real app (contract point 7, docs/MOBILE.md): the PIN gate, /join, /api/lan,
the Socket.IO gate, the headers, the startup banner, and the projector's "ui_rects" forwarding (point 3)."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

import pytest
from flask import Flask
from flask.testing import FlaskClient
from flask_socketio import SocketIO

from biobench_sar import app as app_module
from biobench_sar import lan
from biobench_sar.config import Settings

PHONE = "192.168.1.30"
OTHER_PHONE = "192.168.1.31"
LAN_IP = "192.168.1.20"
LAN_HOST = f"{LAN_IP}:5050"
LOCAL_HOST = "127.0.0.1:5050"
HTML = "text/html,application/xhtml+xml,*/*;q=0.8"
POLLING = "/socket.io/?EIO=4&transport=polling"


def make_app(
    settings: Settings, fakes: app_module.Services, templates_dir: Path, **changes: Any
) -> tuple[Flask, SocketIO]:
    flask_app, socketio = app_module.create_app(
        dataclasses.replace(settings, port=5050, **changes), fakes, template_folder=str(templates_dir)
    )
    flask_app.testing = True
    return flask_app, socketio


@pytest.fixture(autouse=True)
def known_addresses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "local_ip_addresses", lambda: [LAN_IP])


@pytest.fixture
def lan_app(settings: Settings, fakes: app_module.Services, templates_dir: Path) -> tuple[Flask, SocketIO]:
    return make_app(settings, fakes, templates_dir, lan=True, host="0.0.0.0")


def gate_of(flask_app: Flask) -> lan.JoinPin:
    gate = flask_app.config["SAR_JOIN_PIN"]
    assert isinstance(gate, lan.JoinPin)
    return gate


class DeviceClient(FlaskClient):
    """A test client whose requests all go to one Host (a Host header in a call still wins)."""

    host = LOCAL_HOST

    def open(self, *args: Any, **kwargs: Any) -> Any:
        if not args or isinstance(args[0], str):
            kwargs.setdefault("base_url", f"http://{self.host}")
        return super().open(*args, **kwargs)


def device(flask_app: Flask, *, addr: str, host: str) -> DeviceClient:
    client = DeviceClient(flask_app, flask_app.response_class, use_cookies=True)
    client.host = host
    client.environ_base["REMOTE_ADDR"] = addr
    return client


def phone(flask_app: Flask, addr: str = PHONE) -> DeviceClient:
    """A device on the Wi-Fi, opening the laptop by its LAN address as the QR code does."""
    return device(flask_app, addr=addr, host=LAN_HOST)


def laptop(flask_app: Flask) -> DeviceClient:
    return device(flask_app, addr="127.0.0.1", host=LOCAL_HOST)


def wrong(pin: str) -> str:
    return f"{(int(pin) + 1) % 10**6:06d}"


def join(client: FlaskClient, pin: str, **headers: str) -> Any:
    return client.post("/join", data={"pin": pin}, headers=headers)


def phone_socket(socketio: SocketIO, flask_app: Flask, client: FlaskClient) -> Any:
    return socketio.test_client(flask_app, flask_test_client=client, headers={"Host": LAN_HOST})


# ---- the laptop ------------------------------------------------------------------------------------
def test_the_laptop_passes_without_a_pin(lan_app: tuple[Flask, SocketIO]) -> None:
    flask_app, socketio = lan_app
    client = laptop(flask_app)
    assert client.get("/api/status").status_code == 200
    assert client.get("/dashboard").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/api/status", headers={"Host": "localhost:5050"}).status_code == 200
    assert client.get(POLLING).get_data().startswith(b"0{"), "the Engine.IO OPEN packet"
    sio = socketio.test_client(flask_app, flask_test_client=client)
    assert sio.is_connected()
    assert len(flask_app.config["SAR_PHONES"]) == 0, "the laptop is not a phone"


def test_a_loopback_peer_with_a_rebindable_host_is_refused(lan_app: tuple[Flask, SocketIO]) -> None:
    flask_app, _ = lan_app
    client = laptop(flask_app)
    assert client.get("/api/status", headers={"Host": "evil.example"}).status_code == 400
    assert client.get("/join", headers={"Host": "evil.example:5050"}).status_code == 400
    assert client.get(POLLING, headers={"Host": "evil.example"}).status_code == 400


def test_configured_hostnames_pass_the_host_check(
    settings: Settings, fakes: app_module.Services, templates_dir: Path
) -> None:
    flask_app, _ = make_app(
        settings, fakes, templates_dir, lan=True, host="0.0.0.0", lan_hostnames=("my-laptop.local",)
    )
    assert laptop(flask_app).get("/api/status", headers={"Host": "my-laptop.local:5050"}).status_code == 200
    named = phone(flask_app).get("/api/status", headers={"Host": "my-laptop.local:5050"})
    assert named.status_code == 401, "a known name is a valid host; the phone still needs the PIN"


# ---- a phone that hasn't joined ----------------------------------------------------------------------
def test_a_phone_without_the_cookie_is_gated(lan_app: tuple[Flask, SocketIO]) -> None:
    flask_app, _ = lan_app
    client = phone(flask_app)
    status = client.get("/api/status")
    assert status.status_code == 401
    assert status.get_json() == {"error": "join with the PIN first"}
    for path in ("/dashboard", "/", "/precharge", "/tts/fake.mp3", "/calibration.png"):
        page = client.get(path)
        assert page.status_code == 302, path
        assert page.headers["Location"].endswith("/join"), path
    assert client.post("/api/query", json={"text": "hi"}).status_code == 401
    assert client.put("/api/tutor/mode", json={"mode": "quiz"}).status_code == 401
    assert client.post("/dashboard").status_code == 401, "only GET pages redirect"
    assert flask_app.config["SAR_ASSISTANT"].tutor.mode != "quiz"


def test_a_phone_without_the_cookie_gets_the_join_form_and_static_files(lan_app: tuple[Flask, SocketIO]) -> None:
    flask_app, _ = lan_app
    client = phone(flask_app)
    form = client.get("/join")
    assert form.status_code == 200
    assert b"join error=False" in form.data and b"lan=True" in form.data
    assert client.get("/static/manifest.webmanifest").status_code == 200
    assert client.get("/sw.js").status_code == 200


def test_a_phone_without_the_cookie_cannot_open_a_socket(lan_app: tuple[Flask, SocketIO]) -> None:
    flask_app, socketio = lan_app
    client = phone(flask_app)
    refused = client.get(POLLING)
    assert refused.status_code == 401
    assert not refused.get_data().startswith(b"0{"), "no Engine.IO session was opened"
    assert refused.headers["Cache-Control"] == "no-store"
    sio = phone_socket(socketio, flask_app, client)
    assert not sio.is_connected(), "the connect handler refuses it too"
    assert len(flask_app.config["SAR_PHONES"]) == 0


# ---- joining ---------------------------------------------------------------------------------------
def test_the_right_pin_sets_the_cookie_and_lets_the_phone_in(lan_app: tuple[Flask, SocketIO]) -> None:
    flask_app, socketio = lan_app
    client = phone(flask_app)
    joined = join(client, gate_of(flask_app).pin)
    assert joined.status_code == 303
    assert joined.headers["Location"].endswith("/dashboard")
    cookie = joined.headers["Set-Cookie"]
    assert cookie.startswith(f"{lan.COOKIE_NAME}=")
    assert "HttpOnly" in cookie and "SameSite=Lax" in cookie and "Path=/" in cookie
    assert f"Max-Age={12 * 60 * 60}" in cookie
    assert "Secure" not in cookie, "the classroom LAN is plain HTTP"
    assert client.get("/dashboard").status_code == 200
    assert client.get("/api/status").status_code == 200
    assert client.get(POLLING).get_data().startswith(b"0{")
    sio = phone_socket(socketio, flask_app, client)
    assert sio.is_connected()
    assert len(flask_app.config["SAR_PHONES"]) == 1
    sio.disconnect()
    assert len(flask_app.config["SAR_PHONES"]) == 0, "a phone that leaves is no longer counted"


def test_the_pin_is_accepted_as_people_type_it(lan_app: tuple[Flask, SocketIO]) -> None:
    flask_app, _ = lan_app
    pin = gate_of(flask_app).pin
    assert join(phone(flask_app), f" {pin[:3]} {pin[3:]} ").status_code == 303


def test_a_joined_phone_going_back_to_join_lands_on_the_dashboard(lan_app: tuple[Flask, SocketIO]) -> None:
    flask_app, _ = lan_app
    client = phone(flask_app)
    join(client, gate_of(flask_app).pin)
    again = client.get("/join")
    assert again.status_code == 303 and again.headers["Location"].endswith("/dashboard")


def test_the_qr_link_joins_only_on_a_page_load(lan_app: tuple[Flask, SocketIO]) -> None:
    flask_app, _ = lan_app
    gate = gate_of(flask_app)
    client = phone(flask_app)
    for _ in range(3 * lan.MAX_FAILURES):
        img = client.get(f"/join?pin={wrong(gate.pin)}", headers={"Accept": "image/avif,image/webp,*/*;q=0.8"})
        assert img.status_code == 200
        assert b"error=False" in img.data
    assert gate.locked_for(PHONE) == 0.0, "an <img> tag never counts a guess"
    passive = client.get(f"/join?pin={gate.pin}", headers={"Accept": "image/*"})
    assert passive.status_code == 200 and "Set-Cookie" not in passive.headers
    scanned = client.get(f"/join?pin={gate.pin}", headers={"Accept": HTML})
    assert scanned.status_code == 303
    assert lan.COOKIE_NAME in scanned.headers["Set-Cookie"]
    assert client.get("/dashboard").status_code == 200


def test_a_wrong_pin_shows_the_error(lan_app: tuple[Flask, SocketIO]) -> None:
    flask_app, _ = lan_app
    client = phone(flask_app)
    failed = join(client, wrong(gate_of(flask_app).pin))
    assert failed.status_code == 200
    assert b"join error=True" in failed.data
    assert b"That PIN didn&#39;t work" in failed.data or b"That PIN didn't work" in failed.data
    assert "Set-Cookie" not in failed.headers
    assert client.get("/api/status").status_code == 401


def test_a_blank_pin_is_an_error_but_not_a_guess(lan_app: tuple[Flask, SocketIO]) -> None:
    flask_app, _ = lan_app
    gate = gate_of(flask_app)
    client = phone(flask_app)
    for _ in range(2 * lan.MAX_FAILURES):
        assert b"error=True" in join(client, "  ").data
    assert client.post("/join").status_code == 200
    assert gate.locked_for(PHONE) == 0.0
    assert join(client, gate.pin).status_code == 303


def test_too_many_wrong_pins_lock_that_phone_and_not_the_next(lan_app: tuple[Flask, SocketIO]) -> None:
    flask_app, _ = lan_app
    gate = gate_of(flask_app)
    client = phone(flask_app)
    for _ in range(lan.MAX_FAILURES - 1):
        assert b"locked=0" in join(client, wrong(gate.pin)).data
    locked = join(client, wrong(gate.pin))
    assert b"error=True" in locked.data
    assert b"locked=60" in locked.data
    assert b"Too many wrong PINs. Try again in 1 minute." in locked.data
    still = join(client, gate.pin)
    assert still.status_code == 200 and "Set-Cookie" not in still.headers, "locked out, even with the right PIN"
    assert b"Too many wrong PINs" in client.get("/join").data, "the form says so on a plain reload too"
    assert join(phone(flask_app, OTHER_PHONE), gate.pin).status_code == 303, "another phone still joins"


def test_the_join_page_is_never_cached_and_keeps_its_referrer_same_origin(lan_app: tuple[Flask, SocketIO]) -> None:
    flask_app, _ = lan_app
    client = phone(flask_app)
    for response in (client.get("/join"), join(client, wrong(gate_of(flask_app).pin)), join(client, "")):
        assert response.headers["Cache-Control"] == "no-store"
        assert response.headers["Referrer-Policy"] == "same-origin"
        assert response.headers[lan.LAN_MODE_HEADER] == "1"
    joined = join(client, gate_of(flask_app).pin)
    assert joined.headers["Referrer-Policy"] == "same-origin" and joined.headers["Cache-Control"] == "no-store"


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": f"http://{LAN_HOST}"},
        {"Referer": f"http://{LAN_HOST}/join"},
        {"Origin": f"http://{LAN_HOST}", "Referer": f"http://{LAN_HOST}/join?pin=000000"},
    ],
)
def test_the_same_origin_join_form_is_not_blocked_as_cross_origin(
    lan_app: tuple[Flask, SocketIO], headers: dict[str, str]
) -> None:
    flask_app, _ = lan_app
    client = phone(flask_app)
    joined = join(client, gate_of(flask_app).pin, **headers)
    assert joined.status_code == 303, joined.get_data(as_text=True)
    assert client.get("/api/status").status_code == 200


@pytest.mark.parametrize("origin", ["http://evil.example", "null", f"http://{LAN_IP}:9999"])
def test_a_join_post_from_another_origin_is_blocked(lan_app: tuple[Flask, SocketIO], origin: str) -> None:
    flask_app, _ = lan_app
    gate = gate_of(flask_app)
    blocked = join(phone(flask_app), gate.pin, Origin=origin)
    assert blocked.status_code == 403
    assert "Set-Cookie" not in blocked.headers
    assert gate.locked_for(PHONE) == 0.0


def test_the_laptop_and_lan_mode_off_never_see_the_join_form(lan_app: tuple[Flask, SocketIO], built: Any) -> None:
    flask_app, _ = lan_app
    local = laptop(flask_app).get("/join")
    assert local.status_code == 303 and local.headers["Location"].endswith("/dashboard")
    off, _ = built
    plain = off.test_client().get("/join")
    assert plain.status_code == 303 and plain.headers["Location"].endswith("/dashboard")


# ---- the language of the join page -------------------------------------------------------------------
def test_the_join_page_speaks_the_phone_language_in_auto(lan_app: tuple[Flask, SocketIO]) -> None:
    flask_app, _ = lan_app
    client = phone(flask_app)
    assert b"lang=es" in client.get("/join", headers={"Accept-Language": "es-MX,es;q=0.9"}).data
    assert b"lang=en" in client.get("/join", headers={"Accept-Language": "en-US"}).data
    failed = join(client, wrong(gate_of(flask_app).pin), **{"Accept-Language": "es"})
    assert "Ese PIN no funcionó" in failed.get_data(as_text=True)


def test_a_fixed_language_setting_wins_on_the_join_page(lan_app: tuple[Flask, SocketIO]) -> None:
    flask_app, _ = lan_app
    flask_app.config["SAR_ASSISTANT"].set_language("es")
    page = phone(flask_app).get("/join", headers={"Accept-Language": "en-US"})
    assert b"lang=es" in page.data


def test_the_join_form_still_works_without_its_template(
    settings: Settings, fakes: app_module.Services, templates_dir: Path
) -> None:
    (templates_dir / "join.html").unlink()
    flask_app, _ = make_app(settings, fakes, templates_dir, lan=True, host="0.0.0.0")
    client = phone(flask_app)
    form = client.get("/join").get_data(as_text=True)
    assert '<form method="post" action="/join">' in form and 'name="pin"' in form
    assert 'inputmode="numeric"' in form and 'autocomplete="one-time-code"' in form
    failed = join(client, wrong(gate_of(flask_app).pin), **{"Accept-Language": "es"}).get_data(as_text=True)
    assert 'lang="es"' in failed and "Ese PIN no funcionó" in failed
    assert join(client, gate_of(flask_app).pin).status_code == 303


# ---- the laptop's card: /api/lan ---------------------------------------------------------------------
def test_api_lan_gives_the_laptop_the_pin_and_the_phone_urls(lan_app: tuple[Flask, SocketIO]) -> None:
    flask_app, socketio = lan_app
    gate = gate_of(flask_app)
    body = laptop(flask_app).get("/api/lan").get_json()
    assert body == {
        "enabled": True,
        "urls": [f"http://{LAN_IP}:5050/join?pin={gate.pin}"],
        "pin": gate.pin,
        "phones": 0,
        "locked_for": 0,
    }
    client = phone(flask_app)
    join(client, gate.pin)
    phone_socket(socketio, flask_app, client)
    assert laptop(flask_app).get("/api/lan").get_json()["phones"] == 1


def test_a_phone_never_sees_the_pin(lan_app: tuple[Flask, SocketIO]) -> None:
    flask_app, _ = lan_app
    client = phone(flask_app)
    assert client.get("/api/lan").status_code == 401
    join(client, gate_of(flask_app).pin)
    refused = client.get("/api/lan")
    assert refused.status_code == 403
    assert gate_of(flask_app).pin not in refused.get_data(as_text=True)
    assert client.post("/api/lan/rotate").status_code == 403
    assert client.post("/api/lan/disconnect").status_code == 403
    assert client.get("/api/status").status_code == 200, "a refused rotate signed nobody out"


def test_api_lan_with_lan_mode_off(built: Any) -> None:
    flask_app, _ = built
    client = flask_app.test_client()
    assert client.get("/api/lan").get_json() == {
        "enabled": False,
        "urls": [],
        "pin": None,
        "phones": 0,
        "locked_for": 0,
    }
    assert client.post("/api/lan/rotate").status_code == 409
    assert client.post("/api/lan/disconnect").status_code == 409
    assert flask_app.config["SAR_JOIN_PIN"] is None


def test_a_loopback_bind_with_sar_lan_lists_no_phone_urls(
    settings: Settings, fakes: app_module.Services, templates_dir: Path
) -> None:
    flask_app, _ = make_app(settings, fakes, templates_dir, lan=True, host="127.0.0.1")
    body = laptop(flask_app).get("/api/lan").get_json()
    assert body["enabled"] is True and body["urls"] == [], "phones can't reach a loopback bind"


def test_api_lan_reports_the_server_wide_lockout(lan_app: tuple[Flask, SocketIO]) -> None:
    flask_app, _ = lan_app
    gate = gate_of(flask_app)
    for n in range(lan.GLOBAL_MAX_FAILURES):
        gate.verify(wrong(gate.pin), f"10.0.{n // 200}.{n % 200 + 1}")
    assert laptop(flask_app).get("/api/lan").get_json()["locked_for"] == math.ceil(gate.server_locked_for()) > 0


@pytest.mark.parametrize("action", ["/api/lan/rotate", "/api/lan/disconnect"])
def test_rotate_and_disconnect_sign_every_phone_out(lan_app: tuple[Flask, SocketIO], action: str) -> None:
    flask_app, socketio = lan_app
    gate = gate_of(flask_app)
    old_pin = gate.pin
    client = phone(flask_app)
    assert join(client, old_pin).status_code == 303
    sio = phone_socket(socketio, flask_app, client)
    assert sio.is_connected()
    assert client.get(POLLING).get_data().startswith(b"0{")
    projector = socketio.test_client(flask_app, flask_test_client=laptop(flask_app))
    assert projector.is_connected()

    response = laptop(flask_app).post(action)
    assert response.status_code == 200
    body = response.get_json()
    assert body["enabled"] is True and body["phones"] == 0
    assert (body["pin"] != old_pin) is (action == "/api/lan/rotate")
    assert body["pin"] == gate.pin
    assert not sio.is_connected(), "the phone's live socket was closed"
    assert projector.is_connected(), "the laptop's own pages stay connected"
    assert client.get("/api/status").status_code == 401
    assert client.get("/dashboard").status_code == 302
    assert client.get(POLLING).status_code == 401
    assert not phone_socket(socketio, flask_app, client).is_connected()
    assert join(client, gate.pin).status_code == 303, "the current PIN joins again"


def test_rotate_refuses_the_old_pin(lan_app: tuple[Flask, SocketIO]) -> None:
    flask_app, _ = lan_app
    old_pin = gate_of(flask_app).pin
    laptop(flask_app).post("/api/lan/rotate")
    assert join(phone(flask_app), old_pin).status_code == 200


def test_rotate_from_the_laptop_page_passes_the_cross_origin_guard(lan_app: tuple[Flask, SocketIO]) -> None:
    flask_app, _ = lan_app
    client = laptop(flask_app)
    assert client.post("/api/lan/rotate", headers={"Origin": f"http://{LOCAL_HOST}"}).status_code == 200
    assert client.post("/api/lan/rotate", headers={"Origin": "http://evil.example"}).status_code == 403


# ---- headers -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("lan_on", [True, False])
def test_every_api_response_is_no_store(
    settings: Settings, fakes: app_module.Services, templates_dir: Path, lan_on: bool
) -> None:
    flask_app, _ = make_app(settings, fakes, templates_dir, lan=lan_on, host="0.0.0.0" if lan_on else "127.0.0.1")
    client = laptop(flask_app)
    for path in ("/api/status", "/api/lan", "/api/config", "/api/tutor", "/api/history", "/api/nope"):
        response = client.get(path)
        assert response.headers["Cache-Control"] == "no-store", path
        assert (lan.LAN_MODE_HEADER in response.headers) is lan_on, path
    assert client.post("/api/query", json={}).headers["Cache-Control"] == "no-store", "errors too"
    page = client.get("/dashboard")
    assert page.headers.get("Cache-Control") != "no-store", "pages keep their normal caching"
    if lan_on:
        assert phone(flask_app).get("/api/status").headers["Cache-Control"] == "no-store", "the 401 too"


# ---- the service worker ------------------------------------------------------------------------------
def test_the_service_worker_is_404_outside_lan_mode(built: Any) -> None:
    flask_app, _ = built
    assert flask_app.test_client().get("/sw.js").status_code == 404


def test_the_service_worker_in_lan_mode(lan_app: tuple[Flask, SocketIO]) -> None:
    flask_app, _ = lan_app
    worker = laptop(flask_app).get("/sw.js")
    assert worker.status_code == 200
    assert worker.mimetype == "text/javascript"
    assert worker.headers["Service-Worker-Allowed"] == "/"
    assert worker.headers["Cache-Control"] == "no-cache"
    assert worker.headers[lan.LAN_MODE_HEADER] == "1"
    assert b"X-SAR-LAN" in worker.data or lan.LAN_MODE_HEADER.encode() in worker.data


# ---- a network bind without SAR_LAN ------------------------------------------------------------------
@pytest.mark.parametrize("host", ["0.0.0.0", "", "::", LAN_IP])
def test_a_network_bind_without_sar_lan_still_gates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fakes: app_module.Services,
    templates_dir: Path,
    host: str,
) -> None:
    monkeypatch.delenv("SAR_LAN", raising=False)
    monkeypatch.setenv("SAR_HOST", host)
    env_settings = Settings()
    assert env_settings.lan is True
    settings = dataclasses.replace(
        env_settings, upload_dir=tmp_path / "up", tts_dir=tmp_path / "tts", chroma_dir=tmp_path / "chroma"
    )
    flask_app, socketio = make_app(settings, fakes, templates_dir)
    client = phone(flask_app)
    assert client.get("/api/status").status_code == 401
    assert client.get("/dashboard").status_code == 302
    assert client.get(POLLING).status_code == 401
    assert not phone_socket(socketio, flask_app, client).is_connected()
    assert laptop(flask_app).get("/api/status").status_code == 200


def test_sar_lan_from_the_environment_binds_every_interface_and_gates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fakes: app_module.Services, templates_dir: Path
) -> None:
    monkeypatch.setenv("SAR_LAN", "1")
    env_settings = Settings()
    assert (env_settings.lan, env_settings.host) == (True, "0.0.0.0")
    settings = dataclasses.replace(
        env_settings, upload_dir=tmp_path / "up", tts_dir=tmp_path / "tts", chroma_dir=tmp_path / "chroma"
    )
    flask_app, _ = make_app(settings, fakes, templates_dir)
    assert phone(flask_app).get("/api/status").status_code == 401


# ---- templates ---------------------------------------------------------------------------------------
def test_pages_receive_lan_enabled(lan_app: tuple[Flask, SocketIO], built: Any) -> None:
    flask_app, _ = lan_app
    client = laptop(flask_app)
    assert b"lan=True" in client.get("/dashboard").data
    assert b"lan=True" in client.get("/").data
    off, _ = built
    assert b"lan=False" in off.test_client().get("/dashboard").data
    assert b"lan=False" in off.test_client().get("/").data


# ---- Socket.IO handlers check the token again --------------------------------------------------------
def test_socket_handlers_drop_a_phone_whose_token_was_revoked(
    lan_app: tuple[Flask, SocketIO], fakes: app_module.Services
) -> None:
    flask_app, socketio = lan_app
    client = phone(flask_app)
    join(client, gate_of(flask_app).pin)
    sio = phone_socket(socketio, flask_app, client)
    assert sio.is_connected()
    gate_of(flask_app).revoke_all()  # without closing the socket, as a race with the WSGI gate would leave it
    sio.emit("text_query", {"text": "what is x?"})
    assert not sio.is_connected()
    assert fakes.vlm.calls == []

    sio = phone_socket(socketio, flask_app, client)
    assert not sio.is_connected()


@pytest.mark.parametrize("event", ["trigger_listen", "ui_rects"])
def test_other_socket_handlers_check_the_token_too(
    lan_app: tuple[Flask, SocketIO], fakes: app_module.Services, event: str
) -> None:
    flask_app, socketio = lan_app
    seen: list[Any] = []
    flask_app.config["SAR_ASSISTANT"].set_ui_rects = seen.append
    client = phone(flask_app)
    join(client, gate_of(flask_app).pin)
    sio = phone_socket(socketio, flask_app, client)
    gate_of(flask_app).revoke_all()
    sio.emit(event, {"rects": [[1, 2, 3, 4]]})
    assert not sio.is_connected()
    assert fakes.audio.triggered == 0 and seen == []


# ---- ui_rects (contract point 3) ---------------------------------------------------------------------
def test_ui_rects_reach_the_assistant(built: Any) -> None:
    flask_app, socketio = built
    seen: list[Any] = []
    flask_app.config["SAR_ASSISTANT"].set_ui_rects = seen.append
    sio = socketio.test_client(flask_app)
    sio.emit("ui_rects", {"rects": [[10, 20, 300, 120], [900.5, 40, 200, 80.25]]})
    assert seen == [[[10.0, 20.0, 300.0, 120.0], [900.5, 40.0, 200.0, 80.25]]]
    sio.emit("ui_rects", {"rects": []})
    assert seen[-1] == [], "an empty list clears the mask"


def test_ui_rects_drops_malformed_rects_and_ignores_malformed_payloads(built: Any) -> None:
    flask_app, socketio = built
    seen: list[Any] = []
    flask_app.config["SAR_ASSISTANT"].set_ui_rects = seen.append
    sio = socketio.test_client(flask_app)
    sio.emit("ui_rects", {"rects": [[1, 2, 3, 4], "box", [1, 2, 3], [1, 2, 0, 4], [True, 2, 3, 4], [1, "2", 3, 4]]})
    assert seen == [[[1.0, 2.0, 3.0, 4.0]]]
    for payload in (None, "rects", [[1, 2, 3, 4]], {"rects": "all"}, {"other": []}):
        sio.emit("ui_rects", payload)
    assert len(seen) == 1, "nothing that isn't a {rects: [...]} payload reaches the assistant"
    assert sio.is_connected()


def test_ui_rects_without_set_ui_rects_is_harmless(built: Any) -> None:
    flask_app, socketio = built
    assistant = flask_app.config["SAR_ASSISTANT"]
    if hasattr(assistant, "set_ui_rects"):
        pytest.skip("the assistant implements set_ui_rects; covered by the other tests")
    sio = socketio.test_client(flask_app)
    sio.emit("ui_rects", {"rects": [[1, 2, 3, 4]]})
    assert sio.is_connected()


def test_a_failing_set_ui_rects_does_not_drop_the_projector(built: Any) -> None:
    flask_app, socketio = built

    def boom(_rects: Any) -> None:
        raise RuntimeError("bad rects")

    flask_app.config["SAR_ASSISTANT"].set_ui_rects = boom
    sio = socketio.test_client(flask_app)
    sio.emit("ui_rects", {"rects": [[1, 2, 3, 4]]})
    assert sio.is_connected()


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"rects": [[0, 0, 10, 10]]}, [[0.0, 0.0, 10.0, 10.0]]),
        ({"rects": [(0, 0, 10, 10)]}, [[0.0, 0.0, 10.0, 10.0]]),
        ({"rects": [[-5, -5, 10, 10]]}, [[-5.0, -5.0, 10.0, 10.0]]),
        ({"rects": [[float("nan"), 0, 10, 10], [0, 0, float("inf"), 10]]}, []),
        ({"rects": [[0, 0, -1, 10]]}, []),
        ({"rects": []}, []),
        ({"rects": None}, None),
        ([], None),
        ("x", None),
    ],
)
def test_parse_ui_rects(data: Any, expected: list[list[float]] | None) -> None:
    assert app_module.parse_ui_rects(data) == expected


def test_parse_ui_rects_is_bounded() -> None:
    rects = app_module.parse_ui_rects({"rects": [[i, 0, 1, 1] for i in range(1000)]})
    assert rects is not None and len(rects) == app_module.MAX_UI_RECTS


# ---- addresses and the startup banner ----------------------------------------------------------------
@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", []),
        ("localhost", []),
        ("::1", []),
        ("0.0.0.0", [LAN_IP]),
        ("", [LAN_IP]),
        ("::", [LAN_IP]),
        ("10.0.0.7", ["10.0.0.7"]),
        ("my-laptop.local", [LAN_IP]),
    ],
)
def test_phone_addresses(host: str, expected: list[str]) -> None:
    assert app_module.phone_addresses(host) == expected


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", "http://127.0.0.1:5050"),
        ("0.0.0.0", "http://127.0.0.1:5050"),
        ("", "http://127.0.0.1:5050"),
        ("::", "http://[::1]:5050"),
        ("::1", "http://[::1]:5050"),
        ("10.0.0.7", "http://10.0.0.7:5050"),
        ("localhost", "http://localhost:5050"),
    ],
)
def test_laptop_url(host: str, expected: str) -> None:
    assert app_module.laptop_url(host, 5050) == expected


def test_startup_banner_in_lan_mode_prints_the_phone_url_and_pin(settings: Settings) -> None:
    lan_settings = dataclasses.replace(settings, lan=True, host="0.0.0.0", port=5050)
    banner = app_module.startup_banner(lan_settings, "042917", [LAN_IP, "10.0.0.7"])
    assert "Dashboard  http://127.0.0.1:5050/dashboard" in banner, "0.0.0.0 can't be clicked"
    assert f"Phones     http://{LAN_IP}:5050/join   PIN 042 917" in banner
    assert "http://10.0.0.7:5050/join   PIN 042 917" in banner
    assert "?pin=" not in banner
    assert "firewall" in banner and "VPN" in banner


def test_startup_banner_without_a_network(settings: Settings) -> None:
    lan_settings = dataclasses.replace(settings, lan=True, host="0.0.0.0", port=5050)
    assert "isn't on a network" in app_module.startup_banner(lan_settings, "042917", [])
    loopback = dataclasses.replace(settings, lan=True, host="127.0.0.1", port=5050)
    assert "can't reach 127.0.0.1" in app_module.startup_banner(loopback, "042917", [])


def test_startup_banner_outside_lan_mode_has_no_phone_lines(settings: Settings) -> None:
    banner = app_module.startup_banner(dataclasses.replace(settings, port=5050), None, [])
    assert "Projector  http://127.0.0.1:5050/" in banner
    assert "Phones" not in banner and "PIN" not in banner


def test_main_logs_the_pin_in_lan_mode(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    fakes: app_module.Services,
    templates_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    lan_settings = dataclasses.replace(settings, lan=True, host="0.0.0.0", port=5050, seed_knowledge=False)
    created: dict[str, Any] = {}
    real_create_app = app_module.create_app

    def fake_create_app(_settings: Settings) -> tuple[Flask, SocketIO]:
        flask_app, socketio = real_create_app(lan_settings, fakes, template_folder=str(templates_dir))
        created["app"] = flask_app
        monkeypatch.setattr(socketio, "run", lambda *a, **k: None)
        return flask_app, socketio

    fakes.vision.start = lambda: None  # type: ignore[attr-defined]
    fakes.vision.stop = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(app_module, "Settings", lambda: lan_settings)
    monkeypatch.setattr(app_module, "_port_free", lambda host, port: True)
    monkeypatch.setattr(app_module, "create_app", fake_create_app)
    monkeypatch.setattr(app_module, "_request_camera_access", lambda index: None)
    monkeypatch.setattr(app_module, "_warm_up", lambda svc: None)
    monkeypatch.setattr(app_module, "prepare_knowledge", lambda app, settings: None)
    with caplog.at_level("INFO", logger="biobench_sar.app"):
        app_module.main()
    pin = gate_of(created["app"]).pin
    assert f"http://{LAN_IP}:5050/join   PIN {pin[:3]} {pin[3:]}" in caplog.text
    assert "SAR_HOST='0.0.0.0' reaches the network" in caplog.text
