from __future__ import annotations

import ipaddress
import json
import socket
import threading
from collections.abc import Callable

import pytest
from flask import Flask
from flask.testing import FlaskClient
from werkzeug.test import TestResponse

from biobench_sar import lan
from biobench_sar.lan import (
    COOKIE_NAME,
    TOKEN_TTL_S,
    JoinPin,
    PhoneSessions,
    SocketIOGate,
    client_key,
    cookie_kwargs,
    is_allowed_host,
    is_lan_mode,
    is_local_request,
    is_loopback_bind,
    is_open_path,
    is_request_allowed,
    join_url,
    local_ip_addresses,
    looks_like_navigation,
    parse_hostnames,
    rank_addresses,
)


class FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def wrong(pin: str) -> str:
    return f"{(int(pin) + 1) % 1_000_000:06d}"


# ---- addresses -------------------------------------------------------------------
def test_rank_addresses_drops_unusable_and_duplicates() -> None:
    found = rank_addresses(
        ["127.0.0.1", "0.0.0.0", "224.0.0.1", "255.255.255.255", "not-an-ip", "::1", "192.168.1.20", "192.168.1.20"]
    )
    assert found == ["192.168.1.20"]


def test_rank_addresses_prefers_lan_then_vpn_then_link_local_keeping_discovery_order() -> None:
    found = rank_addresses(["169.254.3.4", "100.101.102.103", "10.0.0.7", "192.168.1.20", "203.0.113.9"])
    assert found == ["10.0.0.7", "192.168.1.20", "203.0.113.9", "100.101.102.103", "169.254.3.4"]


def test_local_ip_addresses_uses_the_route_probe_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lan, "_route_addresses", lambda: ["192.168.1.20", "127.0.0.1", "192.168.1.20"])
    monkeypatch.setattr(lan, "_hostname_addresses", lambda: pytest.fail("fallback must not run"))
    assert local_ip_addresses() == ["192.168.1.20"]


def test_local_ip_addresses_falls_back_to_the_hostname(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lan, "_route_addresses", lambda: ["127.0.0.1"])
    monkeypatch.setattr(lan, "_hostname_addresses", lambda: ["127.0.1.1", "10.1.2.3"])
    assert local_ip_addresses() == ["10.1.2.3"]


def test_local_ip_addresses_on_this_machine_never_sends_on_a_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[object] = []
    real_socket = socket.socket

    class NoSend(real_socket):
        def send(self, *args: object, **kwargs: object) -> int:
            sent.append(args)
            raise AssertionError("must not send")

        def sendto(self, *args: object, **kwargs: object) -> int:
            sent.append(args)
            raise AssertionError("must not send")

    monkeypatch.setattr(socket, "socket", NoSend)
    found = local_ip_addresses()
    assert sent == []
    assert all(isinstance(a, str) for a in found)
    assert all(not ipaddress.IPv4Address(a).is_loopback for a in found)


def test_route_addresses_survive_an_unreachable_network(monkeypatch: pytest.MonkeyPatch) -> None:
    class Unreachable:
        def __init__(self, *args: object) -> None:
            pass

        def __enter__(self) -> Unreachable:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def connect(self, target: object) -> None:
            raise OSError("Network is unreachable")

    monkeypatch.setattr(socket, "socket", Unreachable)
    assert lan._route_addresses() == []


def test_hostname_addresses_survive_a_resolver_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object) -> list[object]:
        raise socket.gaierror("no such host")

    monkeypatch.setattr(socket, "getaddrinfo", boom)
    assert lan._hostname_addresses() == []


@pytest.mark.parametrize(
    ("host", "port", "pin", "expected"),
    [
        ("192.168.1.20", 5050, "012345", "http://192.168.1.20:5050/join?pin=012345"),
        (" 10.0.0.7 ", 8080, "999999", "http://10.0.0.7:8080/join?pin=999999"),
        ("fe80::1", 5050, "123456", "http://[fe80::1]:5050/join?pin=123456"),
    ],
)
def test_join_url(host: str, port: int, pin: str, expected: str) -> None:
    assert join_url(host, port, pin) == expected


# ---- who is asking --------------------------------------------------------------
@pytest.mark.parametrize(
    ("addr", "expected"),
    [
        ("127.0.0.1", True),
        ("127.0.1.1", True),
        ("::1", True),
        ("[::1]", True),
        ("::ffff:127.0.0.1", True),
        ("192.168.1.20", False),
        ("::ffff:192.168.1.20", False),
        ("fe80::1%en0", False),
        ("localhost", False),
        ("", False),
        (None, False),
        ("garbage", False),
    ],
)
def test_is_local_request(addr: str | None, expected: bool) -> None:
    assert is_local_request(addr) is expected


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1:5050", True),
        ("localhost:5050", True),
        ("LOCALHOST", True),
        ("localhost.:5050", True),
        ("[::1]:5050", True),
        ("192.168.1.20:5050", True),
        ("192.168.1.20", True),
        ("evil.example:5050", False),
        ("localhost.evil.example", False),
        ("", False),
        (None, False),
        ("[::1", False),
    ],
)
def test_is_allowed_host_refuses_names_that_could_be_rebound(host: str | None, expected: bool) -> None:
    assert is_allowed_host(host) is expected


def test_is_allowed_host_accepts_configured_names() -> None:
    assert is_allowed_host("My-Laptop.local:5050", ["my-laptop.local"])
    assert not is_allowed_host("other.local:5050", ["my-laptop.local"])


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/join", True),
        ("/sw.js", True),
        ("/static/css/dashboard.css", True),
        ("/static/manifest.webmanifest", True),
        ("/dashboard", False),
        ("/", False),
        ("/api/status", False),
        ("/socket.io/", False),
        ("/joinx", False),
        ("/static/../api/status", False),
        ("/static/./x", False),
        ("/static/..", False),
    ],
)
def test_is_open_path(path: str, expected: bool) -> None:
    assert is_open_path(path) is expected


def test_is_request_allowed_lets_the_laptop_in_and_gates_everyone_else() -> None:
    gate = JoinPin()
    token = gate.redeem(gate.pin)
    phone = {"remote_addr": "192.168.1.30", "host": "192.168.1.20:5050"}
    assert is_request_allowed(gate, remote_addr="127.0.0.1", host="127.0.0.1:5050", path="/api/status", token=None)
    assert is_request_allowed(gate, **phone, path="/join", token=None)
    assert is_request_allowed(gate, **phone, path="/static/js/dashboard/main.js", token=None)
    assert not is_request_allowed(gate, **phone, path="/dashboard", token=None)
    assert not is_request_allowed(gate, **phone, path="/socket.io/", token="forged")
    assert is_request_allowed(gate, **phone, path="/dashboard", token=token)
    assert is_request_allowed(gate, **phone, path="/socket.io/", token=token)


def test_is_request_allowed_refuses_a_rebound_name_even_from_loopback() -> None:
    gate = JoinPin()
    assert not is_request_allowed(gate, remote_addr="127.0.0.1", host="evil.example:5050", path="/", token=None)
    assert not is_request_allowed(gate, remote_addr="127.0.0.1", host="evil.example:5050", path="/join", token=None)


def test_cookie_kwargs_fit_flask_set_cookie() -> None:
    from flask import Flask

    kwargs = cookie_kwargs()
    assert kwargs["max_age"] == TOKEN_TTL_S
    assert kwargs["httponly"] is True
    assert kwargs["samesite"] == "Lax"
    app = Flask(__name__)
    with app.test_request_context():
        response = app.make_response("ok")
        response.set_cookie(COOKIE_NAME, "t", **kwargs)
        header = response.headers["Set-Cookie"]
    assert header.startswith(f"{COOKIE_NAME}=t;")
    assert "HttpOnly" in header
    assert "SameSite=Lax" in header
    assert "Path=/" in header


def test_is_request_allowed_accepts_configured_hostnames() -> None:
    gate = JoinPin()
    token = gate.redeem(gate.pin, "192.168.1.30")
    phone = {"remote_addr": "192.168.1.30", "host": "my-laptop.local:5050", "path": "/dashboard", "token": token}
    assert not is_request_allowed(gate, **phone)
    assert is_request_allowed(gate, **phone, extra_names=("my-laptop.local",))
    assert not is_request_allowed(gate, **{**phone, "token": None}, extra_names=("my-laptop.local",))


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"accept": "text/html,application/xhtml+xml,*/*;q=0.8"}, True),  # plain-HTTP QR scan: no Sec-Fetch-*
        ({"accept": "TEXT/HTML"}, True),
        ({"accept": "text/html", "sec_fetch_mode": "navigate", "sec_fetch_site": "none"}, True),
        ({"accept": "text/html", "sec_fetch_mode": "navigate", "sec_fetch_site": "same-origin"}, True),
        ({"accept": "image/avif,image/webp,image/*,*/*;q=0.8"}, False),  # <img src=/join?pin=...>
        ({"accept": "*/*"}, False),  # fetch() or <script>
        ({"accept": None}, False),
        ({"accept": "text/html", "sec_fetch_mode": "no-cors"}, False),
        ({"accept": "text/html", "sec_fetch_mode": "navigate", "sec_fetch_site": "cross-site"}, False),
    ],
)
def test_looks_like_navigation(headers: dict[str, str | None], expected: bool) -> None:
    kwargs = {"accept": None, "sec_fetch_mode": None, "sec_fetch_site": None, **headers}
    assert looks_like_navigation(**kwargs) is expected


# ---- settings ---------------------------------------------------------------------
@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", True),
        ("127.0.1.1", True),
        ("localhost", True),
        ("LOCALHOST.", True),
        ("::1", True),
        ("[::1]", True),
        ("::ffff:127.0.0.1", True),
        ("0.0.0.0", False),
        ("::", False),
        ("", False),  # binds every interface
        ("192.168.1.20", False),
        ("my-laptop.local", False),
        ("localhost.evil.example", False),
        (None, False),
    ],
)
def test_is_loopback_bind(host: str | None, expected: bool) -> None:
    assert is_loopback_bind(host) is expected


@pytest.mark.parametrize(
    ("sar_lan", "bind_host", "expected"),
    [
        (False, "127.0.0.1", False),
        (False, "localhost", False),
        (False, "::1", False),
        (True, "127.0.0.1", True),
        (True, "0.0.0.0", True),
        (False, "0.0.0.0", True),  # SAR_HOST alone opens the server to the network, so the PIN gate is on
        (False, "192.168.1.20", True),
        (False, "", True),
    ],
)
def test_any_bind_other_devices_can_reach_turns_lan_mode_on(sar_lan: bool, bind_host: str, expected: bool) -> None:
    assert is_lan_mode(sar_lan=sar_lan, bind_host=bind_host) is expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, ()),
        ("", ()),
        ("my-laptop.local", ("my-laptop.local",)),
        (" My-Laptop.local. , lab-pc  my-laptop.local", ("my-laptop.local", "lab-pc")),
        ("*.example, -bad.local, bad-.local, a..b, under_score.local, 10.0.0.7, ::1", ()),
        ("x" * 64 + ".local", ()),
    ],
)
def test_parse_hostnames(raw: str | None, expected: tuple[str, ...]) -> None:
    assert parse_hostnames(raw) == expected


# ---- the PIN ----------------------------------------------------------------------
PHONE = "192.168.1.30"
OTHER = "192.168.1.31"
MISSES = lan.MAX_FAILURES


def test_pin_is_six_ascii_digits() -> None:
    pins = {JoinPin().pin for _ in range(50)}
    assert all(len(p) == 6 and p.isascii() and p.isdigit() for p in pins)
    assert len(pins) > 1


def test_rotate_always_changes_the_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = JoinPin()
    old = gate.pin
    draws = iter([int(old), int(old), (int(old) + 7) % 1_000_000])
    monkeypatch.setattr(lan.secrets, "randbelow", lambda n: next(draws))
    new = gate.rotate()
    assert new != old
    assert gate.pin == new
    assert not gate.verify(old)
    assert gate.verify(new)


@pytest.mark.parametrize("typed", ["{pin}", " {pin} ", "{a} {b}", "{a}-{b}", "{wide}"])
def test_verify_accepts_the_pin_as_people_type_it(typed: str) -> None:
    gate = JoinPin()
    pin = gate.pin
    wide = "".join(chr(0xFF10 + int(d)) for d in pin)  # full-width digits from some phone keyboards
    assert gate.verify(typed.format(pin=pin, a=pin[:3], b=pin[3:], wide=wide))


@pytest.mark.parametrize("typed", [None, 123456, "", "12345", "1234567", "abcdef", "١٢٣٤٥٦", "1" * 500])
def test_verify_rejects_malformed_input(typed: object) -> None:
    gate = JoinPin()
    assert not gate.verify(typed)


def test_verify_uses_a_constant_time_compare(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[bytes, bytes]] = []
    real = lan.hmac.compare_digest

    def spy(a: bytes, b: bytes) -> bool:
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(lan.hmac, "compare_digest", spy)
    gate = JoinPin()
    assert gate.verify(gate.pin)
    assert calls and calls[0][1] == gate.pin.encode()


def test_too_many_wrong_tries_in_a_minute_lock_that_device_for_sixty_seconds() -> None:
    clock = FakeClock()
    gate = JoinPin(clock=clock)
    for _ in range(MISSES - 1):
        assert not gate.verify(wrong(gate.pin), PHONE)
        clock.advance(1)
    assert gate.locked_for(PHONE) == 0.0
    assert not gate.verify(wrong(gate.pin), PHONE)
    assert gate.locked_for(PHONE) == pytest.approx(60.0)
    assert not gate.verify(gate.pin, PHONE)  # locked: even the right PIN is refused
    clock.advance(59.9)
    assert not gate.verify(gate.pin, PHONE)
    clock.advance(0.2)
    assert gate.locked_for(PHONE) == 0.0
    assert gate.verify(gate.pin, PHONE)


def test_a_right_pin_before_the_last_allowed_miss_joins_and_forgives_the_device() -> None:
    clock = FakeClock()
    gate = JoinPin(clock=clock)
    for _ in range(MISSES - 1):
        gate.verify(wrong(gate.pin), PHONE)
    assert gate.verify(gate.pin, PHONE)
    for _ in range(MISSES - 1):
        gate.verify(wrong(gate.pin), PHONE)
    assert gate.locked_for(PHONE) == 0.0, "the earlier misses no longer count"


def test_one_device_locking_itself_out_leaves_the_others_free() -> None:
    gate = JoinPin(clock=FakeClock())
    for _ in range(MISSES):
        gate.verify(wrong(gate.pin), PHONE)
    assert gate.locked_for(PHONE) > 0
    assert gate.locked_for(OTHER) == 0.0
    assert gate.server_locked_for() == 0.0
    assert gate.verify(gate.pin, OTHER)
    assert not gate.verify(gate.pin, PHONE)


def test_each_further_lockout_doubles_up_to_an_hour() -> None:
    clock = FakeClock()
    gate = JoinPin(global_max_failures=1_000_000, clock=clock)
    seen = []
    for _ in range(9):
        for _ in range(MISSES):
            gate.verify(wrong(gate.pin), PHONE)
        seen.append(gate.locked_for(PHONE))
        clock.advance(seen[-1])
    assert seen == pytest.approx([60, 120, 240, 480, 960, 1920, 3600, 3600, 3600])


def test_a_device_quiet_for_a_day_starts_over() -> None:
    clock = FakeClock()
    gate = JoinPin(clock=clock)
    for _ in range(3):
        for _ in range(MISSES):
            gate.verify(wrong(gate.pin), PHONE)
        clock.advance(gate.locked_for(PHONE))
    clock.advance(lan.FORGET_S)
    for _ in range(MISSES):
        gate.verify(wrong(gate.pin), PHONE)
    assert gate.locked_for(PHONE) == pytest.approx(60.0)


def test_tries_during_a_lockout_do_not_extend_it() -> None:
    clock = FakeClock()
    gate = JoinPin(clock=clock)
    for _ in range(MISSES):
        gate.verify(wrong(gate.pin), PHONE)
    for _ in range(50):
        clock.advance(1)
        gate.verify(wrong(gate.pin), PHONE)
    clock.advance(10.1)
    assert gate.verify(gate.pin, PHONE)


def test_misses_spread_over_more_than_a_minute_never_lock() -> None:
    clock = FakeClock()
    gate = JoinPin(clock=clock)
    for _ in range(40):
        assert not gate.verify(wrong(gate.pin), PHONE)
        clock.advance(16)  # never MISSES within one minute
    assert gate.locked_for(PHONE) == 0.0
    assert gate.verify(gate.pin, PHONE)


def test_the_server_wide_cap_stops_many_devices_guessing_together() -> None:
    clock = FakeClock()
    gate = JoinPin(clock=clock)
    for n in range(lan.GLOBAL_MAX_FAILURES):
        assert not gate.verify(wrong(gate.pin), f"10.0.{n // 200}.{n % 200 + 1}")  # one miss per address
        clock.advance(10)
    newcomer = "10.9.9.9"
    remaining = lan.GLOBAL_WINDOW_S - 10 * lan.GLOBAL_MAX_FAILURES
    assert gate.locked_for(newcomer) == pytest.approx(remaining)
    assert gate.server_locked_for() == pytest.approx(remaining)
    assert not gate.verify(gate.pin, newcomer), "nobody may guess, not even with the right PIN"
    clock.advance(remaining)
    assert gate.locked_for(newcomer) == 0.0, "the oldest miss left the window: one more guess is allowed"
    assert not gate.verify(wrong(gate.pin), newcomer)
    assert gate.locked_for(newcomer) > 0


def test_the_server_wide_cap_leaves_joined_devices_alone() -> None:
    gate = JoinPin(global_max_failures=3, clock=FakeClock())
    token = gate.redeem(gate.pin, PHONE)
    for n in range(3):
        gate.verify(wrong(gate.pin), f"10.0.0.{n + 1}")
    assert gate.locked_for(OTHER) > 0
    assert gate.is_valid_token(token)


def test_rotate_resets_the_server_budget_but_not_a_device_lockout() -> None:
    clock = FakeClock()
    gate = JoinPin(global_max_failures=MISSES + 2, clock=clock)
    for _ in range(MISSES):
        gate.verify(wrong(gate.pin), PHONE)
    gate.verify(wrong(gate.pin), OTHER)
    gate.verify(wrong(gate.pin), OTHER)
    assert gate.locked_for(OTHER) > 0, "server-wide cap reached"
    new = gate.rotate()
    assert gate.locked_for(OTHER) == 0.0
    assert gate.verify(new, OTHER)
    assert gate.locked_for(PHONE) == pytest.approx(60.0), "a script that locked itself out stays locked out"
    assert not gate.verify(new, PHONE)


def test_unknown_peers_share_one_bucket() -> None:
    gate = JoinPin(clock=FakeClock())
    for _ in range(MISSES):
        gate.verify(wrong(gate.pin))
    assert gate.locked_for() > 0
    assert gate.locked_for("garbage") > 0
    assert gate.locked_for(PHONE) == 0.0


def test_tracked_devices_are_bounded() -> None:
    gate = JoinPin(max_clients=4, clock=FakeClock())
    for n in range(10):
        gate.verify(wrong(gate.pin), f"10.0.0.{n + 1}")
    assert len(gate._clients) == 4
    assert list(gate._clients) == [f"10.0.0.{n}" for n in range(7, 11)]


@pytest.mark.parametrize(
    ("addr", "expected"),
    [
        ("192.168.1.30", "192.168.1.30"),
        ("::ffff:192.168.1.30", "192.168.1.30"),
        ("2001:db8:1:2:aaaa::1", "2001:db8:1:2::/64"),
        ("[2001:db8:1:2:bbbb::9]", "2001:db8:1:2::/64"),  # privacy addresses of one phone share a bucket
        ("fe80::1%en0", "fe80::/64"),
        ("garbage", "?"),
        (None, "?"),
    ],
)
def test_client_key(addr: str | None, expected: str) -> None:
    assert client_key(addr) == expected


def test_concurrent_guesses_are_counted_exactly() -> None:
    clock = FakeClock()
    gate = JoinPin(max_failures=1_000_000, global_max_failures=1_000_000, clock=clock)
    bad = wrong(gate.pin)
    barrier = threading.Barrier(8)

    def guess(n: int) -> None:
        barrier.wait()
        for _ in range(500):
            gate.verify(bad, f"10.0.0.{n % 2 + 1}")

    threads = [threading.Thread(target=guess, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(gate._global_failures) == 4000
    assert [len(c.failures) for c in gate._clients.values()] == [2000, 2000]


def test_concurrent_guesses_lock_once_and_stop_counting() -> None:
    gate = JoinPin(clock=FakeClock())
    pin = gate.pin
    barrier = threading.Barrier(8)

    def guess(offset: int) -> None:
        barrier.wait()
        for i in range(50):
            gate.verify(f"{(int(pin) + 1 + offset * 50 + i) % 1_000_000:06d}", PHONE)

    threads = [threading.Thread(target=guess, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert gate.locked_for(PHONE) == pytest.approx(60.0)
    assert len(gate._global_failures) == MISSES, "refused guesses are not counted"
    assert not gate.verify(pin, PHONE)


@pytest.mark.parametrize(
    "kwargs",
    [{"digits": 3}, {"max_failures": 0}, {"max_tokens": 0}, {"global_max_failures": 0}, {"max_clients": 0}],
)
def test_bad_settings_are_refused(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        JoinPin(**kwargs)


def test_other_digit_counts() -> None:
    gate = JoinPin(digits=8)
    assert len(gate.pin) == 8
    assert gate.verify(gate.pin)


# ---- tokens -------------------------------------------------------------------------
def test_tokens_are_random_and_valid_until_twelve_hours() -> None:
    clock = FakeClock()
    gate = JoinPin(wall_clock=clock)
    a, b = gate.issue_token(), gate.issue_token()
    assert a != b
    assert len(a) >= 40
    assert gate.is_valid_token(a)
    assert gate.is_valid_token(b)
    clock.advance(12 * 3600 - 1)
    assert gate.is_valid_token(a)
    clock.advance(1)
    assert not gate.is_valid_token(a)
    assert not gate.is_valid_token(b)


def test_token_expiry_follows_the_wall_clock_through_a_laptop_sleep() -> None:
    # time.monotonic stops while macOS and Linux sleep; time.time does not.
    monotonic, wall = FakeClock(), FakeClock(1.7e9)
    gate = JoinPin(clock=monotonic, wall_clock=wall)
    token = gate.issue_token()
    wall.advance(TOKEN_TTL_S)
    assert not gate.is_valid_token(token)


@pytest.mark.parametrize("token", [None, "", "x", "a" * 500, 42, b"bytes"])
def test_unknown_or_malformed_tokens_are_invalid(token: object) -> None:
    gate = JoinPin()
    gate.issue_token()
    assert not gate.is_valid_token(token)


def test_tokens_are_stored_hashed() -> None:
    gate = JoinPin()
    token = gate.issue_token()
    assert token not in gate._tokens
    assert all(token not in key for key in gate._tokens)


def test_redeem() -> None:
    gate = JoinPin()
    assert gate.redeem(wrong(gate.pin), PHONE) is None
    token = gate.redeem(gate.pin, PHONE)
    assert token is not None
    assert gate.is_valid_token(token)


def test_revoke_token_and_revoke_all() -> None:
    gate = JoinPin()
    a, b = gate.issue_token(), gate.issue_token()
    gate.revoke_token(a)
    gate.revoke_token("never-issued")
    gate.revoke_token(None)
    assert not gate.is_valid_token(a)
    assert gate.is_valid_token(b)
    pin = gate.pin
    gate.revoke_all()
    assert not gate.is_valid_token(b)
    assert gate.pin == pin, "disconnecting every phone keeps the PIN"


def test_new_pin_signs_every_device_out_unless_asked_not_to() -> None:
    gate = JoinPin()
    token = gate.issue_token()
    gate.rotate(revoke_tokens=False)
    assert gate.is_valid_token(token)
    gate.rotate()
    assert not gate.is_valid_token(token)


def test_token_store_is_bounded_dropping_the_oldest() -> None:
    gate = JoinPin(max_tokens=3)
    tokens = [gate.issue_token() for _ in range(5)]
    assert [gate.is_valid_token(t) for t in tokens] == [False, False, True, True, True]


def test_expired_tokens_are_dropped_on_issue() -> None:
    clock = FakeClock()
    gate = JoinPin(token_ttl_s=10, wall_clock=clock)
    for _ in range(5):
        gate.issue_token()
    clock.advance(11)
    fresh = gate.issue_token()
    assert len(gate._tokens) == 1
    assert gate.is_valid_token(fresh)
    assert gate.token_ttl_s == 10


def test_concurrent_issue_and_check() -> None:
    gate = JoinPin(max_tokens=10_000)
    issued: list[str] = []
    invalid: list[str] = []
    lock = threading.Lock()

    def work() -> None:
        for _ in range(100):
            token = gate.issue_token()
            with lock:
                issued.append(token)
                if not gate.is_valid_token(token):
                    invalid.append(token)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert invalid == []
    assert len(set(issued)) == 800
    assert len(gate._tokens) == 800


# ---- Socket.IO ------------------------------------------------------------------------
def test_phone_sessions() -> None:
    sessions = PhoneSessions()
    sessions.add("b")
    sessions.add("a")
    sessions.add("a")
    sessions.discard("missing")
    assert len(sessions) == 2
    assert sessions.take_all() == ["a", "b"]
    assert len(sessions) == 0
    assert sessions.take_all() == []


def _inner_app(environ: dict[str, object], start_response: Callable[..., object]) -> list[bytes]:
    start_response("200 OK", [("Content-Type", "text/plain")])
    return [b"inner"]


def test_socketio_gate_refuses_before_python_socketio_sees_the_request() -> None:
    from werkzeug.test import Client

    gate = JoinPin()
    token = gate.redeem(gate.pin, PHONE)
    client = Client(SocketIOGate(_inner_app, gate, extra_names=["my-laptop.local"]))
    phone = {"REMOTE_ADDR": PHONE}
    lan_host = {"Host": "192.168.1.20:5050"}

    refused = client.get("/socket.io/?EIO=4&transport=polling", environ_base=phone, headers=lan_host)
    assert refused.status_code == 401
    assert refused.headers["Cache-Control"] == "no-store"
    assert json.loads(refused.get_data()) == {"error": "join with the PIN first"}
    assert client.get("/socket.io", environ_base=phone, headers=lan_host).status_code == 401

    client.set_cookie(COOKIE_NAME, token, domain="192.168.1.20")
    allowed = client.get("/socket.io/?EIO=4&transport=polling", environ_base=phone, headers=lan_host)
    assert allowed.status_code == 200
    assert allowed.get_data() == b"inner"

    rebound = client.get(
        "/socket.io/?EIO=4", environ_base={"REMOTE_ADDR": "127.0.0.1"}, headers={"Host": "evil.example"}
    )
    assert rebound.status_code == 400
    named = client.get("/socket.io/?EIO=4", environ_base=phone, headers={"Host": "my-laptop.local:5050"})
    assert named.status_code == 401, "a configured name is a valid host, the device still needs the cookie"

    gate.revoke_all()
    assert client.get("/socket.io/?EIO=4&sid=abc", environ_base=phone, headers=lan_host).status_code == 401


def test_socketio_gate_leaves_other_paths_and_the_laptop_alone() -> None:
    from werkzeug.test import Client

    client = Client(SocketIOGate(_inner_app, JoinPin()))
    assert client.get("/dashboard", environ_base={"REMOTE_ADDR": PHONE}).get_data() == b"inner"
    assert client.get("/socket.ioX", environ_base={"REMOTE_ADDR": PHONE}).get_data() == b"inner"
    local = client.get("/socket.io/?EIO=4", environ_base={"REMOTE_ADDR": "127.0.0.1"}, headers={"Host": "127.0.0.1"})
    assert local.get_data() == b"inner"


def test_socketio_gate_in_front_of_flask_socketio_blocks_the_handshake() -> None:
    from flask import Flask
    from flask_socketio import SocketIO

    gate = JoinPin()
    app = Flask(__name__)
    SocketIO(app, async_mode="threading", cors_allowed_origins=None, always_connect=True)
    app.wsgi_app = SocketIOGate(app.wsgi_app, gate)  # type: ignore[method-assign]
    client = app.test_client()
    phone = {"environ_base": {"REMOTE_ADDR": PHONE}, "headers": {"Host": "192.168.1.20:5050"}}

    refused = client.get("/socket.io/?EIO=4&transport=polling", **phone)
    assert refused.status_code == 401
    assert not refused.get_data().startswith(b"0{"), "no Engine.IO session was opened"

    client.set_cookie(COOKIE_NAME, gate.redeem(gate.pin, PHONE), domain="192.168.1.20")
    opened = client.get("/socket.io/?EIO=4&transport=polling", **phone)
    assert opened.status_code == 200
    assert opened.get_data().startswith(b"0{"), "the Engine.IO OPEN packet"


def test_one_device_alone_never_reaches_the_server_wide_cap() -> None:
    # A script on one address guessing as fast as it is allowed: about 145 guesses on day 1, then 120 a day.
    clock = FakeClock(0.0)
    gate = JoinPin(clock=clock)
    bad = wrong(gate.pin)
    per_day = [0, 0]
    while clock.now < 2 * 86_400:
        wait = gate.locked_for(PHONE)
        if wait > 0:
            clock.advance(wait)
            continue
        assert gate.server_locked_for() == 0.0
        gate.verify(bad, PHONE)
        per_day[int(clock.now // 86_400)] += 1
        clock.advance(0.01)
    assert per_day == [145, 120]


# ---- reference wiring (docs/MOBILE.md, steps 1-6 and 9) ------------------------------------
LAN_HOST = {"Host": "192.168.1.20:5050"}
HTML = {"Accept": "text/html,application/xhtml+xml,*/*;q=0.8"}


def reference_app(*, sar_lan: bool, bind_host: str, hostnames: str = "") -> tuple[Flask, JoinPin]:
    from pathlib import Path

    from flask import Response, abort, jsonify, make_response, redirect, request, send_from_directory
    from flask_socketio import ConnectionRefusedError, SocketIO

    from biobench_sar.lan import LAN_MODE_HEADER

    lan_on = is_lan_mode(sar_lan=sar_lan, bind_host=bind_host)
    names = parse_hostnames(hostnames)
    gate, phones = JoinPin(), PhoneSessions()
    static = Path(lan.__file__).parent / "static"
    app = Flask(__name__, static_folder=str(static))
    socketio = SocketIO(app, async_mode="threading", cors_allowed_origins=None, always_connect=True)
    if lan_on:
        app.wsgi_app = SocketIOGate(app.wsgi_app, gate, extra_names=names)  # type: ignore[method-assign]

    @app.before_request
    def lan_gate() -> object:
        if not lan_on:
            return None
        if not is_allowed_host(request.host, names):
            return jsonify({"error": "unknown host"}), 400
        if is_request_allowed(
            gate,
            remote_addr=request.remote_addr,
            host=request.host,
            path=request.path,
            token=request.cookies.get(COOKIE_NAME),
            extra_names=names,
        ):
            return None
        if request.path.startswith("/api/") or request.method != "GET":
            return jsonify({"error": "join with the PIN first"}), 401
        return redirect("/join")

    @app.after_request
    def lan_headers(response: Response) -> Response:
        if lan_on:
            response.headers[LAN_MODE_HEADER] = "1"
            if request.path.startswith("/api/") or request.path == "/join":
                response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/dashboard")
    def dashboard() -> str:
        return "dashboard"

    @app.get("/api/status")
    def status() -> object:
        return jsonify({"state": "ready"})

    @app.route("/join", methods=["GET", "POST"])
    def join() -> object:
        if is_local_request(request.remote_addr):
            return redirect("/dashboard")
        pin = request.form.get("pin") if request.method == "POST" else request.args.get("pin")
        if request.method == "GET" and not looks_like_navigation(
            accept=request.headers.get("Accept"),
            sec_fetch_mode=request.headers.get("Sec-Fetch-Mode"),
            sec_fetch_site=request.headers.get("Sec-Fetch-Site"),
        ):
            pin = None
        if pin is not None:
            token = gate.redeem(pin, request.remote_addr)
            if token:
                response = redirect("/dashboard", code=303)
                response.set_cookie(COOKIE_NAME, token, **cookie_kwargs())
                return response
        response = make_response(f"form, wait {gate.locked_for(request.remote_addr):.0f}")
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    def lan_body() -> dict[str, object]:
        urls = [join_url("192.168.1.20", 5050, gate.pin)]
        return {"enabled": True, "pin": gate.pin, "locked_for": gate.server_locked_for(), "urls": urls}

    @app.get("/api/lan")
    def api_lan() -> object:
        if not lan_on:
            return jsonify({"enabled": False})
        if not is_local_request(request.remote_addr):
            abort(403)
        return jsonify(lan_body())

    @app.post("/api/lan/pin")
    def new_pin() -> object:
        if not lan_on or not is_local_request(request.remote_addr):
            abort(403)
        gate.rotate(revoke_tokens=True)
        for sid in phones.take_all():
            socketio.server.disconnect(sid, namespace="/")
        return jsonify(lan_body())

    @app.post("/api/lan/revoke")
    def revoke() -> object:
        if not lan_on or not is_local_request(request.remote_addr):
            abort(403)
        gate.revoke_all()
        for sid in phones.take_all():
            socketio.server.disconnect(sid, namespace="/")
        return jsonify({"ok": True})

    @app.get("/sw.js")
    def service_worker() -> object:
        if not lan_on:
            abort(404)
        response = send_from_directory(str(static), "sw.js", mimetype="text/javascript", max_age=0)
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Service-Worker-Allowed"] = "/"
        return response

    @socketio.on("connect")
    def on_connect(_auth: object = None) -> None:
        if lan_on and not is_request_allowed(
            gate,
            remote_addr=request.remote_addr,
            host=request.host,
            path="/socket.io/",
            token=request.cookies.get(COOKIE_NAME),
            extra_names=names,
        ):
            raise ConnectionRefusedError("join with the PIN first")
        if lan_on and not is_local_request(request.remote_addr):
            phones.add(request.sid)  # type: ignore[attr-defined]

    return app, gate


def phone_get(client: FlaskClient, path: str, *, addr: str = PHONE, **headers: str) -> TestResponse:
    return client.get(path, environ_base={"REMOTE_ADDR": addr}, headers={**LAN_HOST, **headers})


def test_reference_wiring_gates_a_non_loopback_bind_without_sar_lan() -> None:
    app, _gate = reference_app(sar_lan=False, bind_host="0.0.0.0")
    client = app.test_client()
    assert phone_get(client, "/api/status").status_code == 401
    page = phone_get(client, "/dashboard")
    assert page.status_code == 302
    assert page.headers["Location"].endswith("/join")
    local = client.get("/api/status", headers={"Host": "127.0.0.1:5050"})
    assert local.status_code == 200


def test_reference_wiring_with_lan_mode_off_changes_nothing() -> None:
    app, _gate = reference_app(sar_lan=False, bind_host="127.0.0.1")
    client = app.test_client()
    response = client.get("/api/status")
    assert response.status_code == 200
    assert lan.LAN_MODE_HEADER not in response.headers
    assert client.get("/sw.js").status_code == 404, "nothing can register a worker outside LAN mode"
    assert client.get("/api/lan").get_json() == {"enabled": False}


def test_reference_wiring_marks_lan_mode_and_never_lets_api_be_cached() -> None:
    app, _gate = reference_app(sar_lan=True, bind_host="0.0.0.0")
    client = app.test_client()
    local = {"Host": "127.0.0.1:5050"}
    for path in ("/api/status", "/api/lan"):
        response = client.get(path, headers=local)
        assert response.status_code == 200, path
        assert response.headers["Cache-Control"] == "no-store", path
        assert response.headers[lan.LAN_MODE_HEADER] == "1", path
    assert phone_get(client, "/api/status").headers["Cache-Control"] == "no-store", "the 401 too"
    worker = client.get("/sw.js", headers=local)
    assert worker.status_code == 200
    assert worker.headers["Service-Worker-Allowed"] == "/"
    assert worker.headers["Cache-Control"] == "no-cache"
    assert worker.headers[lan.LAN_MODE_HEADER] == "1"


def test_reference_wiring_qr_join_and_passive_requests() -> None:
    app, gate = reference_app(sar_lan=True, bind_host="0.0.0.0")
    client = app.test_client()
    for _ in range(3 * MISSES):
        img = phone_get(client, f"/join?pin={wrong(gate.pin)}", Accept="image/avif,image/webp,*/*;q=0.8")
        assert img.status_code == 200
    assert gate.locked_for(PHONE) == 0.0, "an <img> tag never counts a guess"

    joined = phone_get(client, f"/join?pin={gate.pin}", **HTML)
    assert joined.status_code == 303
    assert joined.headers["Location"].endswith("/dashboard")
    assert COOKIE_NAME in joined.headers["Set-Cookie"]
    assert joined.headers["Cache-Control"] == "no-store"
    assert phone_get(client, "/dashboard").status_code == 200
    assert phone_get(client, "/api/lan").status_code == 403, "a joined phone never sees the PIN"


def test_reference_wiring_locks_one_phone_and_not_the_next() -> None:
    app, gate = reference_app(sar_lan=True, bind_host="0.0.0.0")
    client = app.test_client()
    for _ in range(MISSES):
        phone_get(client, f"/join?pin={wrong(gate.pin)}", **HTML)
    assert phone_get(client, f"/join?pin={gate.pin}", **HTML).status_code == 200, "locked out"
    assert phone_get(client, f"/join?pin={gate.pin}", addr=OTHER, **HTML).status_code == 303


def test_reference_wiring_hosts() -> None:
    app, _gate = reference_app(sar_lan=True, bind_host="0.0.0.0", hostnames="my-laptop.local")
    client = app.test_client()
    assert phone_get(client, "/dashboard", Host="evil.example:5050").status_code == 400
    assert client.get("/api/status", headers={"Host": "evil.example"}).status_code == 400, "even from loopback"
    assert phone_get(client, "/dashboard", Host="my-laptop.local:5050").status_code == 302, "host ok, needs the PIN"


@pytest.mark.parametrize("action", ["/api/lan/pin", "/api/lan/revoke"])
def test_reference_wiring_new_pin_and_disconnect_sign_phones_out(action: str) -> None:
    app, gate = reference_app(sar_lan=True, bind_host="0.0.0.0")
    client = app.test_client()
    old_pin = gate.pin
    assert phone_get(client, f"/join?pin={old_pin}", **HTML).status_code == 303
    assert phone_get(client, "/api/status").status_code == 200
    polling = phone_get(client, "/socket.io/?EIO=4&transport=polling")
    assert polling.get_data().startswith(b"0{")

    refused = client.post(action, environ_base={"REMOTE_ADDR": PHONE}, headers=LAN_HOST)
    assert refused.status_code == 403
    assert client.post(action, headers={"Host": "127.0.0.1:5050"}).status_code == 200
    assert phone_get(client, "/api/status").status_code == 401
    assert phone_get(client, "/socket.io/?EIO=4&transport=polling").status_code == 401
    assert (gate.pin != old_pin) is (action == "/api/lan/pin")
