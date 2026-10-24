from __future__ import annotations

import ipaddress
import socket
import threading

import pytest

from biobench_sar import lan
from biobench_sar.lan import (
    COOKIE_NAME,
    TOKEN_TTL_S,
    JoinPin,
    cookie_kwargs,
    is_allowed_host,
    is_local_request,
    is_open_path,
    is_request_allowed,
    join_url,
    local_ip_addresses,
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


def test_local_ip_addresses_on_this_machine_is_safe_and_sends_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
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


# ---- the PIN ----------------------------------------------------------------------
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


def test_ten_wrong_tries_in_a_minute_lock_for_sixty_seconds() -> None:
    clock = FakeClock()
    gate = JoinPin(clock=clock)
    for _ in range(9):
        assert not gate.verify(wrong(gate.pin))
        clock.advance(1)
    assert gate.locked_for() == 0.0
    assert gate.verify(gate.pin)  # a right PIN still works before the tenth miss
    assert not gate.verify(wrong(gate.pin))
    assert gate.locked_for() == pytest.approx(60.0)
    assert not gate.verify(gate.pin)  # locked: even the right PIN is refused
    clock.advance(59.9)
    assert not gate.verify(gate.pin)
    clock.advance(0.2)
    assert gate.locked_for() == 0.0
    assert gate.verify(gate.pin)


def test_tries_during_a_lockout_do_not_extend_it() -> None:
    clock = FakeClock()
    gate = JoinPin(clock=clock)
    for _ in range(10):
        gate.verify(wrong(gate.pin))
    for _ in range(50):
        clock.advance(1)
        gate.verify(wrong(gate.pin))
    clock.advance(10.1)
    assert gate.verify(gate.pin)


def test_misses_spread_over_more_than_a_minute_never_lock() -> None:
    clock = FakeClock()
    gate = JoinPin(clock=clock)
    for _ in range(40):
        assert not gate.verify(wrong(gate.pin))
        clock.advance(7)  # about 8.6 misses per minute
    assert gate.locked_for() == 0.0
    assert gate.verify(gate.pin)


def test_rotate_clears_a_lockout() -> None:
    clock = FakeClock()
    gate = JoinPin(clock=clock)
    for _ in range(10):
        gate.verify(wrong(gate.pin))
    assert gate.locked_for() > 0
    new = gate.rotate()
    assert gate.locked_for() == 0.0
    assert gate.verify(new)


def test_concurrent_guesses_are_counted_exactly() -> None:
    clock = FakeClock()
    gate = JoinPin(max_failures=1_000_000, clock=clock)
    bad = wrong(gate.pin)
    barrier = threading.Barrier(8)

    def guess() -> None:
        barrier.wait()
        for _ in range(500):
            gate.verify(bad)

    threads = [threading.Thread(target=guess) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(gate._failures) == 4000


def test_concurrent_guesses_lock_once_and_stop_counting() -> None:
    gate = JoinPin(clock=FakeClock())
    pin = gate.pin
    barrier = threading.Barrier(8)

    def guess(offset: int) -> None:
        barrier.wait()
        for i in range(50):
            gate.verify(f"{(int(pin) + 1 + offset * 50 + i) % 1_000_000:06d}")

    threads = [threading.Thread(target=guess, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert gate.locked_for() == pytest.approx(60.0)
    assert len(gate._failures) == 0  # cleared when the lockout began; refused guesses are not counted
    assert not gate.verify(pin)


@pytest.mark.parametrize("kwargs", [{"digits": 3}, {"max_failures": 0}, {"max_tokens": 0}])
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
    gate = JoinPin(clock=clock)
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
    assert gate.redeem(wrong(gate.pin)) is None
    token = gate.redeem(gate.pin)
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
    gate.revoke_all()
    assert not gate.is_valid_token(b)


def test_rotate_keeps_joined_devices_unless_asked() -> None:
    gate = JoinPin()
    token = gate.issue_token()
    gate.rotate()
    assert gate.is_valid_token(token)
    gate.rotate(revoke_tokens=True)
    assert not gate.is_valid_token(token)


def test_token_store_is_bounded_dropping_the_oldest() -> None:
    gate = JoinPin(max_tokens=3)
    tokens = [gate.issue_token() for _ in range(5)]
    assert [gate.is_valid_token(t) for t in tokens] == [False, False, True, True, True]


def test_expired_tokens_are_dropped_on_issue() -> None:
    clock = FakeClock()
    gate = JoinPin(token_ttl_s=10, clock=clock)
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
