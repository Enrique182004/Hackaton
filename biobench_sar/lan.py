"""Phones and tablets as remote dashboards over the classroom Wi-Fi: addresses, the join URL and the PIN gate.

No Flask app needed and thread-safe, so Flask request threads and Socket.IO handlers can share one JoinPin. How
the server uses it (LAN mode, GET /join, the before_request gate, SocketIOGate) is specified in docs/MOBILE.md.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import re
import secrets
import socket
import threading
import time
import unicodedata
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode, urlsplit

from werkzeug.http import parse_cookie

COOKIE_NAME = "sar_lan"
# Every response in LAN mode carries it; the service worker retires itself on an unmarked one (see sw.js).
LAN_MODE_HEADER = "X-SAR-LAN"
PIN_DIGITS = 6
# Per device (TCP peer address): MAX_FAILURES misses within FAILURE_WINDOW_S lock that device for LOCKOUT_S,
# doubling with each further lockout up to MAX_LOCKOUT_S. A device quiet for FORGET_S starts over.
# Five, so that one device (at most 30 misses in its first hour) can never reach the server-wide cap alone.
MAX_FAILURES = 5
FAILURE_WINDOW_S = 60.0
LOCKOUT_S = 60.0
MAX_LOCKOUT_S = 60.0 * 60
FORGET_S = 24 * 60 * 60.0
# For the whole server: past GLOBAL_MAX_FAILURES misses in GLOBAL_WINDOW_S nobody may guess, so many addresses
# guessing in parallel are bounded too. Devices that already joined are unaffected.
GLOBAL_MAX_FAILURES = 50
GLOBAL_WINDOW_S = 60.0 * 60
MAX_CLIENTS = 1024
TOKEN_TTL_S = 12 * 60 * 60
# A class set of devices with room for re-joins; past it the oldest token is dropped, so memory stays bounded.
MAX_TOKENS = 256
MAX_PIN_INPUT = 32
MAX_TOKEN_INPUT = 128

# Reachable without the PIN: the join form itself and what any page needs to load (no data behind them).
OPEN_PATHS = frozenset({"/join", "/sw.js", "/favicon.ico"})
OPEN_PREFIXES = ("/static/",)
LOCAL_NAMES = frozenset({"localhost"})

# connect() on a UDP socket only picks the outgoing interface; no packet is sent. TEST-NET-1 (RFC 5737) follows
# the default route; the private-range targets still find the Wi-Fi on a classroom network with no gateway.
_ROUTE_PROBES = (("192.0.2.1", 9), ("10.254.254.254", 9), ("172.31.254.254", 9), ("192.168.254.254", 9))
# Carrier-grade NAT space: VPN overlays such as Tailscale live here, and phones on the Wi-Fi rarely reach them.
_SHARED = ipaddress.IPv4Network("100.64.0.0/10")
_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")


# ---- addresses -----------------------------------------------------------------
def local_ip_addresses() -> list[str]:
    """Non-loopback IPv4 addresses of this machine, the one phones on the same Wi-Fi most likely reach first.

    The route probe sends nothing. Only when it finds no address is this machine's own hostname resolved, which
    can send an mDNS or DNS query. A full-tunnel VPN makes the probe find the VPN address: disconnect it.
    """
    found = rank_addresses(_route_addresses())
    return found or rank_addresses(_hostname_addresses())


def rank_addresses(candidates: Iterable[str]) -> list[str]:
    """Usable IPv4 addresses among candidates, deduplicated, best first; ties keep the order they were found in."""
    ranks: dict[str, int] = {}
    for raw in candidates:
        try:
            ip = ipaddress.IPv4Address(str(raw).strip())
        except ValueError:
            continue
        rank = _rank(ip)
        if rank is not None:
            ranks.setdefault(str(ip), rank)
    return sorted(ranks, key=ranks.__getitem__)


def _rank(ip: ipaddress.IPv4Address) -> int | None:
    if ip.is_loopback or ip.is_unspecified or ip.is_multicast or ip.is_reserved:
        return None
    if ip.is_link_local:  # 169.254/16: no DHCP answer, so probably not the classroom network
        return 2
    if ip in _SHARED:
        return 1
    return 0


def _route_addresses() -> list[str]:
    found: list[str] = []
    for target in _ROUTE_PROBES:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(target)
                found.append(str(sock.getsockname()[0]))
        except OSError:
            continue
    return found


def _hostname_addresses() -> list[str]:
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET, socket.SOCK_STREAM)
    except (OSError, UnicodeError):
        return []
    return [str(info[4][0]) for info in infos]


def join_url(host_ip: str, port: int, pin: str) -> str:
    """http://<ip>:<port>/join?pin=<pin>: what the laptop dashboard's QR code encodes."""
    host = host_ip.strip()
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{int(port)}/join?{urlencode({'pin': pin})}"


# ---- settings ---------------------------------------------------------------------
def is_loopback_bind(host: str | None) -> bool:
    """True if binding to host keeps the server on this machine: 127.0.0.0/8, ::1 or localhost.

    "" binds every interface, like 0.0.0.0 and ::, and any other name may resolve to a LAN address.
    """
    if not isinstance(host, str):
        return False
    text = host.strip().strip("[]").split("%", 1)[0].rstrip(".").lower()
    if text in LOCAL_NAMES:
        return True
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped.is_loopback
    return ip.is_loopback


def is_lan_mode(*, sar_lan: bool, bind_host: str | None) -> bool:
    """LAN mode (the PIN gate) is on for SAR_LAN=1 and for every bind other devices could reach."""
    return sar_lan or not is_loopback_bind(bind_host)


def parse_hostnames(raw: str | None) -> tuple[str, ...]:
    """SAR_LAN_HOSTNAMES ("my-laptop.local, lab-pc"): lowercase DNS names, deduplicated, invalid ones dropped.

    IP literals are always allowed, so they are dropped here; so are wildcards.
    """
    if not isinstance(raw, str):
        return ()
    names: dict[str, None] = {}
    for part in re.split(r"[\s,]+", raw):
        name = part.strip().lower().rstrip(".")
        if not name or len(name) > 253 or not all(_LABEL.fullmatch(label) for label in name.split(".")):
            continue
        try:
            ipaddress.ip_address(name)
            continue
        except ValueError:
            names.setdefault(name, None)
    return tuple(names)


# ---- who is asking ---------------------------------------------------------------
def is_local_request(remote_addr: str | None) -> bool:
    """True for a loopback peer (127.0.0.0/8, ::1, ::ffff:127.0.0.1): the laptop itself, which never needs the PIN.

    Pass the TCP peer (Flask's request.remote_addr), never a client-supplied header such as X-Forwarded-For.
    """
    if not isinstance(remote_addr, str) or not remote_addr.strip():
        return False
    text = remote_addr.strip().strip("[]").split("%", 1)[0]
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped.is_loopback
    return ip.is_loopback


def client_key(remote_addr: str | None) -> str:
    """Whose wrong PINs these are: the IPv4 peer, or its /64 for IPv6, where one device can hold many addresses."""
    if not isinstance(remote_addr, str):
        return "?"
    text = remote_addr.strip().strip("[]").split("%", 1)[0]
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        return "?"
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return str(ip.ipv4_mapped)
        return str(ipaddress.IPv6Network((ip, 64), strict=False))
    return str(ip)


def is_allowed_host(host_header: str | None, extra_names: Iterable[str] = ()) -> bool:
    """True if the Host header is an IP literal, localhost or one of extra_names (e.g. "my-laptop.local").

    The gate trusts loopback peers, so any other domain name is refused: a web page that DNS-rebinds its own name
    to this machine would otherwise ride on that trust. IP literals, which the QR code uses, cannot be rebound.
    """
    if not isinstance(host_header, str) or not host_header.strip():
        return False
    try:
        name = (urlsplit("//" + host_header.strip()).hostname or "").rstrip(".")
    except ValueError:
        return False
    if not name:
        return False
    try:
        ipaddress.ip_address(name)
        return True
    except ValueError:
        pass
    allowed = LOCAL_NAMES | {n.strip().lower().rstrip(".") for n in extra_names}
    return name in allowed


def is_open_path(path: str) -> bool:
    """Paths a device may load before joining: the PIN form and static files."""
    if not isinstance(path, str) or "/../" in path or "/./" in path or path.endswith(("/..", "/.")):
        return False
    return path in OPEN_PATHS or path.startswith(OPEN_PREFIXES)


def is_request_allowed(
    gate: JoinPin,
    *,
    remote_addr: str | None,
    host: str | None,
    path: str,
    token: Any,
    extra_names: Iterable[str] = (),
) -> bool:
    """The whole LAN-mode rule for an HTTP request or a Socket.IO request.

    Host must name this machine; then the laptop (loopback) passes, open paths pass, and others need a live token.
    """
    if not is_allowed_host(host, extra_names):
        return False
    if is_local_request(remote_addr):
        return True
    return is_open_path(path) or gate.is_valid_token(token)


def looks_like_navigation(*, accept: str | None, sec_fetch_mode: str | None, sec_fetch_site: str | None) -> bool:
    """True if a GET /join?pin= looks like a QR scan or a typed address rather than an <img> or a script's fetch.

    Browsers send Sec-Fetch-* only to https:// and localhost, so on plain-HTTP Wi-Fi the Accept header decides.
    A heuristic: it keeps passive tags on other sites from burning guesses, but a page can still fake it.
    """
    if sec_fetch_mode and sec_fetch_mode.strip().lower() != "navigate":
        return False
    if sec_fetch_site and sec_fetch_site.strip().lower() not in ("none", "same-origin"):
        return False
    return isinstance(accept, str) and "text/html" in accept.lower()


def cookie_kwargs(max_age_s: int = TOKEN_TTL_S) -> dict[str, Any]:
    """Arguments for response.set_cookie(COOKIE_NAME, token, **cookie_kwargs()).

    Not Secure: the classroom LAN is plain HTTP. Lax rather than Strict, so a home-screen launch still sends it.
    """
    return {"max_age": max_age_s, "httponly": True, "samesite": "Lax", "path": "/"}


# ---- the PIN and the device tokens ------------------------------------------------
@dataclass
class _Client:
    failures: deque[float] = field(default_factory=deque)
    strikes: int = 0
    locked_until: float = 0.0
    last_failure: float = 0.0


class JoinPin:
    """The PIN shown on the laptop, the brute-force limits on guessing it, and tokens for devices that typed it.

    Wrong PINs count per device (the TCP peer, see client_key) with a lockout that doubles each time, plus a cap
    for the whole server, so a script on one device locks only itself and many devices together stay bounded.
    Lockouts use a monotonic clock; token expiry uses the wall clock, which keeps running while the laptop
    sleeps. Everything lives in memory, so a restart means a new PIN and every device joins again.
    """

    def __init__(
        self,
        *,
        digits: int = PIN_DIGITS,
        max_failures: int = MAX_FAILURES,
        window_s: float = FAILURE_WINDOW_S,
        lockout_s: float = LOCKOUT_S,
        max_lockout_s: float = MAX_LOCKOUT_S,
        forget_s: float = FORGET_S,
        global_max_failures: int = GLOBAL_MAX_FAILURES,
        global_window_s: float = GLOBAL_WINDOW_S,
        max_clients: int = MAX_CLIENTS,
        token_ttl_s: float = TOKEN_TTL_S,
        max_tokens: int = MAX_TOKENS,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        if digits < 4 or min(max_failures, global_max_failures, max_clients, max_tokens) < 1:
            raise ValueError("digits must be >= 4; max_failures, global_max_failures, max_clients, max_tokens >= 1")
        self._digits = digits
        self._max_failures = max_failures
        self._window_s = window_s
        self._lockout_s = lockout_s
        self._max_lockout_s = max(lockout_s, max_lockout_s)
        self._forget_s = forget_s
        self._global_max = global_max_failures
        self._global_window_s = global_window_s
        self._max_clients = max_clients
        self._token_ttl_s = token_ttl_s
        self._max_tokens = max_tokens
        self._clock = clock
        self._wall_clock = wall_clock
        self._lock = threading.Lock()
        self._clients: OrderedDict[str, _Client] = OrderedDict()
        self._global_failures: deque[float] = deque()
        # sha256(token) -> wall-clock expiry: lookups never compare the secret itself, so timing reveals nothing.
        self._tokens: dict[str, float] = {}
        self._pin = self._new_pin()

    @property
    def pin(self) -> str:
        with self._lock:
            return self._pin

    @property
    def token_ttl_s(self) -> float:
        return self._token_ttl_s

    def rotate(self, *, revoke_tokens: bool = True) -> str:
        """A new PIN, never the same as the old one, and by default every joined device signed out.

        The server-wide budget starts over for the new PIN; each device's own lockout keeps running, so a script
        that was locked out stays locked out.
        """
        with self._lock:
            old = self._pin
            while self._pin == old:
                self._pin = self._new_pin()
            self._global_failures.clear()
            if revoke_tokens:
                self._tokens.clear()
            return self._pin

    def verify(self, pin: object, client: str | None = None) -> bool:
        """Constant-time check of a typed PIN (spaces and dashes ignored) from client, the TCP peer address.

        Always False while that device or the whole server is locked out; refused guesses are not counted.
        """
        candidate = _normalize_pin(pin)
        key = client_key(client)
        with self._lock:
            now = self._clock()
            if self._global_locked_for(now) > 0:
                return False
            record = self._client(key, now)
            if now < record.locked_until:
                return False
            if candidate is not None and hmac.compare_digest(candidate.encode(), self._pin.encode()):
                self._clients.pop(key, None)  # a right PIN forgives that device
                return True
            self._global_failures.append(now)
            record.last_failure = now
            record.failures.append(now)
            while record.failures and record.failures[0] <= now - self._window_s:
                record.failures.popleft()
            if len(record.failures) >= self._max_failures:
                record.strikes += 1
                record.locked_until = now + self._lockout_for(record.strikes)
                record.failures.clear()
            return False

    def locked_for(self, client: str | None = None) -> float:
        """Seconds until client (or, with the server-wide cap reached, anyone) may guess again; 0.0 when it may."""
        key = client_key(client)
        with self._lock:
            now = self._clock()
            record = self._clients.get(key)
            mine = max(0.0, record.locked_until - now) if record is not None else 0.0
            return max(mine, self._global_locked_for(now))

    def server_locked_for(self) -> float:
        """Seconds until anyone may guess again once the server-wide cap is reached; 0.0 below it."""
        with self._lock:
            return self._global_locked_for(self._clock())

    def issue_token(self) -> str:
        """A fresh random cookie token, valid for token_ttl_s. Call it only after verify() returned True."""
        token = secrets.token_urlsafe(32)
        with self._lock:
            now = self._wall_clock()
            self._drop_expired(now)
            while len(self._tokens) >= self._max_tokens:
                del self._tokens[next(iter(self._tokens))]
            self._tokens[_digest(token)] = now + self._token_ttl_s
        return token

    def redeem(self, pin: object, client: str | None = None) -> str | None:
        """verify() then issue_token(): the token for a right PIN, None otherwise."""
        return self.issue_token() if self.verify(pin, client) else None

    def is_valid_token(self, token: object) -> bool:
        if not isinstance(token, str) or not token or len(token) > MAX_TOKEN_INPUT:
            return False
        key = _digest(token)
        with self._lock:
            expiry = self._tokens.get(key)
            if expiry is None:
                return False
            if self._wall_clock() >= expiry:
                del self._tokens[key]
                return False
            return True

    def revoke_token(self, token: object) -> None:
        if isinstance(token, str) and token and len(token) <= MAX_TOKEN_INPUT:
            with self._lock:
                self._tokens.pop(_digest(token), None)

    def revoke_all(self) -> None:
        """Sign every joined device out; the PIN stays the same. Backs "Disconnect all phones"."""
        with self._lock:
            self._tokens.clear()

    def _new_pin(self) -> str:
        return f"{secrets.randbelow(10**self._digits):0{self._digits}d}"

    def _lockout_for(self, strikes: int) -> float:
        return min(self._lockout_s * 2 ** min(strikes - 1, 32), self._max_lockout_s)

    def _global_locked_for(self, now: float) -> float:
        failures = self._global_failures
        while failures and failures[0] <= now - self._global_window_s:
            failures.popleft()
        if len(failures) < self._global_max:
            return 0.0
        return max(0.0, failures[0] + self._global_window_s - now)

    def _client(self, key: str, now: float) -> _Client:
        record = self._clients.get(key)
        if record is not None and now >= record.locked_until and now - record.last_failure >= self._forget_s:
            record = None
        if record is None:
            while len(self._clients) >= self._max_clients:
                self._clients.popitem(last=False)
            record = _Client()
            self._clients[key] = record
        self._clients.move_to_end(key)
        return record

    def _drop_expired(self, now: float) -> None:
        for key in [k for k, expiry in self._tokens.items() if now >= expiry]:
            del self._tokens[key]


def _normalize_pin(pin: object) -> str | None:
    if not isinstance(pin, str) or len(pin) > MAX_PIN_INPUT:
        return None
    # NFKC turns full-width digits from some phone keyboards into ASCII ones.
    text = "".join(ch for ch in unicodedata.normalize("NFKC", pin) if not ch.isspace() and ch != "-")
    return text if text.isascii() and text.isdigit() else None


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8", "surrogatepass")).hexdigest()


# ---- Socket.IO -----------------------------------------------------------------------
class PhoneSessions:
    """Socket.IO sids of devices that joined with the PIN, so "Disconnect all phones" can drop live connections."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sids: set[str] = set()

    def add(self, sid: str) -> None:
        with self._lock:
            self._sids.add(sid)

    def discard(self, sid: str) -> None:
        with self._lock:
            self._sids.discard(sid)

    def take_all(self) -> list[str]:
        """Every sid, emptying the set: disconnect each one."""
        with self._lock:
            sids, self._sids = sorted(self._sids), set()
        return sids

    def __len__(self) -> int:
        with self._lock:
            return len(self._sids)


WSGIApp = Callable[[dict[str, Any], Callable[..., Any]], Iterable[bytes]]


class SocketIOGate:
    """WSGI middleware that refuses Socket.IO requests from devices without a live token, before python-socketio
    sees them, so no broadcast ever reaches a device that hasn't joined.

    Wrap after SocketIO(app), which puts its own middleware in app.wsgi_app, so this one runs first:
    app.wsgi_app = SocketIOGate(app.wsgi_app, gate, extra_names=settings.lan_hostnames). Every polling request
    and the WebSocket upgrade pass through here, so a revoked token stops a polling client at its next request.
    """

    def __init__(
        self, app: WSGIApp, gate: JoinPin, *, extra_names: Iterable[str] = (), path: str = "socket.io"
    ) -> None:
        self._app = app
        self._gate = gate
        self._extra_names = tuple(extra_names)
        self._prefix = "/" + path.strip("/")

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> Iterable[bytes]:
        path = environ.get("PATH_INFO") or ""
        if path == self._prefix or path.startswith(self._prefix + "/"):
            host = environ.get("HTTP_HOST")
            if not is_allowed_host(host, self._extra_names):
                return _refuse(start_response, "400 BAD REQUEST", "unknown host")
            if not is_request_allowed(
                self._gate,
                remote_addr=environ.get("REMOTE_ADDR"),
                host=host,
                path=path,
                token=parse_cookie(environ).get(COOKIE_NAME),
                extra_names=self._extra_names,
            ):
                return _refuse(start_response, "401 UNAUTHORIZED", "join with the PIN first")
        return self._app(environ, start_response)


def _refuse(start_response: Callable[..., Any], status: str, message: str) -> list[bytes]:
    body = ('{"error": "' + message + '"}').encode()
    start_response(
        status,
        [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(body))),
            ("Cache-Control", "no-store"),
        ],
    )
    return [body]
