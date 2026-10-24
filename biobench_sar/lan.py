"""Phones and tablets as remote dashboards over the classroom Wi-Fi: addresses, the join URL and the PIN gate.

Pure Python and thread-safe, so Flask request threads and Socket.IO handlers can share one JoinPin. How the server
uses it (SAR_LAN=1, GET /join, the before_request and connect gates) is specified in docs/MOBILE.md.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import secrets
import socket
import threading
import time
import unicodedata
from collections import deque
from collections.abc import Callable, Iterable
from typing import Any
from urllib.parse import urlencode, urlsplit

COOKIE_NAME = "sar_lan"
PIN_DIGITS = 6
MAX_FAILURES = 10
FAILURE_WINDOW_S = 60.0
LOCKOUT_S = 60.0
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


# ---- addresses -----------------------------------------------------------------
def local_ip_addresses() -> list[str]:
    """Non-loopback IPv4 addresses of this machine, the one phones on the same Wi-Fi most likely reach first.

    No network traffic: the route probe sends nothing, and only when it finds nothing does this machine's own
    hostname get resolved.
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


def is_request_allowed(gate: JoinPin, *, remote_addr: str | None, host: str | None, path: str, token: Any) -> bool:
    """The whole LAN-mode rule for an HTTP request or a Socket.IO connect.

    Host must name this machine; then the laptop (loopback) passes, open paths pass, and others need a live token.
    """
    if not is_allowed_host(host):
        return False
    if is_local_request(remote_addr):
        return True
    return is_open_path(path) or gate.is_valid_token(token)


def cookie_kwargs(max_age_s: int = TOKEN_TTL_S) -> dict[str, Any]:
    """Arguments for response.set_cookie(COOKIE_NAME, token, **cookie_kwargs()).

    Not Secure: the classroom LAN is plain HTTP. Lax rather than Strict, so a home-screen launch still sends it.
    """
    return {"max_age": max_age_s, "httponly": True, "samesite": "Lax", "path": "/"}


# ---- the PIN and the device tokens ------------------------------------------------
class JoinPin:
    """The PIN shown on the laptop, the brute-force limit on guessing it, and tokens for devices that typed it.

    Failures count for the whole server, not per device: a classroom LAN has no trustworthy device identity, and
    a minute's shared lockout costs less than letting many devices guess in parallel. Everything lives in memory,
    so a restart means a new PIN and every device joins again.
    """

    def __init__(
        self,
        *,
        digits: int = PIN_DIGITS,
        max_failures: int = MAX_FAILURES,
        window_s: float = FAILURE_WINDOW_S,
        lockout_s: float = LOCKOUT_S,
        token_ttl_s: float = TOKEN_TTL_S,
        max_tokens: int = MAX_TOKENS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if digits < 4 or max_failures < 1 or max_tokens < 1:
            raise ValueError("digits must be >= 4, max_failures and max_tokens >= 1")
        self._digits = digits
        self._max_failures = max_failures
        self._window_s = window_s
        self._lockout_s = lockout_s
        self._token_ttl_s = token_ttl_s
        self._max_tokens = max_tokens
        self._clock = clock
        self._lock = threading.Lock()
        self._failures: deque[float] = deque()
        self._locked_until: float | None = None
        # sha256(token) -> expiry: lookups never compare the secret itself, so timing reveals nothing about it.
        self._tokens: dict[str, float] = {}
        self._pin = self._new_pin()

    @property
    def pin(self) -> str:
        with self._lock:
            return self._pin

    @property
    def token_ttl_s(self) -> float:
        return self._token_ttl_s

    def rotate(self, *, revoke_tokens: bool = False) -> str:
        """A new PIN, never the same as the old one; guesses at the old PIN no longer count against it."""
        with self._lock:
            old = self._pin
            while self._pin == old:
                self._pin = self._new_pin()
            self._failures.clear()
            self._locked_until = None
            if revoke_tokens:
                self._tokens.clear()
            return self._pin

    def verify(self, pin: object) -> bool:
        """Constant-time check of a typed PIN (spaces and dashes ignored). Always False while locked out."""
        candidate = _normalize_pin(pin)
        with self._lock:
            now = self._clock()
            if self._is_locked(now):
                return False
            if candidate is not None and hmac.compare_digest(candidate.encode(), self._pin.encode()):
                return True
            self._failures.append(now)
            while self._failures and self._failures[0] <= now - self._window_s:
                self._failures.popleft()
            if len(self._failures) >= self._max_failures:
                self._locked_until = now + self._lockout_s
                self._failures.clear()
            return False

    def locked_for(self) -> float:
        """Seconds left in the current lockout; 0.0 when guesses are accepted."""
        with self._lock:
            now = self._clock()
            return max(0.0, self._locked_until - now) if self._is_locked(now) else 0.0

    def issue_token(self) -> str:
        """A fresh random cookie token, valid for token_ttl_s. Call it only after verify() returned True."""
        token = secrets.token_urlsafe(32)
        with self._lock:
            now = self._clock()
            self._drop_expired(now)
            while len(self._tokens) >= self._max_tokens:
                del self._tokens[next(iter(self._tokens))]
            self._tokens[_digest(token)] = now + self._token_ttl_s
        return token

    def redeem(self, pin: object) -> str | None:
        """verify() then issue_token(): the token for a right PIN, None otherwise."""
        return self.issue_token() if self.verify(pin) else None

    def is_valid_token(self, token: object) -> bool:
        if not isinstance(token, str) or not token or len(token) > MAX_TOKEN_INPUT:
            return False
        key = _digest(token)
        with self._lock:
            expiry = self._tokens.get(key)
            if expiry is None:
                return False
            if self._clock() >= expiry:
                del self._tokens[key]
                return False
            return True

    def revoke_token(self, token: object) -> None:
        if isinstance(token, str) and token and len(token) <= MAX_TOKEN_INPUT:
            with self._lock:
                self._tokens.pop(_digest(token), None)

    def revoke_all(self) -> None:
        with self._lock:
            self._tokens.clear()

    def _new_pin(self) -> str:
        return f"{secrets.randbelow(10**self._digits):0{self._digits}d}"

    def _is_locked(self, now: float) -> bool:
        if self._locked_until is None:
            return False
        if now < self._locked_until:
            return True
        self._locked_until = None
        return False

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
