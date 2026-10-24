"""Phone/tablet groundwork assets: the web manifest and icons, the service worker, the vendored QR generator and
docs/MOBILE.md. The service worker and the QR library run under node."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from PIL import Image

from biobench_sar import lan

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "biobench_sar" / "static"
ICONS = STATIC / "icons"
SW = STATIC / "sw.js"
QR = STATIC / "vendor" / "qrcode.min.js"
MOBILE_DOC = ROOT / "docs" / "MOBILE.md"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")

QR_SHA256 = "164fc2c1c9eaf0a03aa8dfdb855e82e41a5c6922fbad8bb3238116207e26bef7"


def run_node(tmp_path: Path, body: str) -> str:
    script = tmp_path / "scenario.mjs"
    prelude = (
        'import assert from "node:assert/strict";\n'
        'import vm from "node:vm";\n'
        'import { readFileSync } from "node:fs";\n'
        f"const SW = {json.dumps(str(SW))};\n"
        f"const QR = {json.dumps(str(QR))};\n"
    )
    script.write_text(prelude + body + '\nconsole.log("ok");\n', encoding="utf-8")
    proc = subprocess.run([NODE or "node", str(script)], capture_output=True, text=True, timeout=60, cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


# ---- manifest + icons ------------------------------------------------------------------
def manifest() -> dict:
    return json.loads((STATIC / "manifest.webmanifest").read_text(encoding="utf-8"))


def static_file(src: str) -> Path:
    assert src.startswith("/static/"), src
    return STATIC / src.removeprefix("/static/")


def test_manifest_opens_the_dashboard_standalone() -> None:
    m = manifest()
    assert m["start_url"] == "/dashboard"
    assert m["scope"] == "/"
    assert m["display"] == "standalone"
    assert m["name"] and m["short_name"]
    for key in ("background_color", "theme_color"):
        assert m[key].startswith("#")


def test_manifest_icons_exist_with_their_declared_sizes() -> None:
    icons = manifest()["icons"]
    purposes = {icon["purpose"] for icon in icons}
    assert purposes == {"any", "maskable"}
    for icon in icons:
        path = static_file(icon["src"])
        assert path.is_file(), path
        if icon["type"] == "image/png":
            width, height = (int(n) for n in icon["sizes"].split("x"))
            with Image.open(path) as image:
                assert image.size == (width, height)
                assert image.format == "PNG"
        else:
            assert icon["type"] == "image/svg+xml"
            ET.fromstring(path.read_text(encoding="utf-8"))
    assert {"192x192", "512x512"} <= {icon["sizes"] for icon in icons if icon["purpose"] == "any"}


@pytest.mark.parametrize(("name", "size"), [("icon-maskable-512.png", 512), ("apple-touch-icon.png", 180)])
def test_full_bleed_icons_are_opaque(name: str, size: int) -> None:
    # Maskable icons are cropped by the launcher, and iOS paints transparency black.
    with Image.open(ICONS / name) as image:
        assert image.size == (size, size)
        assert image.mode == "RGB"


def test_rounded_icons_have_transparent_corners_and_an_opaque_middle() -> None:
    with Image.open(ICONS / "icon-192.png") as image:
        rgba = image.convert("RGBA")
        assert rgba.getpixel((0, 0))[3] == 0
        assert rgba.getpixel((96, 96))[3] == 255


def test_icon_svg_matches_the_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = importlib.util.spec_from_file_location("make_icons", ROOT / "scripts" / "make_icons.py")
    assert spec and spec.loader
    make_icons = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "make_icons", make_icons)  # dataclasses look their module up there
    spec.loader.exec_module(make_icons)
    assert (ICONS / "icon.svg").read_text(encoding="utf-8") == make_icons.render_svg()
    tile = make_icons.render_png(64, full_bleed=False)
    assert tile.size == (64, 64)
    assert tile.mode == "RGBA"


# ---- vendored QR generator ---------------------------------------------------------------
def test_qr_library_is_the_unmodified_cdnjs_build_with_its_license() -> None:
    assert hashlib.sha256(QR.read_bytes()).hexdigest() == QR_SHA256
    notice = (STATIC / "vendor" / "qrcode.LICENSE.txt").read_text(encoding="utf-8")
    assert "MIT License" in notice
    assert "Copyright (c) 2009 Kazuhiko Arase" in notice
    assert "1.4.4" in notice


@needs_node
def test_qr_code_of_the_join_url_scans_back(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    url = lan.join_url("192.168.1.20", 5050, "042917")
    out = run_node(
        tmp_path,
        f"""
const ctx = vm.createContext({{}});
vm.runInContext(readFileSync(QR, "utf8"), ctx);
const qr = vm.runInContext("qrcode", ctx)(0, "M");
qr.addData({json.dumps(url)});
qr.make();
const svg = qr.createSvgTag({{ cellSize: 4, margin: 2, scalable: true }});
assert.match(svg, /^<svg[^>]*viewBox=/);
assert.doesNotMatch(svg, /width="\\d+px"/);
console.log(qr.createDataURL(8, 32));
""",
    )
    data_url = out.splitlines()[0]
    assert data_url.startswith("data:image/gif;base64,")
    with Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))) as gif:
        gray = np.array(gif.convert("L"))
    decoded, _, _ = cv2.QRCodeDetector().detectAndDecode(gray)
    assert decoded == url


# ---- service worker -------------------------------------------------------------------------
SW_HARNESS = r"""
const ORIGIN = "http://127.0.0.1:5050";
const stores = new Map();
const keyOf = (req) => (typeof req === "string" ? new URL(req, ORIGIN).href : req.url);
const caches = {
  async open(name) {
    if (!stores.has(name)) stores.set(name, new Map());
    const store = stores.get(name);
    return {
      async match(req) {
        const hit = store.get(keyOf(req));
        return hit ? hit.clone() : undefined;
      },
      async put(req, res) {
        store.set(keyOf(req), res);
      },
    };
  },
  async has(name) { return stores.has(name); },
  async keys() { return [...stores.keys()]; },
  async delete(name) { return stores.delete(name); },
};
const listeners = {};
const fetched = [];
let unregistered = 0;
let network = async () => { throw new TypeError("offline"); };
// lan: false is a server with LAN mode off, which leaves out the X-SAR-LAN mark.
const reply = (body, { type = "basic", redirected = false, status = 200, headers = {}, lan = true } = {}) => {
  const marked = lan ? { "X-SAR-LAN": "1" } : {};
  const res = new Response(body, { status, headers: { "Content-Type": "text/html", ...marked, ...headers } });
  Object.defineProperty(res, "type", { value: type });
  Object.defineProperty(res, "redirected", { value: redirected });
  return res;
};
const self = {
  location: { origin: ORIGIN },
  addEventListener: (type, fn) => { listeners[type] = fn; },
  skipWaiting: async () => {},
  clients: { claim: async () => {} },
  registration: { unregister: async () => { unregistered += 1; return true; } },
};
vm.runInNewContext(readFileSync(SW, "utf8"), {
  self, caches, Response, Headers, URL, Promise, console, setTimeout,
  fetch: (req, init) => { fetched.push(typeof req === "string" ? req : req.url); return network(req, init); },
});
const CACHE = "sar-shell-" + /const CACHE_VERSION = "([^"]+)"/.exec(readFileSync(SW, "utf8"))[1];

async function dispatch(path, { mode = "cors", method = "GET", headers = {}, origin = ORIGIN } = {}) {
  const waits = [];
  let responded = null;
  const event = {
    request: { url: origin + path, method, mode, headers: new Headers(headers) },
    respondWith(p) { responded = Promise.resolve(p); },
    waitUntil(p) { waits.push(p); },
  };
  listeners.fetch(event);
  if (!responded) return { handled: false };
  let response, error;
  try { response = await responded; } catch (err) { error = err; }
  await Promise.all(waits);
  await new Promise((r) => setTimeout(r, 5));
  await Promise.all(waits);
  return { handled: true, response, error };
}
const stored = async (path) => {
  const store = stores.get(CACHE);
  const hit = store && store.get(ORIGIN + path);
  return hit ? await hit.clone().text() : undefined;
};
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
async function install() {
  const waits = [];
  listeners.install({ waitUntil: (p) => waits.push(p) });
  await Promise.all(waits);
}
"""


@needs_node
def test_service_worker_leaves_live_and_secret_requests_alone(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        SW_HARNESS
        + r"""
network = async () => reply("x");
for (const [path, opts] of [
  ["/socket.io/?EIO=4&transport=polling", {}],
  ["/socket.io/?EIO=4&transport=websocket", {}],
  ["/socket.io", {}],
  ["/join?pin=123456", { mode: "navigate" }],
  ["/join", { mode: "navigate", method: "POST" }],
  ["/api/query", { method: "POST" }],
  ["/api/upload", { method: "POST" }],
  ["/api/frame.jpg?warped=1", {}],
  ["/api/listen", {}],
  ["/tts/abc.mp3", {}],
  ["/tts/abc.mp3", { headers: { range: "bytes=0-" } }],
  ["/calibration.png", {}],
  ["/sw.js", {}],
  ["/static/app.js", { origin: "https://fonts.googleapis.com" }],
  ["/static/css/dashboard.css", { headers: { range: "bytes=0-10" } }],
]) {
  const { handled } = await dispatch(path, opts);
  assert.equal(handled, false, path);
}
assert.equal(stores.get(CACHE), undefined);
""",
    )


@needs_node
def test_service_worker_never_touches_the_api(tmp_path: Path) -> None:
    # /api/lan carries the PIN: it must never reach CacheStorage, and nothing stale may stand in for live data.
    run_node(
        tmp_path,
        SW_HARNESS
        + r"""
const json = (body) => reply(JSON.stringify(body), { headers: { "Content-Type": "application/json" } });
network = async () => json({ enabled: true, pin: "042917" });
for (const [path, opts] of [
  ["/api/lan", {}],
  ["/api/lan/pin", {}],
  ["/api/status", {}],
  ["/api/history", {}],
  ["/api/documents", {}],
  ["/api", {}],
  ["/api/status", { mode: "navigate" }],
]) {
  const { handled } = await dispatch(path, opts);
  assert.equal(handled, false, path);
}
await install();
for (const [name, store] of stores) {
  for (const key of store.keys()) assert.doesNotMatch(new URL(key).pathname, /^\/api/, `${name} holds ${key}`);
}
assert.doesNotMatch(readFileSync(SW, "utf8"), /X-SAR-From-Cache/);
""",
    )


@needs_node
def test_service_worker_pages_are_network_first_with_a_cached_fallback(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        SW_HARNESS
        + r"""
const nav = { mode: "navigate" };
network = async () => reply("dashboard v1");
let r = await dispatch("/dashboard", nav);
assert.equal(await r.response.text(), "dashboard v1");
assert.equal(await stored("/dashboard"), "dashboard v1");

network = async () => reply("dashboard v2");
r = await dispatch("/dashboard", nav);
assert.equal(await r.response.text(), "dashboard v2", "fresh page wins when the laptop answers");

network = async () => { await wait(150); return reply("dashboard v3"); };
r = await dispatch("/dashboard", nav);
assert.equal(await r.response.text(), "dashboard v3", "a slow laptop is waited for, never replaced by the cache");

network = async () => { throw new TypeError("offline"); };
r = await dispatch("/dashboard", nav);
assert.equal(await r.response.text(), "dashboard v3", "cached page when the laptop is unreachable");

r = await dispatch("/precharge", nav);
assert.equal(r.response.status, 503);
assert.match(await r.response.text(), /Can't reach the laptop/);
""",
    )


@needs_node
def test_service_worker_never_stores_the_gate_redirect_or_no_store(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        SW_HARNESS
        + r"""
const nav = { mode: "navigate" };
network = async () => reply("<form>PIN</form>", { redirected: true });
let r = await dispatch("/dashboard", nav);
assert.equal(await r.response.text(), "<form>PIN</form>");
assert.equal(await stored("/dashboard"), undefined);

network = async () => reply("secret", { headers: { "Cache-Control": "no-store" } });
await dispatch("/dashboard", nav);
assert.equal(await stored("/dashboard"), undefined);

network = async () => reply("missing", { status: 404 });
await dispatch("/static/js/gone.js");
assert.equal(await stored("/static/js/gone.js"), undefined);

network = async () => reply("opaque", { type: "opaque" });
await dispatch("/static/js/x.js");
assert.equal(await stored("/static/js/x.js"), undefined);
assert.equal(unregistered, 0, "none of these mean LAN mode is off");
""",
    )


@needs_node
def test_service_worker_static_files_are_stale_while_revalidate_once_lan_mode_is_seen(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        SW_HARNESS
        + r"""
const path = "/static/css/dashboard.css";
await (await caches.open(CACHE)).put(path, reply("css v0"));  // left over from an earlier run of this worker
network = async () => reply("css v1");
let r = await dispatch(path);
assert.equal(await r.response.text(), "css v1", "a fresh worker goes to the network first");
network = async () => reply("css v2");
r = await dispatch(path);
assert.equal(await r.response.text(), "css v1", "after a LAN-mode answer the cached copy answers at once");
assert.equal(await stored(path), "css v2", "and the fresh one is stored for next time");
r = await dispatch(path);
assert.equal(await r.response.text(), "css v2");
network = async () => { throw new TypeError("offline"); };
r = await dispatch(path);
assert.equal(await r.response.text(), "css v2");
r = await dispatch("/static/never-seen.js");
assert.ok(r.error, "nothing cached and no network: the request fails as it would without a worker");
""",
    )


@needs_node
def test_service_worker_retires_itself_when_lan_mode_is_off(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        SW_HARNESS
        + r"""
network = async () => reply("dashboard (LAN)");
await dispatch("/dashboard", { mode: "navigate" });
await dispatch("/static/css/dashboard.css");
await caches.open("sar-shell-old");
assert.equal(await stored("/dashboard"), "dashboard (LAN)");

// The same port, later, in an ordinary development run: no X-SAR-LAN mark.
network = async () => reply("dashboard (dev, edited)", { lan: false });
let r = await dispatch("/dashboard", { mode: "navigate" });
assert.equal(await r.response.text(), "dashboard (dev, edited)");
assert.equal(unregistered, 1);
assert.deepEqual(await caches.keys(), [], "every sar-* cache is deleted");

for (const [path, opts] of [["/static/css/dashboard.css", {}], ["/dashboard", { mode: "navigate" }]]) {
  const { handled } = await dispatch(path, opts);
  assert.equal(handled, false, `${path}: a retired worker leaves everything to the network`);
}
assert.deepEqual(await caches.keys(), []);
assert.equal(unregistered, 1);
""",
    )


@needs_node
def test_service_worker_never_stores_an_unmarked_static_file(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        SW_HARNESS
        + r"""
network = async () => reply("dev css", { lan: false });
const r = await dispatch("/static/css/dashboard.css");
assert.equal(await r.response.text(), "dev css");
assert.equal(unregistered, 1);
assert.deepEqual(await caches.keys(), []);
""",
    )


@needs_node
def test_service_worker_install_precaches_best_effort_and_activate_drops_old_caches(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        SW_HARNESS
        + r"""
network = async (path) => {
  if (path === "/static/js/dashboard/tutor.js") throw new TypeError("offline");
  if (path === "/dashboard") return reply("<form>PIN</form>", { redirected: true });
  return reply("body of " + path);
};
await install();
assert.ok(fetched.includes("/static/vendor/qrcode.min.js"));
assert.equal(await stored("/static/css/dashboard.css"), "body of /static/css/dashboard.css");
assert.equal(await stored("/static/js/dashboard/tutor.js"), undefined);
assert.equal(await stored("/dashboard"), undefined, "a gated device must not precache the PIN form");

await caches.open("sar-shell-old");
await caches.open("someone-else");
const activations = [];
listeners.activate({ waitUntil: (p) => activations.push(p) });
await Promise.all(activations);
assert.deepEqual((await caches.keys()).sort(), [CACHE, "someone-else"].sort());
assert.equal(unregistered, 0);
""",
    )


@needs_node
def test_service_worker_installed_against_a_server_without_lan_mode_retires(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        SW_HARNESS
        + r"""
network = async (path) => reply("body of " + path, { lan: false });
await install();
assert.equal(unregistered, 1);
assert.equal(stores.get(CACHE), undefined);
""",
    )


def test_service_worker_lan_mark_matches_lan_py() -> None:
    import re

    text = SW.read_text(encoding="utf-8")
    header = re.search(r'const LAN_HEADER = "([^"]+)"', text)
    assert header and header.group(1) == lan.LAN_MODE_HEADER


def test_service_worker_shell_lists_files_that_exist() -> None:
    import re

    text = SW.read_text(encoding="utf-8")
    shell = re.search(r"const SHELL = \[(.*?)\];", text, re.S)
    assert shell
    paths = re.findall(r'"(/[^"]*)"', shell.group(1))
    assert "/dashboard" in paths
    for path in paths:
        if path.startswith("/static/"):
            assert static_file(path).is_file(), path


# ---- docs/MOBILE.md -----------------------------------------------------------------------
def test_mobile_doc_matches_the_code() -> None:
    doc = MOBILE_DOC.read_text(encoding="utf-8")
    for needle in (
        "SAR_LAN=1",
        "0.0.0.0",
        "GET /join",
        "before_request",
        "connect",
        lan.COOKIE_NAME,
        "is_request_allowed",
        "join_url",
        "JoinPin",
        "/sw.js",
        "manifest.webmanifest",
        "qrcode.min.js",
        "Continuity Camera",
        "DroidCam",
        "HTTPS",
        "SAR_LAN_HOSTNAMES",
        "SAR_HOST",
        "Cache-Control: no-store",
        "Service-Worker-Allowed",
        "rotate(revoke_tokens=True)",
        "POST /api/lan/revoke",
        "Disconnect all phones",
        "X-SAR-LAN: 1",
        "getRegistrations",
        "VPN",
        "tunnel",
    ):
        assert needle in doc, needle
    for name in (
        "local_ip_addresses",
        "is_local_request",
        "is_allowed_host",
        "cookie_kwargs",
        "redeem",
        "revoke_all",
        "server_locked_for",
        "locked_for",
        "is_lan_mode",
        "parse_hostnames",
        "looks_like_navigation",
        "SocketIOGate",
        "PhoneSessions",
        "LAN_MODE_HEADER",
    ):
        assert hasattr(lan, name) or hasattr(lan.JoinPin, name), name
        assert name in doc, name


def test_mobile_doc_states_the_real_guessing_limits() -> None:
    doc = MOBILE_DOC.read_text(encoding="utf-8")
    assert f"{lan.MAX_FAILURES} wrong PINs within a minute lock that device for 1 minute" in doc
    assert lan.LOCKOUT_S == 60 and lan.FAILURE_WINDOW_S == 60
    assert lan.MAX_LOCKOUT_S == 3600 and "up to **1 hour**" in doc
    assert f"at most **{lan.GLOBAL_MAX_FAILURES} wrong PINs an hour**" in doc and lan.GLOBAL_WINDOW_S == 3600
    assert lan.TOKEN_TTL_S == 12 * 3600 and "12 hours" in doc
    assert "takes weeks" not in doc, "the old claim hid the real odds"
    assert "X-SAR-From-Cache" not in doc, "the worker no longer stores /api/*"
