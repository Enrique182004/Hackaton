"""Browser pages: offline-safe font loading and the vanilla JS behaviour, run under node with a fake DOM."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "biobench_sar" / "templates"
STATIC = ROOT / "biobench_sar" / "static"
FAKE_DOM = Path(__file__).with_name("js") / "fake_dom.mjs"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")

GENERIC_FAMILIES = {"serif", "sans-serif", "monospace", "system-ui", "ui-monospace", "ui-sans-serif"}
BLAND_FIRST_CHOICES = {"inter", "roboto", "arial", "helvetica", "system-ui", "-apple-system"}


def template_ids(name: str) -> list[str]:
    return re.findall(r'\bid="([^"]+)"', (TEMPLATES / name).read_text(encoding="utf-8"))


def without_noscript(html: str) -> str:
    return re.sub(r"<noscript>.*?</noscript>", "", html, flags=re.S | re.I)


# ---- fonts -----------------------------------------------------------------------


def test_precharge_web_fonts_never_block_first_paint() -> None:
    html = (TEMPLATES / "precharge.html").read_text(encoding="utf-8")
    font_links = re.findall(r"<link\b[^>]*fonts\.googleapis\.com/css[^>]*>", without_noscript(html))
    assert font_links, "dashboard should still request its web fonts"
    for link in font_links:
        assert 'rel="stylesheet"' not in link, f"render-blocking web font stylesheet: {link}"
        assert 'rel="preload"' in link and 'as="style"' in link
        assert "this.rel='stylesheet'" in link
    noscript = re.findall(r"<noscript>(.*?)</noscript>", html, flags=re.S | re.I)
    assert any("fonts.googleapis.com" in n and 'rel="stylesheet"' in n for n in noscript)


def test_projector_page_loads_no_remote_fonts() -> None:
    html = (TEMPLATES / "index.html").read_text(encoding="utf-8")
    css = (STATIC / "css" / "style.css").read_text(encoding="utf-8")
    assert "fonts.googleapis.com" not in html and "@import" not in css and "@font-face" not in css


def test_dashboard_font_stacks_fall_back_to_local_fonts() -> None:
    css = (STATIC / "css" / "dashboard.css").read_text(encoding="utf-8")
    stacks = dict(re.findall(r"--(font-[a-z]+):\s*([^;]+);", css))
    assert {"font-display", "font-body", "font-mono"} <= set(stacks)
    for name, stack in stacks.items():
        families = [f.strip().strip('"').lower() for f in stack.split(",")]
        assert families[0] not in BLAND_FIRST_CHOICES | GENERIC_FAMILIES, f"--{name} leads with {families[0]}"
        assert families[-1] in GENERIC_FAMILIES, f"--{name} has no generic fallback"
        assert [f for f in families[1:-1] if f not in GENERIC_FAMILIES], f"--{name} has no local named fallback"


# ---- JS under node -----------------------------------------------------------------


def run_node(tmp_path: Path, template: str, body: str) -> str:
    # A "type": "module" package makes any node version load the dashboard's .js files as ES modules.
    shutil.copytree(STATIC / "js" / "dashboard", tmp_path / "dashboard")
    (tmp_path / "package.json").write_text('{"type": "module"}', encoding="utf-8")
    prelude = (
        'import assert from "node:assert/strict";\n'
        'import vm from "node:vm";\n'
        'import { readFileSync } from "node:fs";\n'
        f"import {{ installDom, installFetch, installIo, settle }} from {json.dumps(FAKE_DOM.as_uri())};\n"
        f"const IDS = {json.dumps(template_ids(template))};\n"
        f"const PROJECTOR_JS = {json.dumps(str(STATIC / 'js' / 'main.js'))};\n"
        "const runClassic = (path) => vm.runInThisContext(readFileSync(path, 'utf8'), { filename: path });\n"
    )
    # Polling loops keep the event loop alive, so a passing scenario exits explicitly.
    script = tmp_path / "scenario.mjs"
    script.write_text(prelude + body + '\nconsole.log("ok");\nprocess.exit(0);\n', encoding="utf-8")
    proc = subprocess.run([NODE or "node", str(script)], capture_output=True, text=True, timeout=60, cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


DASHBOARD = """
const doc = installDom(IDS);
const $ = (id) => doc.getElementById(id);
"""


@needs_node
def test_dashboard_boots_without_socketio(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + """
const http = installFetch();
http.setStatus({ state: "listening" });
await import("./dashboard/main.js");
await settle();
const chip = $("conn-chip");
assert.match(chip.querySelector("span").textContent, /^offline/);
assert.match(chip.title, /socket\\.io/);
assert.equal(chip.dataset.tone, "warn");
for (const [id, type] of [["ask-form", "submit"], ["listen-btn", "click"], ["file-input", "change"],
                          ["search-form", "submit"], ["calibrate-btn", "click"]]) {
  assert.ok($(id).listeners[type]?.length, `${id} has no ${type} handler`);
}
const paths = http.calls.map((c) => c.path);
for (const p of ["/api/status", "/api/documents", "/api/config"]) assert.ok(paths.includes(p), `${p} never fetched`);
// With no socket, the status poll drives the state chip and the push-to-talk note.
assert.equal($("state-chip").dataset.state, "listening");
await $("listen-btn").fire("click");
assert.equal($("ask-msg").textContent, "Listening — speak now.");
""",
    )


@needs_node
def test_listen_note_follows_server_status_not_the_202(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + """
const sock = installIo();
installFetch();
await import("./dashboard/main.js");
sock.fire("connect");
await settle();
const msg = $("ask-msg");
await $("listen-btn").fire("click");
assert.equal(msg.textContent, "Starting microphone…");
sock.fire("status", { state: "idle" });  // e.g. the previous answer finishing first
assert.equal(msg.textContent, "Starting microphone…");
sock.fire("status", { state: "listening" });
assert.equal(msg.textContent, "Listening — speak now.");
sock.fire("status", { state: "transcribing" });
assert.equal(msg.textContent, "Transcribing…");
sock.fire("status", { state: "idle" });
assert.equal(msg.textContent, "");
""",
    )


@needs_node
def test_listen_status_that_beats_the_202_is_kept(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + """
const sock = installIo();
let reply;
installFetch({ "POST /api/listen": () => new Promise((resolve) => { reply = resolve; }) });
await import("./dashboard/main.js");
sock.fire("connect");
await settle();
const click = $("listen-btn").fire("click");
await settle();
sock.fire("status", { state: "listening" });
reply({ status: 202, body: { accepted: true } });
await click;
assert.equal($("ask-msg").textContent, "Listening — speak now.");
""",
    )


@needs_node
def test_idle_status_keeps_unrelated_ask_note(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + """
const sock = installIo();
installFetch();
await import("./dashboard/main.js");
sock.fire("connect");
$("ask-input").value = "Is my flowchart missing a step?";
await $("ask-form").fire("submit");
sock.fire("status", { state: "thinking" });
sock.fire("status", { state: "idle" });
assert.equal($("ask-msg").textContent, "Sent — watch the projector.");
""",
    )


@needs_node
def test_listen_503_explains_missing_microphone(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + """
installIo();
installFetch({ "POST /api/listen": { status: 503, body: { accepted: false, error: "no microphone available" } } });
await import("./dashboard/main.js");
await $("listen-btn").fire("click");
assert.match($("ask-msg").textContent, /^No microphone on the server machine\\./);
assert.match($("ask-msg").textContent, /type your question/);
assert.equal($("ask-msg").dataset.tone, "bad");
""",
    )


PROJECTOR = """
const doc = installDom(IDS, { bodyData: { projectorWidth: "1920", projectorHeight: "1080" } });
const $ = (id) => doc.getElementById(id);
const sock = installIo();
runClassic(PROJECTOR_JS);
"""


@needs_node
def test_projector_resync_keeps_status_that_arrived_first(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "index.html",
        PROJECTOR
        + """
sock.fire("status", { state: "thinking" });
sock.fire("connect");
assert.equal($("status").dataset.state, "thinking");
assert.equal($("status-label").textContent, "thinking");
""",
    )


@needs_node
def test_projector_resync_clears_transient_ui_after_reconnect(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "index.html",
        PROJECTOR
        + """
sock.fire("connect");
sock.fire("status", { state: "thinking" });
sock.fire("show_calibration", {});
sock.fire("transcript", { text: "what is this" });
assert.ok($("calibration-img"));
sock.fire("disconnect");
assert.equal($("status-label").textContent, "offline");
sock.fire("connect");
assert.equal($("calibration-img"), null);
assert.equal($("transcript").textContent, "");
assert.ok($("transcript").classList.contains("hidden"));
assert.ok(!doc.body.classList.contains("offline"));
// The pre-disconnect status is stale, so idle stands in until the server's push arrives.
assert.equal($("status-label").textContent, "idle");
sock.fire("status", { state: "speaking" });
assert.equal($("status-label").textContent, "speaking");
""",
    )
