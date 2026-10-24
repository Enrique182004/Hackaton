"""Browser pages: offline-safe font loading and the vanilla JS behaviour, run under node with a fake DOM.

Also what the pages' readers rely on: the docs and the seed documents must stay true to the code.
"""

from __future__ import annotations

import functools
import html
import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

from biobench_sar.config import Settings
from biobench_sar.rag.quiz_bank import QuizBank, is_worksheet, parse_quiz_file, parse_worksheet_text

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "biobench_sar" / "templates"
STATIC = ROOT / "biobench_sar" / "static"
MATH_SEEDS = ROOT / "biobench_sar" / "knowledge" / "math"
DOC_NAMES = (
    "README.md",
    "CLAUDE.md",
    "AGENTS.md",
    "CONTRIBUTING.md",
    "docs/BUILD_PLAN.md",
    "docs/sar_math_tutor_prompt.md",
)
DOCS = {name: (ROOT / name).read_text(encoding="utf-8") for name in DOC_NAMES}
FAKE_DOM = Path(__file__).with_name("js") / "fake_dom.mjs"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")

GENERIC_FAMILIES = {"serif", "sans-serif", "monospace", "cursive", "system-ui", "ui-monospace", "ui-sans-serif"}
BLAND_FIRST_CHOICES = {"inter", "roboto", "arial", "helvetica", "system-ui", "-apple-system"}


def _dataset_key(attr: str) -> str:
    return re.sub(r"-([a-z0-9])", lambda m: m.group(1).upper(), attr.removeprefix("data-"))


class _NodeReader(HTMLParser):
    """Every element the JS can reach: an id, or a data-i18n* text for i18n.js to fill in."""

    def __init__(self) -> None:
        super().__init__()
        self.nodes: list[dict[str, object]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"html", "body"}:
            return
        values = {name: value or "" for name, value in attrs}
        data = {_dataset_key(name): value for name, value in values.items() if name.startswith("data-")}
        if "id" in values or any(key.startswith("i18n") for key in data):
            node = {"id": values.get("id", ""), "tag": tag, "data": data, "hidden": "hidden" in values}
            self.nodes.append({**node, "classes": values.get("class", "").split()})


def template_nodes(name: str) -> list[dict[str, object]]:
    reader = _NodeReader()
    reader.feed((TEMPLATES / name).read_text(encoding="utf-8"))
    return reader.nodes


@functools.cache
def strings() -> dict[str, dict[str, str]]:
    """i18n.js's STRINGS, read by node so the tests see exactly what the pages use."""
    script = f"import {{ STRINGS }} from {json.dumps((STATIC / 'js' / 'i18n.js').as_uri())};"
    script += "console.log(JSON.stringify(STRINGS));"
    proc = subprocess.run([NODE or "node", "--input-type=module", "-e", script], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


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
    assert {"font-head", "font-body", "font-hand"} <= set(stacks)
    for name, stack in stacks.items():
        families = [f.strip().strip('"').lower() for f in stack.split(",")]
        assert families[0] not in BLAND_FIRST_CHOICES | GENERIC_FAMILIES, f"--{name} leads with {families[0]}"
        assert families[-1] in GENERIC_FAMILIES, f"--{name} has no generic fallback"
        assert [f for f in families[1:-1] if f not in GENERIC_FAMILIES], f"--{name} has no local named fallback"


# ---- JS under node -----------------------------------------------------------------


def run_node(tmp_path: Path, template: str, body: str) -> str:
    # A "type": "module" package makes any node version load the pages' .js files as ES modules. The layout
    # mirrors static/js: the projector's main.js and i18n.js beside the dashboard/ folder.
    shutil.copytree(STATIC / "js" / "dashboard", tmp_path / "dashboard")
    for name in ("main.js", "i18n.js", "join.js"):
        shutil.copy(STATIC / "js" / name, tmp_path / name)
    (tmp_path / "package.json").write_text('{"type": "module"}', encoding="utf-8")
    prelude = (
        'import assert from "node:assert/strict";\n'
        f"import {{ installDom, installFetch, installIo, settle }} from {json.dumps(FAKE_DOM.as_uri())};\n"
        f"const NODES = {json.dumps(template_nodes(template))};\n"
    )
    # Polling loops keep the event loop alive, so a passing scenario exits explicitly.
    script = tmp_path / "scenario.mjs"
    script.write_text(prelude + body + '\nconsole.log("ok");\nprocess.exit(0);\n', encoding="utf-8")
    proc = subprocess.run([NODE or "node", str(script)], capture_output=True, text=True, timeout=60, cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


DASHBOARD = """
const doc = installDom(NODES);
const $ = (id) => doc.getElementById(id);
const LISTENING = "Listening… ask your question now. It stops by itself after about 5 seconds.";
const text = (node) => (typeof node === "string" ? node : [node.textContent, ...node.children.map(text)].join(" ").trim());
const logText = () => $("log").children.map(text).join("\\n");
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
"""


@needs_node
def test_poll_loop_never_runs_twice_at_once(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + """
const { every } = await import("./dashboard/api.js");
let active = 0, most = 0, calls = 0;
const stop = every(20, async () => {
  calls += 1; active += 1; most = Math.max(most, active);
  await wait(60);  // slower than the interval, like a stalled /api/status
  active -= 1;
});
for (let i = 0; i < 5; i++) {  // the tab is shown again while a request is still out
  await wait(5);
  for (const fn of doc.listeners.visibilitychange) fn();
}
await wait(300);
stop();
const after = calls;
await wait(100);
assert.equal(most, 1, "two polls overlapped");
assert.ok(calls >= 3, `only ${calls} polls`);
assert.equal(calls, after, "polled after stop()");
""",
    )


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
assert.equal($("conn-text").textContent, "Connected (updates every few seconds)");
for (const [id, type] of [["ask-form", "submit"], ["listen-btn", "click"], ["file-input", "change"],
                          ["search-form", "submit"], ["calibrate-btn", "click"]]) {
  assert.ok($(id).listeners[type]?.length, `${id} has no ${type} handler`);
}
const paths = http.calls.map((c) => c.path);
for (const p of ["/api/status", "/api/documents", "/api/config"]) assert.ok(paths.includes(p), `${p} never fetched`);
// With no socket, the status poll drives the status line and the push-to-talk note.
assert.equal($("status-line").dataset.state, "listening");
assert.equal($("status-text").textContent, LISTENING);
await $("listen-btn").fire("click");
assert.equal($("ask-msg").textContent, LISTENING);
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
assert.equal(msg.textContent, "Getting the microphone ready…");
sock.fire("status", { state: "idle" });  // e.g. the previous answer finishing first
assert.equal(msg.textContent, "Getting the microphone ready…");
sock.fire("status", { state: "listening" });
assert.equal(msg.textContent, LISTENING);
sock.fire("status", { state: "transcribing" });
assert.equal(msg.textContent, "Writing down what you said…");
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
assert.equal($("ask-msg").textContent, LISTENING);
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
assert.equal($("ask-msg").textContent, "Sent! Watch the board for the answer.");
""",
    )


@needs_node
def test_file_list_explains_which_files_are_used(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + """
installIo();
installFetch({
  "GET /api/documents": {
    status: 200,
    body: {
      documents: [
        { source: "protocol.pdf", type: "pdf", chunks: 12, active: true },
        { source: "old_notes.md", type: "md", chunks: 1, active: false },
      ],
      total_chunks: 13, active_count: 1, all_active: false,
    },
  },
});
await import("./dashboard/main.js");
await settle();
assert.equal($("files-summary").textContent, "The assistant uses 1 of 2 files when answering.");
assert.equal($("use-all-btn").hidden, false);
assert.equal($("files").children.length, 2);
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
assert.match($("ask-msg").textContent, /^No microphone found\\./);
assert.match($("ask-msg").textContent, /type your question/);
assert.equal($("ask-msg").dataset.tone, "bad");
""",
    )


PROJECTOR = """
const doc = installDom(NODES, { bodyData: { projectorWidth: "1920", projectorHeight: "1080" } });
const $ = (id) => doc.getElementById(id);
const sock = installIo();
installFetch();
await import("./main.js");
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
assert.equal($("status-label").textContent, "Thinking…");
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
assert.equal($("status-label").textContent, "Not connected");
sock.fire("connect");
assert.equal($("calibration-img"), null);
assert.equal($("transcript").textContent, "");
assert.ok($("transcript").classList.contains("hidden"));
assert.ok(!doc.body.classList.contains("offline"));
// The pre-disconnect status is stale, so idle stands in until the server's push arrives.
assert.equal($("status-label").textContent, "Ready – press Space to talk");
sock.fire("status", { state: "speaking" });
assert.equal($("status-label").textContent, "Answering…");
""",
    )


@needs_node
def test_projector_corner_uses_plain_words(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "index.html",
        PROJECTOR
        + """
const label = () => $("status-label").textContent;
sock.fire("connect");
for (const [state, words] of [["idle", "Ready – press Space to talk"], ["listening", "Listening…"],
                              ["transcribing", "Writing it down…"], ["thinking", "Thinking…"], ["speaking", "Answering…"]]) {
  sock.fire("status", { state });
  assert.equal(label(), words);
}
sock.fire("status", { state: "thinking", detail: "busy" });
assert.equal(label(), "Still busy – wait a moment");
const sent = sock.emitted.length;
sock.fire("disconnect");
const space = { code: "Space", key: " ", repeat: false, preventDefault() {} };
for (const fn of doc.listeners.keydown) fn(space);
assert.equal(label(), "Not connected – question not sent");
await settle();
// Nothing at all while disconnected: no question, and no UI boxes either.
assert.deepEqual(sock.emitted.slice(sent), []);
""",
    )


QUIZ = '{"active": true, "number": 2, "total": 8, "score": 1, "problem": "3x + 5 = 20"}'


@needs_node
def test_projector_quiz_card_follows_tutor_events(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "index.html",
        PROJECTOR
        + f"const QUIZ = {QUIZ};\n"
        + """
const card = $("problem-card");
installFetch({ "GET /api/tutor": { status: 200, body: { mode: "quiz", learn_topic: null, quiz: QUIZ } } });
sock.fire("connect");
await settle();
// A reconnect asks the server, so a quiz that started while the page was away still shows.
assert.ok(!card.classList.contains("hidden"));
assert.equal($("problem-card-label").textContent, "Quiz · problem 2 of 8");
assert.equal($("problem-card-text").textContent, "3x + 5 = 20");
// Made-up problems have no text: the card keeps the count only.
sock.fire("tutor", { mode: "quiz", learn_topic: null, quiz: { ...QUIZ, number: 3, problem: null } });
assert.equal($("problem-card-label").textContent, "Quiz · problem 3 of 8");
assert.equal($("problem-card-text").textContent, "");
sock.fire("tutor", { mode: "help", learn_topic: null, quiz: null });
assert.ok(card.classList.contains("hidden"));
sock.fire("tutor", { mode: "quiz", learn_topic: null, quiz: { ...QUIZ, active: false } });
assert.ok(card.classList.contains("hidden"));
""",
    )


@needs_node
def test_projector_problem_card_shows_dictated_and_board_problems(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "index.html",
        PROJECTOR
        + """
const card = $("problem-card");
sock.fire("connect");
await settle();
sock.fire("tutor", { mode: "practice", learn_topic: null, quiz: null, problem: { text: "4x + 7 = 19", source: "dictated" } });
assert.ok(!card.classList.contains("hidden"));
assert.equal($("problem-card-label").textContent, "Try this");
assert.equal($("problem-card-text").textContent, "4x + 7 = 19");
sock.fire("tutor", { mode: "help", learn_topic: null, quiz: null, problem: { text: "<b>2x</b> = 4", source: "board" } });
assert.equal($("problem-card-label").textContent, "Your problem");
assert.equal($("problem-card-text").textContent, "<b>2x</b> = 4", "set as text, never as HTML");
// The labels follow the language the assistant answers in, like the quiz label.
sock.fire("language", { setting: "es", current: "es" });
assert.equal($("problem-card-label").textContent, "Tu problema");
sock.fire("tutor", { mode: "practice", learn_topic: null, quiz: null, problem: { text: "4x + 7 = 19", source: "dictated" } });
assert.equal($("problem-card-label").textContent, "Prueba este");
sock.fire("language", { setting: "en", current: "en" });
assert.equal($("problem-card-label").textContent, "Try this");
sock.fire("tutor", { mode: "help", learn_topic: "fractions", quiz: null, problem: null });
assert.ok(card.classList.contains("hidden"));
""",
    )


@needs_node
def test_projector_quiz_card_ignores_a_stale_get(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "index.html",
        PROJECTOR
        + f"const QUIZ = {QUIZ};\n"
        + """
let reply;
installFetch({ "GET /api/tutor": () => new Promise((resolve) => { reply = resolve; }) });
sock.fire("connect");
await settle();
sock.fire("tutor", { mode: "help", learn_topic: null, quiz: null });
reply({ status: 200, body: { mode: "quiz", learn_topic: null, quiz: QUIZ } });
await settle();
assert.ok($("problem-card").classList.contains("hidden"));
""",
    )


# ---- dashboard: connection, tutor modes, wording ------------------------------------


@needs_node
def test_dashboard_says_connecting_until_the_socket_fails(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + """
const sock = installIo();
installFetch();
await import("./dashboard/main.js");
await settle();
// The template's "Connecting…" stays: nothing has failed yet.
assert.equal($("status-text").textContent, "Connecting…");
assert.equal($("conn-text").textContent, "Connecting…");
assert.notEqual($("badge-ask").textContent, "not connected");
sock.fire("connect_error", { message: "xhr poll error" });
sock.fire("connect_error", { message: "xhr poll error" });
assert.match($("status-text").textContent, /^Can't reach the assistant\\. Check it is still running \\(restart the assistant/);
assert.equal($("badge-ask").textContent, "not connected");
assert.equal(logText().match(/connect_error/g).length, 1);
sock.fire("connect");
assert.equal($("conn-text").textContent, "Connected");
assert.equal($("status-text").textContent, "Ready. Ask a question below, or on the board.");
""",
    )


@needs_node
@pytest.mark.parametrize("connects", [False, True])
def test_dashboard_offline_only_after_the_grace_period(tmp_path: Path, connects: bool) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + f"const CONNECTS = {json.dumps(connects)};\n"
        + """
const sock = installIo();
installFetch();
await import("./dashboard/main.js");
if (CONNECTS) sock.fire("connect");
await wait(3200);
assert.equal($("conn-text").textContent, CONNECTS ? "Connected" : "Not connected");
""",
    )


@needs_node
def test_mode_picker_sets_and_follows_the_tutor_mode(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + """
const sock = installIo();
const http = installFetch({
  "PUT /api/tutor/mode": (init) => ({ status: 200, body: { mode: JSON.parse(init.body).mode, learn_topic: null, quiz: null } }),
});
await import("./dashboard/main.js");
sock.fire("connect");
await settle();
const pressed = () => ["help", "practice", "quiz"].filter((m) => $(`mode-${m}`).getAttribute("aria-pressed") === "true");
assert.equal($("modes").hidden, false);
assert.deepEqual(pressed(), ["help"]);
assert.equal($("badge-ask").textContent, "step-by-step help");
assert.equal($("quiz-bar").hidden, true);
await $("mode-practice").fire("click");
const put = http.calls.find((c) => c.method === "PUT" && c.path === "/api/tutor/mode");
assert.deepEqual(JSON.parse(put.body), { mode: "practice" });
assert.deepEqual(pressed(), ["practice"]);
assert.equal($("badge-ask").textContent, "similar problem");
assert.equal($("ask-msg").textContent, "Now write a problem on the board and ask for one like it.");
assert.equal($("mode-practice").disabled, false);
// The student says "quiz me": the server pushes the new mode and the picker follows.
sock.fire("tutor", { mode: "quiz", learn_topic: null, quiz: null });
assert.deepEqual(pressed(), ["quiz"]);
assert.equal($("quiz-bar").hidden, false);
assert.equal($("quiz-start-btn").hidden, false);
assert.equal($("quiz-stop-btn").hidden, true);
assert.match($("quiz-progress").textContent, /worksheets you tick/);
// Busy states still win the badge; idle shows the mode again.
sock.fire("status", { state: "thinking" });
assert.equal($("badge-ask").textContent, "thinking…");
sock.fire("status", { state: "idle" });
assert.equal($("badge-ask").textContent, "quiz");
""",
    )


@needs_node
def test_mode_picker_failure_is_explained_and_logged(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + """
const sock = installIo();
installFetch({ "PUT /api/tutor/mode": { status: 500, body: { error: "KeyError: mode" } } });
await import("./dashboard/main.js");
sock.fire("connect");
await settle();
await $("mode-quiz").fire("click");
assert.equal($("ask-msg").textContent,
  "Couldn't change how the assistant helps. Check the assistant is still running (restart the assistant if it closed), then try again.");
assert.equal($("ask-msg").dataset.tone, "warn");
assert.match(logText(), /KeyError: mode/);
assert.equal($("mode-help").getAttribute("aria-pressed"), "true");
""",
    )


@needs_node
def test_quiz_bar_shows_progress_problem_and_stop(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + f"const QUIZ = {QUIZ};\n"
        + """
const sock = installIo();
const http = installFetch({
  "POST /api/tutor/quiz/start": { status: 200, body: { mode: "quiz", learn_topic: null, quiz: { ...QUIZ, number: 1, score: 0 } } },
  "POST /api/tutor/quiz/stop": { status: 200, body: { mode: "help", learn_topic: null, quiz: null } },
});
await import("./dashboard/main.js");
sock.fire("connect");
await settle();
sock.fire("tutor", { mode: "quiz", learn_topic: null, quiz: null });
await $("quiz-start-btn").fire("click");
assert.equal($("quiz-progress").textContent, "Problem 1 of 8 · 0 right on the first try");
assert.equal($("quiz-problem").textContent, "3x + 5 = 20");
assert.equal($("quiz-problem").hidden, false);
assert.equal($("quiz-start-btn").hidden, true);
assert.equal($("quiz-stop-btn").hidden, false);
assert.match($("ask-msg").textContent, /^Quiz started! Say or type “I'm ready”/);
sock.fire("tutor", { mode: "quiz", learn_topic: "equations", quiz: QUIZ });
// The server's score only counts first-try answers; "so far" read like a bug after a fixed answer.
assert.equal($("quiz-progress").textContent, "Problem 2 of 8 · 1 right on the first try");
assert.equal($("badge-ask").textContent, "quiz · 2 of 8");
assert.equal($("learn-line").textContent, "Learning: equations");
assert.equal($("learn-line").hidden, false);
sock.fire("tutor", { mode: "quiz", learn_topic: null, quiz: { ...QUIZ, number: 3, total: 10, problem: null } });
assert.equal($("quiz-problem").hidden, true);
assert.equal($("learn-line").hidden, true);
await $("quiz-stop-btn").fire("click");
assert.ok(http.calls.some((c) => c.method === "POST" && c.path === "/api/tutor/quiz/stop"));
assert.equal($("quiz-bar").hidden, true);
assert.equal($("ask-msg").textContent, "Quiz stopped.");
assert.equal($("badge-ask").textContent, "step-by-step help");
// Back in quiz mode after a quiz that got somewhere: the bar reports how it went.
sock.fire("tutor", { mode: "quiz", learn_topic: null, quiz: { ...QUIZ, active: false, score: 5 } });
assert.equal($("quiz-progress").textContent, "Last quiz: 5 right on the first try. Press Start quiz to go again.");
assert.equal($("quiz-start-btn").hidden, false);
""",
    )


@needs_node
def test_math_examples_and_wording_by_default(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + """
installIo();
const http = installFetch();
await import("./dashboard/main.js");
await settle();
const examples = $("examples").children.slice(1);
assert.deepEqual(examples.map((b) => b.textContent),
  ["I want to learn how to solve fractions", "Help me with the next step", "Quiz me on equations"]);
assert.equal($("badge-files").textContent, "add worksheets");
assert.match($("files-summary").textContent, /^No worksheets or notes yet\\./);
await examples[0].fire("click");
await settle();
const query = http.calls.find((c) => c.method === "POST" && c.path === "/api/query");
assert.deepEqual(JSON.parse(query.body), { text: "I want to learn how to solve fractions" });
""",
    )


@needs_node
def test_biology_profile_hides_modes_and_keeps_its_examples(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + """
const sock = installIo();
installFetch({ "GET /api/config": { status: 200, body: { profile: "biology", listen_mode: "push", wake_word: "assistant" } } });
await import("./dashboard/main.js");
sock.fire("connect");
await settle();
assert.equal($("modes").hidden, true);
assert.equal($("quiz-bar").hidden, true);
assert.equal($("badge-ask").textContent, "ready");
assert.deepEqual($("examples").children.slice(1).map((b) => b.textContent), [
  "Is my flowchart missing a sterile step?",
  "Why does Leptosol runoff make the water cloudy?",
  "Where should I take my water samples?",
]);
assert.equal($("files-tab-name").textContent, "Study files");
assert.equal($("files-hint").textContent,
  "Add your notes, protocols or papers. Before answering, the assistant looks in the files you tick. Untick a file to leave it out.");
assert.equal($("badge-files").textContent, "add your notes");
assert.equal($("ask-input").placeholder, "Ask about your drawing or your notes…");
""",
    )


UNREACHABLE = "I couldn't reach the AI model. Please check that it is running."
HISTORY = [
    # Ollama not running while the board has ink: a failed call, not a mistake in the student's work.
    {"query": "check my answer", "spoken_text": UNREACHABLE, "animation_state": "error", "ink_regions": 2,
     "had_frame": True, "latency_s": 3.2, "mode": "quiz", "task_kind": "quiz", "sources": []},
    # A wrong quiz answer said out loud, empty board and no camera.
    {"query": "is it 5?", "spoken_text": "Not quite. What undoes adding five?", "animation_state": "error",
     "ink_regions": 0, "had_frame": False, "latency_s": 2.6, "mode": "quiz", "task_kind": "quiz", "sources": []},
    # A server that says so: `failed` wins over any guess from the text or the board.
    {"query": "help", "spoken_text": "Hmm.", "animation_state": "error", "ink_regions": 1, "had_frame": True,
     "latency_s": 1, "mode": "help", "task_kind": None, "sources": [], "failed": True},
    {"query": "is this right?", "spoken_text": "The second line drops the minus sign.", "animation_state": "error",
     "ink_regions": 1, "had_frame": True, "latency_s": 1, "mode": "help", "task_kind": None, "sources": [],
     "failed": False},
    {"query": "teach me fractions", "spoken_text": "…", "animation_state": "idle", "ink_regions": 0,
     "had_frame": True, "latency_s": 2, "mode": "help", "task_kind": "learn", "sources": ["guide.md"]},
    {"query": "next step?", "spoken_text": "…", "animation_state": "pointing", "ink_regions": 1, "had_frame": True,
     "latency_s": 4, "mode": "help", "task_kind": None, "sources": []},
]  # fmt: skip


@needs_node
@pytest.mark.parametrize("profile", ["math", "biology"])
def test_margin_notes_name_the_mode_and_what_went_wrong(tmp_path: Path, profile: str) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + f"const ITEMS = {json.dumps(HISTORY)};\nconst PROFILE = {json.dumps(profile)};\n"
        + """
installIo();
installFetch({
  "GET /api/history": { status: 200, body: { interactions: ITEMS } },
  "GET /api/config": { status: 200, body: { profile: PROFILE } },
});
await import("./dashboard/main.js");
await settle();
const notes = $("timeline").children.map((li) => li.children.find((c) => c.className === "margin-note").textContent);
const math = PROFILE === "math";
const failed = "something went wrong, so this isn't a real answer";
assert.deepEqual(notes, [
  (math ? "quiz · " : "") + `${failed} · didn't use your files · took 3 s`,
  (math ? "quiz · " : "") + "that answer isn't right yet · didn't use your files · couldn't see the board · took 3 s",
  (math ? "step-by-step help · " : "") + `${failed} · didn't use your files · took 1 s`,
  (math ? "step-by-step help · found a mistake in your work" : "found a mistake in your drawing") + " · didn't use your files · took 1 s",
  (math ? "learning something new · " : "") + "read guide.md · took 2 s",
  (math ? "step-by-step help · pointed at your work" : "pointed at your drawing") + " · didn't use your files · took 4 s",
]);
""",
    )


@needs_node
def test_file_rows_name_photos_and_have_unique_ids(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + """
installIo();
installFetch({
  "GET /api/documents": {
    status: 200,
    body: {
      documents: [
        { source: "worksheet photo.jpg", type: "image", chunks: 2, active: true },
        { source: "scan", type: "image", chunks: 1, active: true },
        { source: "unit 1.md", type: "md", chunks: 3, active: true },
        { source: "unit-1.md", type: "md", chunks: 3, active: false },
      ],
    },
  },
});
await import("./dashboard/main.js");
await settle();
const rows = $("files").children;
const meta = rows.map((li) => li.children[1].children[0].textContent);
assert.deepEqual(meta, ["Photo (text read from the image) · 2 sections", "Photo (text read from the image) · 1 section",
                        "Notes · 3 sections", "Notes · 3 sections"]);
const ids = rows.map((li) => li.children[0].id);
assert.equal(new Set(ids).size, 4);
for (const li of rows) assert.equal(li.children[1].htmlFor, li.children[0].id);
assert.equal($("files-summary").textContent, "The assistant uses 3 of 4 files when answering.");
""",
    )


@needs_node
def test_camera_polling_waits_for_its_tab(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + """
installIo();
const http = installFetch();
const { selectStep } = await import("./dashboard/tabs.js");
$("ink-toggle").checked = true;  // as the template ships it
await import("./dashboard/main.js");
await settle();
// The Ask tab opens first: neither the picture nor the drawing count is on screen.
assert.equal($("panel-board").hidden, true);
assert.equal($("frame").src, "");
assert.ok(!http.calls.some((c) => c.path === "/api/ink"));
selectStep("board");
await wait(1700);
assert.match($("frame").src, /^\\/api\\/frame\\.jpg\\?warped=1/);
assert.ok(http.calls.some((c) => c.path === "/api/ink"));
""",
    )


@needs_node
def test_settings_use_plain_words(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + """
installIo();
installFetch();
const { renderConfig } = await import("./dashboard/log.js");
const rows = () => {
  const items = $("config").children;
  const out = {};
  for (let i = 0; i < items.length; i += 2) out[items[i].textContent] = items[i + 1].textContent;
  return out;
};
renderConfig({ profile: "math", listen_mode: "push", wake_word: "assistant", tts_backend: "edge", checkerboard: [9, 6] });
assert.deepEqual(rows(), {
  "Subject": "Math",
  "Microphone mode": "Press Talk (or Space on the projector page)",
  "Voice": "Online voice (needs internet)",
  "Checkerboard inner corners": "9 × 6",
});
renderConfig({ listen_mode: "wake", wake_word: "hey board", tts_backend: "none" });
assert.deepEqual(rows(), {
  "Microphone mode": "Always listening: say “hey board” first",
  "Wake word": "hey board",
  "Voice": "Off (answers are only written)",
});
renderConfig({ language: "auto", current_language: "es" });
assert.deepEqual(rows(), {
  "Answer language": "Auto (the language the question is asked in)",
  "Answering in right now": "Spanish",
});
""",
    )


@needs_node
def test_activity_log_reads_as_words_not_json(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + """
const sock = installIo();
installFetch();
const { describe } = await import("./dashboard/log.js");
await import("./dashboard/main.js");
sock.fire("connect");
sock.fire("language", { setting: "auto", current: "es" });
sock.fire("status", { state: "thinking", detail: "busy" });
assert.match(logText(), /language setting: auto · current: es/);
assert.match(logText(), /status state: thinking · detail: busy/);
assert.doesNotMatch(logText(), /[{}"]/);
assert.equal(describe({ mode: "quiz", learn_topic: null, quiz: { number: 2 }, confirm_stop: false }),
  'mode: quiz · quiz: {"number":2} · confirm stop: false');
assert.equal(describe("plain text"), "plain text");
assert.equal(describe(null), "");
assert.equal(describe([1, 2]), "[1,2]");
""",
    )


@needs_node
def test_checklist_explains_camera_model_and_voice(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + """
installIo();
installFetch();
const { renderStatus } = await import("./dashboard/status.js");
const base = { camera_running: true, calibrated: true, total_documents: 1, active_documents: 1, audio_available: true };
const checks = () => $("checks").children.map(text).join("\\\\n");
renderStatus({ ...base, camera_running: false, vlm_backend: "mock", vlm_model: "mock", tts_backend: "edge" });
assert.match(checks(), /Camera not found .*not open in another app \\(Zoom, Teams, FaceTime\\)\\. On a Mac, .*On Windows, .*Then restart the assistant\\./);
assert.match(checks(), /Demo mode: answers are pretend Ask whoever set up the assistant/);
assert.match(checks(), /Answers are read aloud on the projector page \\(click it once to allow sound\\)/);
renderStatus({ ...base, vlm_backend: "ollama", vlm_model: "qwen2.5vl:3b", tts_backend: "none" });
assert.match(checks(), /AI model chosen: qwen2\\.5vl:3b If an answer says it couldn't reach the AI model, start Ollama and ask again\\./);
assert.doesNotMatch(checks(), /AI model ready/);
// A server that says which computer it runs on gets only that computer's steps.
renderStatus({ ...base, camera_running: false, host_os: "windows" });
assert.match(checks(), /Camera not found .*“Let desktop apps access your camera”.*restart the assistant\\./);
assert.doesNotMatch(checks(), /System Settings|On a Mac/);
renderStatus({ ...base, camera_running: false, host_os: "mac" });
assert.match(checks(), /Camera not found .*System Settings → Privacy & Security → Camera, and restart the assistant\\./);
assert.doesNotMatch(checks(), /Let desktop apps|On Windows/);
renderStatus({ ...base, camera_running: false, host_os: "other" });
assert.match(checks(), /Camera not found Check the camera is plugged in and not open in another app, then restart the assistant\\./);
""",
    )


@needs_node
def test_calibration_errors_point_at_the_projector_page(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + """
installIo();
const replies = [
  { status: 200, body: { ok: false, error: "checkerboard not detected; check camera view and lighting" } },
  { status: 500, body: { error: "cv2.error: (-215) assertion failed" } },
  { status: 200, body: { ok: true, error: "PermissionError: data/calibration.json" } },
];
installFetch({ "POST /api/calibrate": () => replies.shift() });
await import("./dashboard/main.js");
const msg = $("calib-msg");
await $("calibrate-btn").fire("click");
assert.match(msg.textContent, /^Couldn't see the checkerboard\\. Check the projector page is open on the projector in full screen/);
await $("calibrate-btn").fire("click");
assert.match(msg.textContent, /^Couldn't line up\\. Check the projector page is open on the projector in full screen/);
assert.doesNotMatch(msg.textContent, /cv2/);
await $("calibrate-btn").fire("click");
assert.equal(msg.textContent, "Lined up, but it couldn't be saved, so do this again after restarting the assistant.");
assert.match(logText(), /assertion failed/);
assert.match(logText(), /PermissionError/);
""",
    )


@needs_node
def test_error_notes_are_friendly_and_raw_detail_goes_to_the_log(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + """
const sock = installIo();
installFetch({
  "POST /api/query": { status: 500, body: { error: "RuntimeError: boom" } },
  "POST /api/listen": { status: 500, body: { error: "OSError: -9996" } },
  "GET /api/search": { status: 500, body: { error: "chromadb exploded" } },
  "PUT /api/documents/selection": { status: 500, body: { error: "disk full" } },
  "GET /api/documents": { status: 200, body: { documents: [{ source: "a.pdf", type: "pdf", chunks: 1, active: true }] } },
  "POST /api/upload": { status: 200, body: { files: [
    { name: "b.pdf", chunks: 0, error: "PDF is encrypted" },
    { name: "scan.pdf", chunks: 0, error: "no readable text found (scanned PDF or blank image?)" },
  ], total_chunks: 0 } },
});
await import("./dashboard/main.js");
sock.fire("connect");
await settle();
const hint = "Check the assistant is still running (restart the assistant if it closed), then try again.";
$("ask-input").value = "help";
await $("ask-form").fire("submit");
assert.equal($("ask-msg").textContent, `Couldn't send your question. ${hint}`);
await $("listen-btn").fire("click");
assert.equal($("ask-msg").textContent, "The microphone isn't working right now. Try Talk again, or type your question.");
$("search-input").value = "fractions";
await $("search-form").fire("submit");
await settle();
assert.equal(text($("hits").children[0]), `Search didn't work. ${hint}`);
await $("files").children[0].children[0].fire("change");
await settle();
assert.equal($("upload-msg").textContent, `Couldn't save which files are ticked. ${hint}`);
$("file-input").files = [new File(["x"], "b.pdf")];
await $("file-input").fire("change");
await settle();
// Saving a scan again as a PDF doesn't add text to it, so that file gets its own advice.
assert.equal($("upload-msg").textContent,
  "Couldn't read b.pdf (try saving it again as a PDF or text file); " +
  "scan.pdf (it looks like a scan with no text, so type it up or upload a clearer photo).");
for (const raw of ["RuntimeError: boom", "OSError: -9996", "chromadb exploded", "disk full", "PDF is encrypted"]) {
  assert.ok(logText().includes(raw), `${raw} missing from the activity log`);
}
""",
    )


# ---- dashboard: subject, files, notes ------------------------------------------------


@needs_node
def test_subject_known_before_the_file_list_keeps_the_ask_tab(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + """
installIo();
let reply;
installFetch({
  "GET /api/documents": () => new Promise((resolve) => { reply = resolve; }),
  "GET /api/config": { status: 200, body: { profile: "biology" } },
});
await import("./dashboard/main.js");
await settle();
// /api/config answered first: the page's empty placeholder list is not "this student has no files".
assert.equal($("panel-ask").hidden, false);
assert.equal($("panel-files").hidden, true);
reply({ status: 200, body: { documents: [{ source: "protocol.pdf", type: "pdf", chunks: 3, active: true }] } });
await settle();
assert.equal($("badge-files").textContent, "1 in use");
assert.equal($("panel-ask").hidden, false);
assert.equal($("files-summary").textContent, "The assistant uses all 1 file when answering.");
""",
    )


@needs_node
def test_subject_follows_a_server_restarted_with_another_profile(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + """
const sock = installIo();
let profile = "math";
installFetch({ "GET /api/config": () => ({ status: 200, body: { profile } }) });
await import("./dashboard/main.js");
sock.fire("connect");
await settle();
assert.equal($("modes").hidden, false);
// Restarted with SAR_PROFILE=biology while this page stayed open: the reconnect asks again.
profile = "biology";
sock.fire("disconnect");
sock.fire("connect");
await settle();
assert.equal($("modes").hidden, true);
assert.equal($("files-tab-name").textContent, "Study files");
assert.equal($("examples").children[1].textContent, "Is my flowchart missing a sterile step?");
""",
    )


@needs_node
def test_status_poll_sets_the_subject_when_config_fails(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + """
installIo();
const http = installFetch({ "GET /api/config": { status: 500, body: { error: "boom" } } });
http.setStatus({ profile: "biology" });
await import("./dashboard/main.js");
await settle();
assert.equal($("files-tab-name").textContent, "Study files");
assert.equal($("modes").hidden, true);
// Still retried, so the settings list fills in once /api/config works.
await wait(2300);
assert.ok(http.calls.filter((c) => c.path === "/api/config").length >= 2);
""",
    )


@needs_node
def test_mode_picker_waits_until_the_subject_is_known(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + """
const sock = installIo();
const answers = [];
installFetch({ "GET /api/config": () => new Promise((resolve) => answers.push(resolve)) });
await import("./dashboard/main.js");
sock.fire("connect");
await settle();
// /api/tutor answered but the subject didn't yet: a biology server must never show the math picker.
assert.equal($("modes").hidden, true);
for (const answer of answers) answer({ status: 200, body: { profile: "math" } });
await settle();
assert.equal($("modes").hidden, false);
assert.equal($("badge-ask").textContent, "step-by-step help");
""",
    )


@needs_node
def test_mode_message_keeps_the_microphone_watchdog(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + """
const sock = installIo();
installFetch({
  "PUT /api/tutor/mode": (init) => ({ status: 200, body: { mode: JSON.parse(init.body).mode, learn_topic: null, quiz: null } }),
});
// Capture the 10 s "the microphone didn't start" watchdog instead of waiting for it.
const watchdogs = [];
const realSetTimeout = globalThis.setTimeout;
const realClearTimeout = globalThis.clearTimeout;
globalThis.setTimeout = (fn, ms, ...args) => {
  if (ms !== 10000) return realSetTimeout(fn, ms, ...args);
  const token = { fn };
  watchdogs.push(token);
  return token;
};
globalThis.clearTimeout = (token) => {
  const i = watchdogs.indexOf(token);
  if (i >= 0) watchdogs.splice(i, 1); else realClearTimeout(token);
};
await import("./dashboard/main.js");
sock.fire("connect");
await settle();
await $("listen-btn").fire("click");
assert.equal($("ask-msg").textContent, "Getting the microphone ready…");
await $("mode-practice").fire("click");
assert.equal($("ask-msg").textContent, "Now write a problem on the board and ask for one like it.");
assert.equal(watchdogs.length, 1, "picking a mode cancelled the microphone watchdog");
watchdogs[0].fn();
assert.equal($("ask-msg").textContent, "The microphone didn't start. Try Talk again, or type your question.");
""",
    )


@needs_node
def test_math_answers_show_normal_notation_when_the_server_sends_it(tmp_path: Path) -> None:
    item = {**HISTORY[-1], "spoken_text": "Subtract five from both sides of three x plus five equals twenty.",
            "display_text": "Subtract 5 from both sides of 3x + 5 = 20."}  # fmt: skip
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + f"const ITEMS = {json.dumps([item, HISTORY[-2]])};\n"
        + """
installIo();
installFetch({ "GET /api/history": { status: 200, body: { interactions: ITEMS } } });
await import("./dashboard/main.js");
await settle();
const answers = $("timeline").children.map((li) => li.children.find((c) => c.className === "a").textContent);
// Older servers send no display_text: the spoken words are shown as before.
assert.deepEqual(answers, ["Subtract 5 from both sides of 3x + 5 = 20.", "…"]);
""",
    )


# ---- projector: problem card and bubble -------------------------------------------------


@needs_node
def test_projector_quiz_card_steps_aside_while_the_board_is_read(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "index.html",
        PROJECTOR
        + f"const QUIZ = {QUIZ};\n"
        + """
const card = $("problem-card");
sock.fire("connect");
sock.fire("tutor", { mode: "quiz", learn_topic: null, quiz: QUIZ });
await settle();
assert.ok(!card.classList.contains("hidden"));
for (const state of ["listening", "transcribing", "thinking"]) {
  sock.fire("status", { state });
  assert.ok(card.classList.contains("capturing"), `card still in the camera picture while ${state}`);
}
sock.fire("status", { state: "speaking" });
assert.ok(!card.classList.contains("capturing"));
assert.ok(!card.classList.contains("hidden"));
sock.fire("status", { state: "idle" });
assert.ok(!card.classList.contains("capturing"));
""",
    )


def test_projector_quiz_card_is_faint_and_hidden_while_capturing() -> None:
    css = (STATIC / "css" / "style.css").read_text(encoding="utf-8")
    rule = re.search(r"#problem-card\s*\{([^}]*)\}", css)
    assert rule, "no #problem-card rule"
    body = rule.group(1)
    # A glow or a thick bright outline is picked up as ink by find_ink_regions and read by the VLM.
    assert "box-shadow" not in body and "text-shadow" not in body
    border = re.search(r"border:\s*([\d.]+)px", body)
    assert border and float(border.group(1)) <= 2
    width = re.search(r"max-width:\s*([\d.]+)vw", body)
    assert width and float(width.group(1)) <= 30
    assert re.search(r"#problem-card\.capturing\s*\{[^}]*visibility:\s*hidden", css)


@needs_node
def test_projector_bubble_prefers_normal_notation(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "index.html",
        PROJECTOR
        + """
sock.fire("connect");
const words = "Subtract five from both sides of three x plus five equals twenty.";
const notation = "Subtract 5 from both sides of 3x + 5 = 20.";
sock.fire("avatar_update", { animation_state: "pointing", target_x: 900, target_y: 400, spoken_text: words, display_text: notation });
assert.equal($("bubble").textContent, notation);
sock.fire("speak", { url: null, text: words, display_text: notation });
assert.equal($("bubble").textContent, notation);
sock.fire("speak", { url: null, text: words });
assert.equal($("bubble").textContent, words);
// The final contract names the spoken words spoken_text in speak events too.
sock.fire("speak", { url: null, spoken_text: words });
assert.equal($("bubble").textContent, words);
""",
    )


# ---- language: one dictionary, the switch, Spanish pages --------------------------------

VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
TEXT_ATTRS = ("placeholder", "alt", "title", "aria-label")
LANGUAGE_FREE = re.compile(r"^[\W\d_]*$")  # numbers, dashes, arrows: the same in every language
SLOT = re.compile(r"\{(\w+)\}")
# The same text on purpose: language names written in their own language, and words Spanish keeps.
SAME_IN_SPANISH = {"lang.en", "lang.es", "lang.auto", "mode.quiz", "files.meta", "board.latency"}


class _Untranslated(HTMLParser):
    """Visible template text (and text attributes) that no data-i18n* key covers."""

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[tuple[str, bool]] = []
        self.missing: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name: value or "" for name, value in attrs}
        for attr in TEXT_ATTRS:
            text = values.get(attr, "")
            if text and not LANGUAGE_FREE.match(text) and f"data-i18n-{attr}" not in values:
                self.missing.append(f'<{tag} {attr}="{text}">')
        if tag not in VOID_TAGS:
            self.stack.append((tag, "data-i18n" in values or bool(self.stack and self.stack[-1][1])))

    def handle_endtag(self, tag: str) -> None:
        while self.stack and self.stack.pop()[0] != tag:
            pass

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text or LANGUAGE_FREE.match(text) or (self.stack and self.stack[-1][0] in {"script", "style"}):
            return
        if not (self.stack and self.stack[-1][1]):
            self.missing.append(text)


def template_source(name: str) -> str:
    # Jinja comments and statements ({% if lan_enabled %}) are not page text.
    source = re.sub(r"\{#.*?#\}", "", (TEMPLATES / name).read_text(encoding="utf-8"), flags=re.S)
    return re.sub(r"\{%.*?%\}", "", source, flags=re.S)


def template_keys(name: str) -> list[str]:
    keys = re.findall(r'\bdata-i18n(?:-[a-z-]+)?="([^"]+)"', template_source(name))
    return [k.replace("{profile}", p) for k in keys for p in (("math", "biology") if "{profile}" in k else ("",))]


@needs_node
def test_dictionary_has_every_text_in_both_languages() -> None:
    en, es = strings()["en"], strings()["es"]
    assert en.keys() == es.keys()
    for key, text in en.items():
        assert text.strip() and es[key].strip(), key
        # "tu único archivo" needs no number: a .one text may leave out {n}.
        slots = set(SLOT.findall(text)) - ({"n"} if key.endswith(".one") else set())
        assert slots <= set(SLOT.findall(es[key])) <= set(SLOT.findall(text)), f"{key}: the {{slots}} differ"
        assert re.findall(r"</?\w+>", text) == re.findall(r"</?\w+>", es[key]), f"{key}: the markup differs"
        if key not in SAME_IN_SPANISH:
            assert es[key] != text, f"{key} is not translated"
    for key in en:
        if key.endswith(".one"):
            assert key.removesuffix(".one") + ".other" in en, f"{key} has no .other"


@needs_node
def test_spanish_is_mexican_classroom_spanish() -> None:
    es = " ".join(strings()["es"].values())
    # Tú, not usted or vosotros; Mexican classroom words (pizarrón, computadora), not Spain's.
    for word in ("usted", "ustedes", "vosotros", "habéis", "pizarra", "ordenador", "vale", "coger", "móvil", "fichero"):
        assert not re.search(rf"\b{word}\b", es, re.I), f"Spanish copy uses '{word}'"
    assert "pizarrón" in es and "presiona" in es
    # Opening marks on every question and exclamation, as a teacher would write them.
    for text in strings()["es"].values():
        assert text.count("?") <= text.count("¿") and text.count("!") <= text.count("¡"), text


@pytest.mark.parametrize("template", ["precharge.html", "index.html", "join.html"])
def test_every_visible_template_text_has_a_dictionary_key(template: str) -> None:
    reader = _Untranslated()
    reader.feed(template_source(template))
    assert reader.missing == []


@needs_node
@pytest.mark.parametrize("template", ["precharge.html", "index.html", "join.html"])
def test_template_english_is_the_dictionary_english(template: str) -> None:
    en, es = strings()["en"], strings()["es"]
    for key in template_keys(template):
        assert key in en and key in es, f"{template} names a missing key {key}"
    src = template_source(template)
    elements = re.findall(r'<(\w+)\b[^>]*?\bdata-i18n="([^"]+)"[^>]*>(.*?)</\1>', src, flags=re.S)
    assert len(elements) == src.count('data-i18n="')
    for tag, key, inner in elements:
        assert f"<{tag}" not in inner, f"nested <{tag}> inside data-i18n={key}"
        assert " id=" not in inner, f"data-i18n={key} would wipe out an element the scripts use"
        assert " ".join(html.unescape(inner).split()) == en[key.replace("{profile}", "math")], key
    for tag in re.findall(r"<\w+\b[^>]*\bdata-i18n-[a-z-]+=[^>]*>", src):
        for attr, key in re.findall(r'\bdata-i18n-([a-z-]+)="([^"]+)"', tag):
            value = re.search(rf'\s{attr}="([^"]*)"', tag)
            assert value and html.unescape(value.group(1)) == en[key.replace("{profile}", "math")], tag


@needs_node
def test_every_key_the_scripts_name_exists() -> None:
    en = strings()["en"]
    prefixes = {key.split(".")[0] for key in en}
    sources = [
        STATIC / "js" / "main.js",
        STATIC / "js" / "join.js",
        *sorted((STATIC / "js" / "dashboard").glob("*.js")),
    ]
    named = 0
    for path in sources:
        for key in re.findall(r'"([a-z]+(?:\.[a-zA-Z]+)+)"', path.read_text(encoding="utf-8")):
            if key.split(".")[0] not in prefixes:
                continue
            named += 1
            assert key in en or f"{key}.one" in en, f"{path.name} names a missing key {key}"
    assert named > 40


@needs_node
def test_language_switch_follows_and_sets_the_server_language(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + """
const sock = installIo();
const http = installFetch({
  "PUT /api/language": (init) => {
    const { language } = JSON.parse(init.body);
    return { status: 200, body: { setting: language, current: language === "auto" ? "en" : language } };
  },
});
await import("./dashboard/main.js");
sock.fire("connect");
await settle();
const pressed = () => ["en", "es", "auto"].filter((s) => $(`lang-${s}`).getAttribute("aria-pressed") === "true");
assert.equal($("lang-switch").hidden, false);
assert.deepEqual(pressed(), ["auto"]);
assert.equal(doc.documentElement.lang, "en");
assert.equal($("status-text").textContent, "Ready. Ask a question below, or on the board.");
await $("lang-es").fire("click");
const put = http.calls.find((c) => c.method === "PUT" && c.path === "/api/language");
assert.deepEqual(JSON.parse(put.body), { language: "es" });
assert.deepEqual(pressed(), ["es"]);
assert.equal(doc.documentElement.lang, "es");
assert.equal($("lang-es").disabled, false);
// Everything already on screen switches: the status line, footer, tabs, placeholders, badges, labels.
assert.equal($("status-text").textContent, "Listo. Haz una pregunta aquí abajo o en el pizarrón.");
assert.equal($("conn-text").textContent, "Conectado");
assert.equal($("files-tab-name").textContent, "Ejercicios y apuntes");
assert.equal($("ask-input").placeholder, "Pregunta sobre lo que escribiste en el pizarrón o sobre lo que quieres aprender…");
assert.equal($("search-input").placeholder, "p. ej. fracciones");
assert.equal($("badge-ask").textContent, "ayuda paso a paso");
assert.equal($("badge-files").textContent, "agrega ejercicios");
assert.equal($("lang-switch").getAttribute("aria-label"), "Idioma");
assert.equal($("lang-auto").title, "Contesta en el mismo idioma en que le preguntan");
assert.equal($("frame").alt, "Lo que ve la cámara en el pizarrón");
// Auto: the page follows the language of the latest answer.
sock.fire("language", { setting: "auto", current: "en" });
assert.deepEqual(pressed(), ["auto"]);
assert.equal(doc.documentElement.lang, "en");
assert.equal($("badge-ask").textContent, "step-by-step help");
sock.fire("language", { setting: "auto", current: "es" });
assert.deepEqual(pressed(), ["auto"]);
assert.equal(doc.documentElement.lang, "es");
assert.equal($("examples").children[1].textContent, "Quiero aprender a resolver fracciones");
""",
    )


@needs_node
def test_language_switch_failure_is_explained_and_changes_nothing(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + """
const sock = installIo();
installFetch({ "PUT /api/language": { status: 400, body: { error: "language must be auto, en or es" } } });
await import("./dashboard/main.js");
sock.fire("connect");
await settle();
await $("lang-es").fire("click");
assert.equal($("lang-msg").textContent,
  "Couldn't change the language. Check the assistant is still running (restart the assistant if it closed), then try again.");
assert.equal($("lang-msg").dataset.tone, "bad");
assert.match(logText(), /language must be auto, en or es/);
assert.equal($("lang-auto").getAttribute("aria-pressed"), "true");
assert.equal(doc.documentElement.lang, "en");
assert.equal($("lang-es").disabled, false);
""",
    )


@needs_node
def test_language_from_an_older_get_never_undoes_a_newer_event(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + """
const sock = installIo();
const replies = [];
installFetch({ "GET /api/language": () => new Promise((resolve) => replies.push(resolve)) });
await import("./dashboard/main.js");
sock.fire("connect");
await settle();
sock.fire("language", { setting: "es", current: "es" });
for (const reply of replies) reply({ status: 200, body: { setting: "auto", current: "en" } });
await settle();
assert.equal(doc.documentElement.lang, "es");
assert.equal($("lang-es").getAttribute("aria-pressed"), "true");
""",
    )


@needs_node
def test_status_poll_sets_the_language_and_older_servers_hide_the_switch(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + """
// No socket.io: the status poll is the only news of the language.
const http = installFetch({ "GET /api/language": { status: 404, body: { error: "not found" } } });
http.setStatus({ language: "auto", current_language: "es" });
await import("./dashboard/main.js");
await settle();
assert.equal(doc.documentElement.lang, "es");
assert.equal($("conn-text").textContent, "Conectado (se actualiza cada pocos segundos)");
assert.equal($("lang-auto").getAttribute("aria-pressed"), "true");
""",
    )
    run_node(
        tmp_path / "older",
        "precharge.html",
        DASHBOARD
        + """
installIo();
const http = installFetch({ "GET /api/language": { status: 404, body: { error: "not found" } } });
http.setStatus({ language: undefined, current_language: undefined });
await import("./dashboard/main.js");
await settle();
assert.equal($("lang-switch").hidden, true);
assert.equal(doc.documentElement.lang, "en");
""",
    )


@needs_node
def test_dashboard_first_paint_uses_the_language_it_showed_last(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + """
const store = { "sar.dashboard.language": "es" };
globalThis.localStorage = { getItem: (k) => store[k] ?? null, setItem: (k, v) => { store[k] = String(v); } };
installIo();
let reply;
const http = installFetch({ "GET /api/language": () => new Promise((resolve) => { reply = resolve; }) });
http.setStatus({ language: undefined, current_language: undefined });
await import("./dashboard/main.js");
await settle();
// Before the server answers: last time's Spanish, not a flash of English.
assert.equal(doc.documentElement.lang, "es");
assert.equal($("status-text").textContent, "Conectando…");
assert.equal($("timeline-empty").children.map(text).join(" ").startsWith("Las respuestas van a aparecer aquí"), true);
reply({ status: 200, body: { setting: "en", current: "en" } });
await settle();
assert.equal(doc.documentElement.lang, "en");
assert.equal(store["sar.dashboard.language"], "en");
""",
    )


SPANISH_HISTORY = [
    {"query": "¿cuál es el siguiente paso?", "spoken_text": "Resta cinco de los dos lados de tres equis más cinco igual a veinte.",
     "display_text": "Resta 5 de los dos lados de 3x + 5 = 20.", "language": "es", "animation_state": "pointing",
     "ink_regions": 1, "had_frame": True, "latency_s": 4, "mode": "help", "task_kind": None, "sources": []},
    {"query": "¿es 5?", "spoken_text": "Casi. ¿Qué deshace sumar cinco?", "language": "es", "animation_state": "error",
     "ink_regions": 0, "had_frame": False, "latency_s": 2.6, "mode": "quiz", "task_kind": "quiz",
     "sources": ["ecuaciones.pdf"], "failed": False},
]  # fmt: skip
# English a Spanish page must not show (user data aside: file names, model names, the activity log).
ENGLISH_WORDS = re.compile(
    r"\b(the|and|your|you|with|files?|ready|connected|camera|answers?|board|step|tick(ed)?|thinking|worksheets?|"
    r"notes|please|try|again|here|press|couldn't|can't|isn't|won't|right|first|glare)\b",
    re.I,
)


@needs_node
def test_dashboard_in_spanish_leaves_no_english_behind(tmp_path: Path) -> None:
    out = run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + f"const ITEMS = {json.dumps(SPANISH_HISTORY)};\nconst QUIZ = {QUIZ};\n"
        + """
const sock = installIo();
const http = installFetch({
  "GET /api/language": { status: 200, body: { setting: "es", current: "es" } },
  "GET /api/history": { status: 200, body: { interactions: ITEMS } },
  "GET /api/config": { status: 200, body: { profile: "math", language: "es", current_language: "es", listen_mode: "push",
                                            tts_backend: "edge", camera_size: [1920, 1080], checkerboard: [9, 6] } },
  "GET /api/documents": { status: 200, body: { documents: [
    { source: "ecuaciones.pdf", type: "pdf", chunks: 3, active: true },
    { source: "foto_tarea.jpg", type: "image", chunks: 1, active: false },
  ] } },
  "GET /api/tutor": { status: 200, body: { mode: "quiz", learn_topic: "ecuaciones", quiz: QUIZ } },
  "POST /api/upload": { status: 200, body: { files: [{ name: "a.pdf", chunks: 2 }, { name: "scan.pdf", chunks: 0,
                                                       error: "no readable text found" }] } },
});
http.setStatus({ language: "es", current_language: "es", calibrated: false, total_documents: 2, active_documents: 1,
                 tts_backend: "edge", last_latency_s: 3.2, uptime_s: 75,
                 glare: { fraction: 0.03, severe: true, hotspots: [[10, 10, 80, 60]] } });
await import("./dashboard/main.js");
sock.fire("connect");
await settle();
sock.fire("status", { state: "listening" });
$("ask-input").value = "ayuda";
await $("ask-form").fire("submit");
$("file-input").files = [new File(["x"], "a.pdf")];
await $("file-input").fire("change");
await settle();
assert.equal(doc.documentElement.lang, "es");
assert.equal($("status-text").textContent,
  "Escuchando… haz tu pregunta ahora. El micrófono se apaga solo en unos 5 segundos.");
assert.equal($("ask-msg").textContent, "¡Enviada! Mira el pizarrón para ver la respuesta.");
assert.equal($("upload-msg").textContent,
  "Se agregó a.pdf. No se pudo leer scan.pdf (parece un escaneo sin texto; escríbelo en la computadora o sube una foto más clara).");
assert.equal($("files-summary").textContent, "El asistente usa 1 de 2 archivos al contestar.");
assert.equal($("badge-board").textContent, "reflejo en el pizarrón");
assert.equal($("quiz-progress").textContent, "Problema 2 de 8 · 1 bien a la primera");
assert.equal($("uptime").textContent, "1 min 15 s");
const entries = $("timeline").children;
const part = (li, cls) => li.children.find((c) => c.className === cls);
// Answers keep their math notation and say which language they are in.
assert.equal(part(entries[0], "a").textContent, "Resta 5 de los dos lados de 3x + 5 = 20.");
assert.equal(part(entries[0], "a").getAttribute("lang"), "es");
assert.equal(part(entries[0], "margin-note").textContent, "ayuda paso a paso · señaló tu trabajo · no usó tus archivos · tardó 4 s");
assert.equal(part(entries[1], "margin-note").textContent,
  "quiz · esa respuesta todavía no está bien · leyó ecuaciones.pdf · no pudo ver el pizarrón · tardó 3 s");
// Sweep everything a student can read, attributes included.
const seen = [];
const walk = (node) => {
  if (node.id === "log") return;
  for (const attr of ["placeholder", "title", "alt"]) if (node[attr]) seen.push(node[attr]);
  if (node.attributes["aria-label"]) seen.push(node.attributes["aria-label"]);
  if (node.textContent) seen.push(node.textContent);
  for (const c of node.children) if (typeof c === "string") seen.push(c); else walk(c);
};
walk(doc.body);
console.log(JSON.stringify(seen));
""",
    )
    seen = json.loads(out.splitlines()[-2])
    assert len(seen) > 80
    for text in seen:
        # The hardware guide is only in English, so its section name stays as written there.
        plain = text.replace("“Checking for glare”", "")
        assert not ENGLISH_WORDS.search(plain), f"English left on the Spanish page: {text!r}"


@needs_node
def test_text_written_by_the_page_switches_language_with_it(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + """
const sock = installIo();
installFetch({
  "GET /api/documents": { status: 200, body: { documents: [{ source: "unit 1.md", type: "md", chunks: 1, active: true }] } },
  "POST /api/calibrate": { status: 200, body: { ok: true } },
});
await import("./dashboard/main.js");
sock.fire("connect");
await settle();
$("ask-input").value = "help";
await $("ask-form").fire("submit");
await $("calibrate-btn").fire("click");
sock.fire("transcript", { text: "what now" });
assert.equal($("files-summary").textContent, "The assistant uses all 1 file when answering.");
sock.fire("language", { setting: "es", current: "es" });
// Not the template's "No files yet." in Spanish: the page wrote these, and they say what they said, in Spanish.
assert.equal($("files-summary").textContent, "El asistente usa tu único archivo al contestar.");
assert.equal($("ask-msg").textContent, "¡Enviada! Mira el pizarrón para ver la respuesta.");
assert.equal($("calib-msg").textContent, "¡Listo, quedó alineado! El asistente ya sabe dónde está el pizarrón.");
assert.equal($("files").children[0].children[1].children[0].textContent, "Apuntes · 1 sección");
const pending = $("timeline").children[0];
assert.deepEqual(pending.children.map((c) => c.textContent), ["Preguntaste", "what now", "Pensando…"]);
""",
    )


@needs_node
def test_projector_follows_the_server_language(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "index.html",
        PROJECTOR
        + f"const QUIZ = {QUIZ};\n"
        + """
installFetch({ "GET /api/language": { status: 200, body: { setting: "auto", current: "es" } } });
sock.fire("connect");
await settle();
const label = () => $("status-label").textContent;
const title = [...doc.body.descendants()].find((n) => n.tagName === "TITLE");
assert.equal(doc.documentElement.lang, "es");
assert.equal(label(), "Listo – presiona Espacio para hablar");
assert.equal(title.textContent, "Asistente del pizarrón – proyector");
assert.equal($("text-input").placeholder, "Escribe una pregunta: Enter para enviar, Esc para cancelar");
assert.equal($("debug-conn").textContent, "conexión: conectada");
sock.fire("tutor", { mode: "quiz", learn_topic: null, quiz: QUIZ });
assert.equal($("problem-card-label").textContent, "Quiz · problema 2 de 8");
sock.fire("status", { state: "thinking", detail: "busy" });
assert.equal(label(), "Ocupado – espera un momento");
// The answer is shown as the model wrote it, whatever the page's language.
sock.fire("avatar_update", { animation_state: "idle", spoken_text: "x is 5", display_text: "x = 5" });
assert.equal($("bubble").textContent, "x = 5");
sock.fire("language", { setting: "en", current: "en" });
assert.equal(doc.documentElement.lang, "en");
assert.equal(label(), "Still busy – wait a moment");
assert.equal($("problem-card-label").textContent, "Quiz · problem 2 of 8");
assert.equal($("bubble").textContent, "x = 5");
sock.fire("language", { setting: "es", current: "es" });
sock.fire("disconnect");
assert.equal(label(), "Sin conexión");
const space = { code: "Space", key: " ", repeat: false, preventDefault() {} };
for (const fn of doc.listeners.keydown) fn(space);
assert.equal(label(), "Sin conexión – no se envió la pregunta");
""",
    )


# ---- board: glare line and the projector-shaped camera box ----------------------------


@needs_node
def test_board_checklist_reports_glare_and_the_badge_follows_it(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + """
installIo();
installFetch();
const { renderStatus } = await import("./dashboard/status.js");
const base = { camera_running: true, calibrated: true, total_documents: 1, active_documents: 1, audio_available: true,
               vlm_backend: "ollama", vlm_model: "qwen2.5vl:3b", tts_backend: "edge" };
const lines = () => $("checks").children;
const glare = () => lines().map(text).find((line) => /glare|bright spot/i.test(line));
const badge = () => [$("badge-board").textContent, $("tab-board").dataset.tone];
renderStatus({ ...base, glare: { fraction: 0, severe: false, hotspots: [] } });
assert.match(text(lines()[1]), /^✓ No glare on the board$/);
assert.deepEqual(badge(), ["ready", "ok"]);
renderStatus({ ...base, glare: { fraction: 0.004, severe: false, hotspots: [[10, 20, 40, 30]] } });
assert.equal(lines()[1].dataset.tone, "warn");
assert.match(glare(), /^! A small bright spot on the board Writing inside it may be missed\\./);
assert.deepEqual(badge(), ["ready", "ok"]);
// Severe glare: what to do, and where the hardware guide explains it. It outranks lining up on the badge.
renderStatus({ ...base, calibrated: false, glare: { fraction: 0.03, severe: true, hotspots: [[10, 20, 400, 300]] } });
assert.equal(lines()[1].dataset.tone, "bad");
assert.match(glare(), /^✗ Glare on the board: move the camera Projector light bounces off the board/);
assert.match(glare(), /Move the webcam further from the projector lens, up or down/);
assert.match(glare(), /“Checking for glare” in the hardware guide \\(docs\\/HARDWARE_SETUP\\.md\\)\\.$/);
assert.deepEqual(badge(), ["glare on the board", "bad"]);
renderStatus({ ...base, calibrated: false, glare: { fraction: 0, severe: false, hotspots: [] } });
assert.deepEqual(badge(), ["not lined up yet", "warn"]);
// No camera: glare is null and the camera line says what to fix; an older server sends no glare at all.
renderStatus({ ...base, camera_running: false, glare: null });
assert.equal(glare(), undefined);
assert.deepEqual(badge(), ["camera not found", "bad"]);
renderStatus({ ...base });
assert.equal(glare(), undefined);
assert.equal(lines().length, 6);
""",
    )


@needs_node
def test_camera_box_has_the_projector_shape_and_the_overlay_sits_on_the_picture(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        """
const doc = installDom(NODES, { bodyData: { projectorWidth: "1280", projectorHeight: "800" } });
const $ = (id) => doc.getElementById(id);
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const sock = installIo();
installFetch({ "GET /api/ink": { status: 200, body: {
  regions: [{ x: 100, y: 200, w: 300, h: 80, cx: 250, cy: 240 }], size: [1280, 800] } } });
const { selectStep } = await import("./dashboard/tabs.js");
const { containedBox } = await import("./dashboard/camera.js");
$("ink-toggle").checked = true;
await import("./dashboard/main.js");
// Epson WXGA, 1280×800: a 16:10 box, so the lined-up picture fills it edge to edge.
assert.equal($("viewport").style.aspectRatio, "1280 / 800");
const near = (box, want) => Object.keys(want).every((k) => Math.abs(box[k] - want[k]) < 0.001);
assert.ok(near(containedBox(1280, 800, 1280, 800), { left: 0, top: 0, width: 100, height: 100 }));
// What object-fit: contain does with another shape: a 16:9 picture gets bars above and below, 4:3 at the sides.
assert.ok(near(containedBox(1280, 800, 1920, 1080), { left: 0, top: 5, width: 100, height: 90 }));
assert.ok(near(containedBox(1280, 800, 640, 480), { left: 8.333, top: 0, width: 83.333, height: 100 }));
sock.fire("connect");
sock.fire("avatar_update", { animation_state: "pointing", target_x: 640, target_y: 400 });
await settle();  // the empty file list suggests step 1 first
selectStep("board");
await wait(1700);
const svg = $("ink-overlay");
assert.equal(svg.getAttribute("viewBox"), "0 0 1280 800");
const [rect, dot, ring] = svg.children;
assert.deepEqual(["x", "y", "width", "height"].map((k) => rect.getAttribute(k)), ["100", "200", "300", "80"]);
assert.deepEqual([dot.getAttribute("cx"), dot.getAttribute("cy")], ["250", "240"]);
// The target ring is in the same projector space as the ink boxes, sized for this board.
assert.equal(ring.getAttribute("class"), "target");
assert.deepEqual([ring.getAttribute("cx"), ring.getAttribute("cy")], ["640", "400"]);
assert.ok(Math.abs(parseFloat(ring.getAttribute("r")) - 26 * 1280 / 1920) < 1e-9);
// The overlay goes exactly where the picture is drawn.
const img = $("frame");
const placed = () => ["left", "top", "width", "height"].map((k) => svg.style[k]);
Object.assign(img, { naturalWidth: 1280, naturalHeight: 800 });
img.onload();
assert.deepEqual(placed(), ["0%", "0%", "100%", "100%"]);
Object.assign(img, { naturalWidth: 1920, naturalHeight: 1080 });
img.onload();
assert.deepEqual(placed(), ["0%", "5%", "100%", "90%"]);
// An answer with no target clears the ring instead of drawing one at "undefined".
sock.fire("avatar_update", { animation_state: "idle" });
await wait(1600);
assert.equal(svg.children.filter((c) => c.getAttribute("class") === "target").length, 0);
""",
    )


@needs_node
def test_viewport_css_defaults_to_the_kit_projector_shape() -> None:
    css = (STATIC / "css" / "dashboard.css").read_text(encoding="utf-8")
    rule = re.search(r"\.viewport\s*\{([^}]*)\}", css)
    assert rule and re.search(r"aspect-ratio:\s*16\s*/\s*10", rule.group(1))
    assert re.search(r"\.lang\[hidden\]\s*\{\s*display:\s*none", css)


# ---- seed documents ------------------------------------------------------------------

# The worksheet rule as the contract words it ("1. 3x + 5 = 20", "Problem 2: ..."), read literally.
NUMBERED_LINE = re.compile(r"^\s*(?:problem\s*)?\d+\s*[.):]\s+\S", re.I | re.M)


def test_seed_worksheet_levels_match_the_quiz_bank(tmp_path: Path) -> None:
    sheet = parse_quiz_file(MATH_SEEDS / "worksheet_linear_equations.md", "worksheet_linear_equations.md")
    bank = parse_quiz_file(MATH_SEEDS / "quiz_bank.csv", "quiz_bank.csv")
    level = {p.topic: p.level for p in bank}
    one, two, brackets = level["one-step equations"], level["two-step equations"], level["distributive property"]
    assert level["combining like terms"] == brackets
    assert [p.level for p in sheet] == [one] * 4 + [two] * 4 + [brackets] * 4
    quiz = QuizBank(tmp_path / "quiz_bank.json")
    quiz.replace_source("quiz_bank.csv", bank)
    quiz.replace_source("worksheet_linear_equations.md", sheet)
    order = [p.problem for p in quiz.problems()]
    # Both files ticked (the default): the hardest worksheet problems no longer come before easy bank ones.
    assert order.index("x + 9 = 15") < order.index("3x + 5 = 20")
    assert order.index("2(x + 5) = 3x + 4") > order.index("7 - 2x = 1")


def test_guides_have_no_lines_a_worksheet_parser_would_take() -> None:
    guides = [p for p in sorted(MATH_SEEDS.glob("*.md")) if "worksheet" not in p.stem]
    assert guides
    for path in guides:
        text = path.read_text(encoding="utf-8")
        assert NUMBERED_LINE.findall(text) == [], f"{path.name} has numbered lines a worksheet parser would take"
        assert parse_worksheet_text(text, path.name) == []
        # Headings like "(quiz mode)" used to make the guide pass the worksheet test.
        assert not is_worksheet(path.name, text), f"{path.name} reads as a worksheet"


# ---- docs stay true to the code ------------------------------------------------------


@needs_node
def test_docs_call_the_files_tab_by_its_name_in_each_subject() -> None:
    en = strings()["en"]
    math_name, biology_name = en["tab.files.math"], en["tab.files.biology"]
    for name in ("README.md", "docs/BUILD_PLAN.md"):
        text = re.sub(r"[*_]", "", DOCS[name])
        assert math_name in text, f"{name} never names the math tab '{math_name}'"
        for found in re.finditer(re.escape(biology_name), text):
            before = text[max(0, found.start() - 120) : found.start()]
            assert math_name in before, f"{name}: '{biology_name}' without the default subject's '{math_name}'"


def test_documented_benchmark_flags_exist() -> None:
    src = (ROOT / "scripts" / "bench_vlm.py").read_text(encoding="utf-8")
    flags = set(re.findall(r'add_argument\("(-[-a-z]+)"', src))
    profiles = set(re.findall(r'"(\w+)": \w+_SCENARIOS', src))
    assert {"--scenarios", "--profile", "--only", "--raw"} <= flags and profiles == {"math", "biology"}
    for name in ("README.md", "CLAUDE.md", "docs/BUILD_PLAN.md"):
        text = DOCS[name]
        assert "--scenarios --profile math" in text, f"{name} gives no way to benchmark the default math profile"
        used = re.findall(r"`(-[-a-z]+)", text)
        for line in text.splitlines():
            if "bench_vlm" in line:
                used += re.findall(r"(?<![\w-])(-[-a-z]+)", line.split("bench_vlm", 1)[1])
        assert used and set(used) <= flags, f"{name}: {set(used) - flags} not in bench_vlm's arguments"
        assert set(re.findall(r"--profile (\w+)", text)) <= profiles


def test_claude_md_keeps_the_specs_non_negotiables() -> None:
    spec = (ROOT / "docs" / "sar_whiteboard_assistant_prompt.md").read_text(encoding="utf-8")
    assert "Wait for the user to specify which module" in spec and "Windows PowerShell" in spec
    rules = DOCS["CLAUDE.md"].split("Non-negotiables:")[1].split("## Stack")[0]
    assert "wait for the user to name it" in rules
    assert "Windows PowerShell" in rules
    for name in ("README.md", "CLAUDE.md"):
        assert r".\.venv\Scripts\python.exe -m biobench_sar" in DOCS[name]
        assert r"$env:SAR_PROFILE='biology'; .\.venv\Scripts\python.exe -m biobench_sar" in DOCS[name]


def test_agent_docs_point_at_the_cleanup_pass() -> None:
    assert "## Cleanup pass (every branch and PR)" in DOCS["CONTRIBUTING.md"]
    for name in ("CLAUDE.md", "AGENTS.md"):
        assert "CONTRIBUTING.md#cleanup-pass-every-branch-and-pr" in DOCS[name], name


def test_documented_knowledge_paths_exist() -> None:
    for name, text in DOCS.items():
        for folder, file in re.findall(r"knowledge/([a-z_]+)/([\w.]+\.\w+)?", text):
            path = ROOT / "biobench_sar" / "knowledge" / folder / file
            assert path.exists(), f"{name} points at missing {path.relative_to(ROOT)}"


def test_docs_explain_the_shared_knowledge_base_and_how_to_reseed() -> None:
    settings = Settings()
    # "Delete data/chroma_db to load the bundled documents again" only holds while the marker lives inside it.
    assert settings.seed_marker_file.parent == settings.chroma_dir
    readme = DOCS["README.md"]
    assert "data/chroma_db/.seeded" in readme and "delete `data/chroma_db`" in readme
    assert "SAR_PROFILE=biology SAR_DATA_DIR=data/biology" in readme
    assert "Both profiles share one knowledge base" in readme
    assert "`seed_marker_file: Path`" in DOCS["docs/BUILD_PLAN.md"]


def test_readme_worksheet_rules_match_the_parser() -> None:
    section = DOCS["README.md"].split("### Quiz banks")[1].split("## The dashboard")[0]
    listed = re.search(r"says so\*\*: (.+?) \(e\.g\.", section, re.S)
    assert listed, "the README should say what makes a file a worksheet"
    words = [w.strip() for w in re.split(r",|\bor\b", listed.group(1).replace("\n", " ")) if w.strip()]
    assert len(words) >= 8
    for word in words:
        assert is_worksheet("unit3.md", f"# {word.title()}: fractions\n1. 1/2 + 1/4"), word
    assert is_worksheet("unit3_worksheet.md", "1. 3x + 5 = 20")
    example = re.search(r"```md\n(.*?)```", section, re.S)
    assert example, "the README should show a levelled worksheet"
    problems = parse_worksheet_text(example.group(1), "example.md")
    assert [(p.problem, p.answer, p.level, p.topic) for p in problems] == [
        ("4x + 3 = 19", "x = 4", 3, "two-step equations")
    ]
    assert is_worksheet("example.md", example.group(1).strip())


def test_math_spec_says_where_quiz_problems_come_from() -> None:
    spec = DOCS["docs/sar_math_tutor_prompt.md"]
    quiz = spec.split("**Quiz Mode:**")[1].split("### C.")[0]
    assert "rag/quiz_bank.py" in quiz and "SAR_QUIZ_BANK_FILE" in quiz and "SAR_QUIZ_LENGTH" in quiz


def test_help_and_practice_scenarios_use_equations_outside_the_seed_files() -> None:
    plan = DOCS["docs/BUILD_PLAN.md"]
    section = plan.split("## Scenario test cases — Math")[1].split("## Scenario test cases — Environmental")[0]
    seeded = {p.problem.replace(" ", "") for f in sorted(MATH_SEEDS.iterdir()) for p in parse_quiz_file(f, f.name)}
    written = re.findall(r"`([^`]*=[^`]*)`", section)
    assert written, "the math scenarios should name the equations to write"
    # Retrieval brings the seed files' answer keys along with a matching problem.
    assert [w for w in written if w.replace(" ", "") in seeded] == []
    assert "Answers never reach the model except when checking" in plan


def test_contract_describes_the_integrated_quiz() -> None:
    plan = DOCS["docs/BUILD_PLAN.md"]
    assert '"stage": "dictate" \\| "check" \\| "hint"' in plan
    assert "`answer` is set only in stage `check`" in plan
    assert "| quiz `hint` |" in plan
    assert "`quiz_length: int` — env `SAR_QUIZ_LENGTH`, default `10`" in plan
    assert "| `SAR_QUIZ_LENGTH` | `10` |" in DOCS["README.md"]
    from biobench_sar.tutor import DEFAULT_QUIZ_LENGTH

    assert DEFAULT_QUIZ_LENGTH == 10
    if hasattr(Settings, "quiz_length"):  # Settings gains it with the tutor review fixes (contract: Shared)
        assert Settings().quiz_length == 10 or "SAR_QUIZ_LENGTH" in __import__("os").environ


def test_documented_settings_exist_in_config() -> None:
    config = (ROOT / "biobench_sar" / "config.py").read_text(encoding="utf-8")
    known = set(re.findall(r'"(SAR_[A-Z_]+)"', config))
    rows = [line for line in DOCS["README.md"].splitlines() if line.startswith("| `SAR_")]
    documented = {v for line in rows for v in re.findall(r"`(SAR_[A-Z_]+)`", line)}
    assert len(documented) > 20
    # Declared in BUILD_PLAN's Shared contract; config.py gains it with the tutor review fixes.
    pending = {"SAR_QUIZ_LENGTH"} - known
    assert pending <= {"SAR_QUIZ_LENGTH"}
    assert documented - known - pending == set()


# ---- final integration contract: highlight, ui_rects, stop confirmation, phone mode, join page ----------------

# Every timer is recorded, so a test can run the 8 s fade (or any other one) now instead of waiting for it.
TIMERS = """
const timers = [];
const realSetTimeout = globalThis.setTimeout;
globalThis.setTimeout = (fn, ms, ...args) => {
  const id = realSetTimeout(fn, ms, ...args);
  timers.push({ fn, ms, id });
  return id;
};
const runTimer = (ms) => {
  const timer = timers.filter((x) => x.ms === ms).at(-1);
  assert.ok(timer, `no ${ms} ms timer`);
  clearTimeout(timer.id);
  timer.fn();
};
"""
PROJECTOR_TIMED = PROJECTOR.replace('await import("./main.js");', TIMERS + 'await import("./main.js");')
assert PROJECTOR_TIMED != PROJECTOR


@needs_node
def test_projector_outlines_the_ink_the_answer_is_about(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "index.html",
        PROJECTOR_TIMED
        + """
const hl = $("highlight");
const shown = () => !hl.classList.contains("hidden");
const box = () => ["left", "top", "width", "height"].map((k) => hl.style[k]);
sock.fire("connect");
assert.ok(!shown());
// A 1920×1080 projector on a 1920×1080 window: the outline sits 14 px out from the ink, plus its 3 px border,
// so it frames the writing and never lights it up.
sock.fire("avatar_update", { animation_state: "pointing", target_x: 380, target_y: 350,
                             display_text: "Subtract 5 from both sides.", highlight: [400, 300, 200, 100] });
assert.ok(shown());
assert.ok(hl.classList.contains("pulse"));
assert.deepEqual(box(), ["383px", "283px", "234px", "134px"]);
assert.equal(hl.dataset.tone, undefined);
// The speak event of the same answer keeps it; so does one from a server that sends no highlight at all.
sock.fire("speak", { url: null, display_text: "Subtract 5 from both sides.", highlight: [400, 300, 200, 100] });
sock.fire("speak", { url: null, text: "still the same answer" });
assert.ok(shown() && !hl.classList.contains("fading"));
// About 8 s later it fades, then goes.
runTimer(8000);
assert.ok(hl.classList.contains("fading"));
runTimer(800);
assert.ok(!shown());
// A wrong answer glows amber; the next answer without a highlight fades it straight away.
sock.fire("avatar_update", { animation_state: "error", target_x: 380, target_y: 350, highlight: [10, 20, 30, 40] });
assert.ok(shown() && !hl.classList.contains("fading"));
assert.equal(hl.dataset.tone, "error");
sock.fire("avatar_update", { animation_state: "idle", target_x: 1700, target_y: 540, highlight: null });
assert.ok(hl.classList.contains("fading"));
runTimer(800);
assert.ok(!shown());
// Anything but four finite numbers with a size is no highlight.
for (const bad of [[1, 2, 3], [0, 0, -5, 10], [0, 0, 10, "x"], "400,300,200,100", { x: 1 }]) {
  sock.fire("avatar_update", { animation_state: "pointing", highlight: bad });
  assert.ok(!shown(), JSON.stringify(bad));
}
// A new question: gone at once, before the camera reads the board.
sock.fire("avatar_update", { animation_state: "celebrating", highlight: [400, 300, 200, 100] });
assert.equal(hl.dataset.tone, "celebrate");
sock.fire("status", { state: "listening" });
assert.ok(!shown());
// Near a corner it stays on the screen.
sock.fire("avatar_update", { animation_state: "pointing", highlight: [1900, 1070, 100, 50] });
assert.deepEqual(box(), ["1883px", "1053px", "37px", "27px"]);
// A reconnect drops it with the rest of the transient UI.
sock.fire("disconnect");
sock.fire("connect");
assert.ok(!shown());
// The stop-the-quiz question is an answer like any other: it shows in the bubble.
const ask = "Do you want to end the quiz? Say yes to stop, or keep going.";
sock.fire("avatar_update", { animation_state: "idle", spoken_text: ask, display_text: ask, highlight: null });
sock.fire("speak", { url: null, text: ask, display_text: ask, highlight: null });
assert.equal($("bubble").textContent, ask);
assert.ok(!$("bubble").classList.contains("hidden"));
assert.ok(!shown());
""",
    )


@needs_node
def test_projector_bubble_keeps_off_the_highlighted_writing(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "index.html",
        PROJECTOR
        + """
const avatar = $("avatar");
const bubble = $("bubble");
Object.assign(avatar, { offsetWidth: 160, offsetHeight: 160 });
Object.assign(bubble, { offsetWidth: 400, offsetHeight: 100 });
sock.fire("connect");
// Pointing at a tall block of working: centred over the avatar, the bubble would cover its top lines, so it
// moves to the right of the outline (the left side has no room for it).
const answer = { animation_state: "pointing", target_x: 380, target_y: 350, display_text: "What comes next?" };
sock.fire("avatar_update", { ...answer, highlight: [400, 200, 300, 300] });
assert.equal(bubble.style.left, "514px");
assert.ok(!avatar.classList.contains("bubble-below"));
// No room beside it either: the bubble goes under the avatar instead.
sock.fire("avatar_update", { ...answer, highlight: [400, 230, 1500, 100] });
assert.ok(avatar.classList.contains("bubble-below"));
// Without a highlight it is centred over the avatar, as always.
sock.fire("avatar_update", { ...answer, highlight: null });
assert.equal(bubble.style.left, "-120px");
assert.ok(!avatar.classList.contains("bubble-below"));
""",
    )


@needs_node
def test_projector_reports_where_its_ui_is(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "index.html",
        """
// A 1280×800 projector on a 2560×1600 window: every window px is half a projector px.
const doc = installDom(NODES, { bodyData: { projectorWidth: "1280", projectorHeight: "800" } });
window.innerWidth = 2560;
window.innerHeight = 1600;
const $ = (id) => doc.getElementById(id);
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const sock = installIo();
installFetch();
await import("./main.js");
const sent = () => sock.emitted.filter(([name]) => name === "ui_rects").map(([, body]) => body.rects);
const latest = () => sent().at(-1);
$("avatar").rect = { left: 1000, top: 600, width: 160, height: 160 };
$("bubble").rect = { left: 900, top: 480, width: 400, height: 100 };
$("problem-card").rect = { left: 32, top: 28, width: 300, height: 80 };
$("status").rect = { left: 2300, top: 1540, width: 240, height: 40 };
await wait(300);
assert.deepEqual(sent(), [], "nothing goes out before the socket connects");
sock.fire("connect");
await settle();
// The avatar and the status corner; the hidden bubble, problem card, transcript and outline are not on the board.
// Each box gets 10 px of glow around it.
const AVATAR = [495, 295, 90, 90];
const STATUS = [1145, 765, 130, 30];
assert.deepEqual(latest(), [AVATAR, STATUS]);
// An answer with a highlight: the bubble, and the outline as its four edges, so the writing inside it is never
// counted as UI. (No target: the avatar's box is pinned above, and a walk would announce its own end.)
const ink = [500, 300, 200, 100];
sock.fire("avatar_update", { animation_state: "pointing", display_text: "Look here", highlight: ink });
await settle();
const rects = latest();
assert.equal(rects.length, 7);
assert.deepEqual(rects.slice(0, 2), [AVATAR, [445, 235, 210, 60]]);
const edges = rects.slice(2, 6);
const overlap = (a, b) => Math.max(0, Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]))
  * Math.max(0, Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]));
for (const edge of edges) assert.equal(overlap(edge, ink), 0, `edge ${edge} covers the ink`);
const left = Math.min(...edges.map((e) => e[0]));
const top = Math.min(...edges.map((e) => e[1]));
const right = Math.max(...edges.map((e) => e[0] + e[2]));
const bottom = Math.max(...edges.map((e) => e[1] + e[3]));
assert.ok(left < 500 && top < 300 && right > 700 && bottom > 400, "the edges go all the way round the ink");
assert.deepEqual(rects[6], STATUS);
// The problem card is on the board while a quiz runs...
sock.fire("tutor", { mode: "quiz", learn_topic: null, quiz: { active: true, number: 1, total: 5, score: 0, problem: "2x = 6" } });
await settle();
assert.deepEqual(latest()[2], [11, 9, 160, 50]);
// ...but not while the board is read, and the outline is gone by then too.
sock.fire("status", { state: "thinking" });
await settle();
assert.deepEqual(latest(), [AVATAR, [445, 235, 210, 60], STATUS]);
// Unchanged boxes go again once a second, so the server's copy (kept 3 s) never runs out while the page is up.
const before = sent().length;
await wait(600);
assert.equal(sent().length, before, "nothing changed, and a second hasn't passed");
await wait(700);
assert.equal(sent().length, before + 1);
assert.deepEqual(latest(), [AVATAR, [445, 235, 210, 60], STATUS]);
// A move no handler announces (the avatar mid-walk) goes out within a tick; the idle bob doesn't.
$("avatar").rect = { left: 1004, top: 602, width: 160, height: 160 };
await wait(400);
assert.equal(sent().length, before + 1, "a few px of bobbing is not news");
$("avatar").rect = { left: 1200, top: 600, width: 160, height: 160 };
await wait(400);
assert.equal(sent().length, before + 2);
assert.deepEqual(latest()[0], [595, 295, 90, 90]);
// The checkerboard covers everything while it is up.
sock.fire("show_calibration", {});
await settle();
assert.deepEqual(latest(), []);
sock.fire("hide_calibration", {});
await settle();
assert.ok(latest().length > 0);
// A hidden tab projects nothing new and sends nothing; it reports again the moment it is shown.
doc.hidden = true;
const hiddenAt = sent().length;
await wait(1300);
assert.equal(sent().length, hiddenAt);
doc.hidden = false;
for (const fn of doc.listeners.visibilitychange) fn();
await settle();
assert.equal(sent().length, hiddenAt + 1);
// Never while disconnected: socket.io would replay stale boxes after the reconnect.
sock.fire("disconnect");
const offline = sock.emitted.length;
await wait(1300);
assert.equal(sock.emitted.length, offline);
""",
    )


@needs_node
def test_quiz_bar_waits_for_yes_or_no_before_ending_the_quiz(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + f"const QUIZ = {QUIZ};\n"
        + """
const sock = installIo();
const http = installFetch({
  "GET /api/tutor": { status: 200, body: { mode: "quiz", learn_topic: null, quiz: QUIZ, confirm_stop: true } },
  "POST /api/tutor/quiz/stop": { status: 200, body: { mode: "help", learn_topic: null, quiz: null, confirm_stop: false } },
});
await import("./dashboard/main.js");
sock.fire("connect");
await settle();
const line = $("quiz-confirm");
// A page opened while the assistant waits for the answer shows it too (GET /api/tutor).
assert.equal(line.hidden, false);
assert.equal(line.textContent, "Waiting for yes/no to end the quiz");
sock.fire("tutor", { mode: "quiz", learn_topic: null, quiz: QUIZ, confirm_stop: false });
assert.equal(line.hidden, true);
sock.fire("tutor", { mode: "quiz", learn_topic: null, quiz: QUIZ, confirm_stop: true });
assert.equal(line.hidden, false);
assert.equal($("quiz-stop-btn").hidden, false);
assert.equal($("quiz-progress").textContent, "Problem 2 of 8 · 1 right on the first try");
sock.fire("language", { setting: "es", current: "es" });
assert.equal(line.textContent, "Esperando un «sí» o un «no» para terminar el quiz");
sock.fire("language", { setting: "en", current: "en" });
// The dashboard's Stop quiz button never asks: it stops at once.
let asked = false;
globalThis.confirm = () => { asked = true; return false; };
await $("quiz-stop-btn").fire("click");
assert.ok(http.calls.some((c) => c.method === "POST" && c.path === "/api/tutor/quiz/stop"));
assert.equal(asked, false);
assert.equal(line.hidden, true);
assert.equal($("ask-msg").textContent, "Quiz stopped.");
// Only a running quiz waits for an answer; servers without the flag never show the line.
sock.fire("tutor", { mode: "quiz", learn_topic: null, quiz: { ...QUIZ, active: false }, confirm_stop: true });
assert.equal(line.hidden, true);
sock.fire("tutor", { mode: "quiz", learn_topic: null, quiz: QUIZ });
assert.equal(line.hidden, true);
""",
    )


@needs_node
def test_board_overlay_outlines_the_highlighted_ink(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + """
const sock = installIo();
installFetch({ "GET /api/ink": { status: 200, body: { regions: [], size: [1920, 1080] } } });
const { selectStep } = await import("./dashboard/tabs.js");
$("ink-toggle").checked = true;
await import("./dashboard/main.js");
sock.fire("connect");
await settle();
selectStep("board");
const outlines = () => $("ink-overlay").children.filter((c) => c.getAttribute("class") === "highlight");
sock.fire("avatar_update", { animation_state: "pointing", target_x: 380, target_y: 350, highlight: [400, 300, 200, 100] });
await wait(1600);
const [rect] = outlines();
assert.deepEqual(["x", "y", "width", "height"].map((k) => rect.getAttribute(k)), ["400", "300", "200", "100"]);
// A server without highlights leaves it; the next answer's null clears it.
sock.fire("avatar_update", { animation_state: "idle", target_x: 1700, target_y: 540 });
await wait(1600);
assert.equal(outlines().length, 1);
sock.fire("avatar_update", { animation_state: "idle", target_x: 1700, target_y: 540, highlight: null });
await wait(1600);
assert.equal(outlines().length, 0);
""",
    )


@needs_node
def test_file_list_warns_about_files_that_still_have_answers(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + """
const sock = installIo();
installFetch({ "GET /api/documents": { status: 200, body: { documents: [
  { source: "my_worksheet.md", type: "md", chunks: 3, active: true, has_answers: true },
  { source: "clean.md", type: "md", chunks: 2, active: true },
] } } });
await import("./dashboard/main.js");
sock.fire("connect");
await settle();
const [flagged, clean] = $("files").children;
const warning = flagged.children[1].children[1];
assert.equal(warning.className, "file-warn");
assert.equal(warning.textContent, "This file still has its answers. Upload it again so the assistant can't see them.");
assert.equal(clean.children[1].children.length, 1, "a clean file has its meta line only");
assert.deepEqual([$("badge-files").textContent, $("tab-files").dataset.tone], ["1 file has answers", "warn"]);
sock.fire("language", { setting: "es", current: "es" });
assert.equal($("files").children[0].children[1].children[1].textContent,
  "Este archivo todavía tiene sus respuestas. Vuelve a subirlo para que el asistente no las pueda ver.");
assert.equal($("badge-files").textContent, "1 archivo con respuestas");
// Uploaded again without its answers: the warning and the badge go.
sock.fire("documents_changed", { documents: [{ source: "my_worksheet.md", type: "md", chunks: 3, active: true }] });
assert.equal($("files").children[0].children[1].children.length, 1);
assert.deepEqual([$("badge-files").textContent, $("tab-files").dataset.tone], ["1 en uso", "ok"]);
""",
    )


LAN = {
    "enabled": True,
    "urls": ["http://192.168.1.20:5050/join?pin=482913", "http://10.0.0.7:5050/join?pin=482913"],
    "pin": "482913",
    "phones": 2,
    "locked_for": 0,
}
PHONE_DASHBOARD = DASHBOARD.replace("installDom(NODES)", 'installDom(NODES, { bodyData: { lan: "1" } })')
assert PHONE_DASHBOARD != DASHBOARD


@needs_node
def test_phone_card_shows_the_qr_code_pin_and_phones(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        PHONE_DASHBOARD
        + f"const LAN = {json.dumps(LAN)};\n"
        + """
const added = [];
globalThis.qrcode = (type, level) => {
  assert.deepEqual([type, level], [0, "M"]);
  return { addData: (url) => added.push(url), make() {}, getModuleCount: () => 2, isDark: (row, col) => row === col };
};
let lan = structuredClone(LAN);
let disconnectReply = { status: 200, body: { ok: true } };
const sock = installIo();
const http = installFetch({
  "GET /api/lan": () => ({ status: 200, body: structuredClone(lan) }),
  "POST /api/lan/rotate": () => {
    lan = { ...lan, pin: "105377", phones: 0, urls: lan.urls.map((u) => u.replace("482913", "105377")) };
    return { status: 200, body: structuredClone(lan) };
  },
  "POST /api/lan/disconnect": () => {
    if (disconnectReply.status === 200) lan = { ...lan, phones: 0 };
    return disconnectReply;
  },
});
const { refreshPhone } = await import("./dashboard/phone.js");
await import("./dashboard/main.js");
await settle();
assert.equal($("phone-card").hidden, false);
assert.equal($("phone-pin").textContent, "482 913");
// The address to type has no ?pin=, so typing it opens the PIN form; the QR code carries the PIN.
assert.equal($("phone-url").textContent, "http://192.168.1.20:5050/join");
assert.equal($("phone-other").textContent, "Other addresses: http://10.0.0.7:5050/join");
assert.equal($("phone-other").hidden, false);
assert.equal($("phone-count").textContent, "2 phones or tablets connected");
assert.deepEqual(added, [LAN.urls[0]]);
// One SVG path for the dark modules, black on white, with a 4-module quiet zone.
const svg = $("phone-qr").children[0];
assert.equal(svg.getAttribute("viewBox"), "0 0 10 10");
const [paper, dark] = svg.children;
assert.equal(paper.getAttribute("fill"), "#ffffff");
assert.equal(dark.getAttribute("fill"), "#000000");
assert.equal(dark.getAttribute("d"), "M4 4h1v1h-1zM5 5h1v1h-1z");
// New PIN asks first; Cancel changes nothing.
globalThis.confirm = () => false;
await $("phone-rotate-btn").fire("click");
assert.ok(!http.calls.some((c) => c.path === "/api/lan/rotate"));
globalThis.confirm = () => true;
await $("phone-rotate-btn").fire("click");
assert.ok(http.calls.some((c) => c.method === "POST" && c.path === "/api/lan/rotate"));
assert.equal($("phone-pin").textContent, "105 377");
assert.equal($("phone-count").textContent, "No phones connected");
assert.equal($("phone-msg").textContent, "New PIN made. Every phone was signed out.");
assert.equal(added.at(-1), "http://192.168.1.20:5050/join?pin=105377");
// Disconnect all phones: its reply is not the card's body, so the card asks for it again.
lan = { ...lan, phones: 1 };
await refreshPhone();
assert.equal($("phone-count").textContent, "1 phone or tablet connected");
await $("phone-disconnect-btn").fire("click");
assert.ok(http.calls.some((c) => c.method === "POST" && c.path === "/api/lan/disconnect"));
assert.equal($("phone-count").textContent, "No phones connected");
assert.equal($("phone-msg").textContent, "Every phone was disconnected.");
disconnectReply = { status: 500, body: { error: "RuntimeError: socket gone" } };
await $("phone-disconnect-btn").fire("click");
assert.equal($("phone-msg").textContent,
  "Couldn't disconnect the phones. Check the assistant is still running (restart the assistant if it closed), then try again.");
assert.equal($("phone-msg").dataset.tone, "bad");
assert.match(logText(), /RuntimeError: socket gone/);
// A guesser locked everyone out: say so, and for how long.
lan = { ...lan, locked_for: 90 };
await refreshPhone();
assert.equal($("phone-msg").textContent, "Too many wrong PINs: nobody can join for 2 minutes.");
assert.equal($("phone-msg").dataset.tone, "warn");
lan = { ...lan, locked_for: 0 };
await refreshPhone();
assert.equal($("phone-msg").textContent, "");
// Off the network: no code to scan, and what to do about it.
lan = { ...lan, urls: [] };
await refreshPhone();
assert.equal($("phone-qr").hidden, true);
assert.equal($("phone-url").textContent, "This laptop isn't on a network. Join the classroom Wi-Fi.");
assert.equal($("phone-other").hidden, true);
// A reconnect asks again: a restarted server has a new PIN.
lan = structuredClone(LAN);
const asked = http.calls.filter((c) => c.path === "/api/lan").length;
sock.fire("connect");
await settle();
assert.equal(http.calls.filter((c) => c.path === "/api/lan").length, asked + 1);
assert.equal($("phone-pin").textContent, "482 913");
assert.equal($("phone-qr").hidden, false);
sock.fire("language", { setting: "es", current: "es" });
assert.equal($("phone-count").textContent, "2 dispositivos conectados");
assert.equal($("phone-other").textContent, "Otras direcciones: http://10.0.0.7:5050/join");
""",
    )


@needs_node
@pytest.mark.parametrize(
    ("lan", "reply"), [("1", "403"), ("1", "404"), ("1", "disabled"), ("0", "ok")], ids=["phone", "old", "off", "local"]
)
def test_phone_card_stays_hidden_off_the_laptop_or_outside_lan_mode(tmp_path: Path, lan: str, reply: str) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD.replace("installDom(NODES)", f'installDom(NODES, {{ bodyData: {{ lan: "{lan}" }} }})')
        + f"const LAN = {json.dumps(LAN)};\nconst REPLY = {json.dumps(reply)};\n"
        + """
const replies = {
  "403": { status: 403, body: { error: "local only" } },
  "404": { status: 404, body: { error: "not found" } },
  disabled: { status: 200, body: { enabled: false, urls: [], pin: null, phones: 0 } },
  ok: { status: 200, body: LAN },
};
const sock = installIo();
const http = installFetch({ "GET /api/lan": replies[REPLY] });
const { refreshPhone } = await import("./dashboard/phone.js");
await import("./dashboard/main.js");
sock.fire("connect");
await settle();
assert.equal($("phone-card").hidden, true);
assert.doesNotMatch($("phone-pin").textContent, /\\d/, "no PIN written anywhere");
const calls = () => http.calls.filter((c) => c.path === "/api/lan").length;
// Outside LAN mode the page never asks; a phone (403) or an older server (404) is asked once, then left alone.
if (REPLY === "ok") assert.equal(calls(), 0);
else if (REPLY !== "disabled") {
  const seen = calls();
  await refreshPhone().catch(() => {});
  assert.equal(calls(), seen);
}
""",
    )


@needs_node
def test_phone_card_off_the_network_from_the_start(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        PHONE_DASHBOARD
        + f"const LAN = {json.dumps({**LAN, 'urls': []})};\n"
        + """
globalThis.qrcode = () => assert.fail("no address, so no code to draw");
installIo();
installFetch({ "GET /api/lan": { status: 200, body: LAN } });
await import("./dashboard/main.js");
await settle();
assert.equal($("phone-card").hidden, false);
assert.equal($("phone-qr").hidden, true, "no empty white square where the code would be");
assert.equal($("phone-url").textContent, "This laptop isn't on a network. Join the classroom Wi-Fi.");
assert.equal($("phone-pin").textContent, "482 913");
""",
    )


@needs_node
def test_phone_qr_code_from_the_vendored_library(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + f"const VENDOR = {json.dumps(str(STATIC / 'vendor' / 'qrcode.min.js'))};\n"
        + """
const { readFileSync } = await import("node:fs");
// A classic script that defines a global `qrcode`, as the template loads it.
new Function(readFileSync(VENDOR, "utf8") + "\\nglobalThis.qrcode = qrcode;")();
const { groupPin, qrSvg, typedUrl } = await import("./dashboard/phone.js");
const svg = qrSvg("http://192.168.1.20:5050/join?pin=482913");
const size = Number(svg.getAttribute("viewBox").split(" ")[2]);
// Version v has 4v + 17 modules a side; this address needs version 3 or 4 at level M. Plus 4 + 4 quiet zone.
assert.ok([29, 33].includes(size - 8), `size ${size}`);
const d = svg.children[1].getAttribute("d");
// The top-left finder pattern: its first row is 7 dark modules from (4, 4).
for (let col = 0; col < 7; col++) assert.ok(d.includes(`M${col + 4} 4h1v1h-1z`), `finder module ${col}`);
assert.ok(!d.includes("M11 4h1v1h-1z"), "the separator after the finder is light");
assert.equal(groupPin("482913"), "482 913");
assert.equal(groupPin("12345"), "12345");
assert.equal(typedUrl("http://[fe80::1]:5050/join?pin=123456"), "http://[fe80::1]:5050/join");
delete globalThis.qrcode;
assert.equal(qrSvg("http://x/join"), null, "no library: no code, and no crash");
""",
    )


@needs_node
def test_service_worker_only_in_lan_mode(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        DASHBOARD
        + """
installFetch();
const registered = [];
let unregistered = 0;
const deleted = [];
const workers = {
  register: (url, options) => { registered.push([url, options]); return Promise.resolve(); },
  getRegistrations: () => Promise.resolve([{ unregister: () => { unregistered += 1; } }]),
};
Object.defineProperty(globalThis, "navigator", { value: { serviceWorker: workers }, configurable: true });
globalThis.caches = { keys: async () => ["sar-shell-v3", "someone-else"], delete: async (name) => deleted.push(name) };
const { initServiceWorker } = await import("./dashboard/phone.js");
doc.body.dataset.lan = "1";
initServiceWorker();
await settle();
assert.deepEqual(registered, [["/sw.js", { scope: "/" }]]);
assert.equal(unregistered, 0);
assert.deepEqual(deleted, []);
// Outside LAN mode a worker left from an earlier phone-mode run goes, with its caches (and only its own).
doc.body.dataset.lan = "0";
initServiceWorker();
await settle();
assert.equal(registered.length, 1);
assert.equal(unregistered, 1);
assert.deepEqual(deleted, ["sar-shell-v3"]);
// Plain-HTTP phones have no serviceWorker at all.
Object.defineProperty(globalThis, "navigator", { value: {}, configurable: true });
initServiceWorker();
""",
    )


def render_page(name: str, **variables: object) -> str:
    """A template as Flask renders it, with only the variables a test passes (a server may not send them all)."""
    from flask import Flask, render_template

    app = Flask("pages", template_folder=str(TEMPLATES), static_folder=str(STATIC))
    with app.test_request_context():
        return render_template(name, **variables)


def test_phone_mode_parts_render_only_in_lan_mode() -> None:
    size = {"projector_width": 1280, "projector_height": 800}
    on = render_page("precharge.html", lan_enabled=True, **size)
    for page in (render_page("precharge.html", lan_enabled=False, **size), render_page("precharge.html", **size)):
        assert 'data-lan="0"' in page
        for part in ('rel="manifest"', "qrcode.min.js", 'id="phone-card"', "apple-touch-icon"):
            assert part not in page, part
    assert 'data-lan="1"' in on
    assert '<link rel="manifest" href="/static/manifest.webmanifest">' in on
    assert 'id="phone-card"' in on and "apple-touch-icon.png" in on
    # The QR library is a classic script, loaded before the module that draws with it.
    assert on.index("/static/vendor/qrcode.min.js") < on.index("/static/js/dashboard/main.js")
    projector = render_page("index.html", lan_enabled=True, **size)
    assert 'rel="manifest"' not in projector and "qrcode" not in projector


def test_join_page_form_error_and_lockout() -> None:
    base = {"language": "en", "lan_enabled": True}
    plain = render_page("join.html", error=False, locked_for=0, **base)
    assert 'id="join-error"' not in plain and 'id="join-locked"' not in plain
    form = re.search(r"<form\b[^>]*>", plain).group(0)
    assert 'method="post"' in form and 'action="/join"' in form
    pin = re.search(r"<input\b[^>]*>", plain).group(0)
    for attr in ('name="pin"', 'inputmode="numeric"', 'autocomplete="one-time-code"', "required"):
        assert attr in pin, attr
    # no-referrer would make the PIN form post Origin: null, which the cross-origin guard refuses.
    assert '<meta name="referrer" content="same-origin">' in plain and "no-referrer" not in plain
    assert 'rel="manifest"' in plain
    wrong = render_page("join.html", error=True, locked_for=0, **base)
    assert 'id="join-error"' in wrong and 'id="join-locked"' not in wrong
    locked = render_page("join.html", error=True, locked_for=90, lockout="Too many", message="Too many", **base)
    assert 'data-locked-for="90"' in locked and 'id="join-error"' not in locked
    spanish = render_page("join.html", error=False, language="es")
    assert '<html lang="es">' in spanish and 'data-language="es"' in spanish
    # Only the contract's two variables: still a working form, and no phone-mode extras.
    bare = render_page("join.html", error=False, language="en")
    assert 'id="join-form"' in bare and 'rel="manifest"' not in bare and 'id="join-locked"' not in bare


JOIN = """
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const ticks = [];
let stopped = false;
globalThis.setInterval = (fn) => { ticks.push(fn); return ticks.length; };
globalThis.clearInterval = () => { stopped = true; };
const join = (nodes, { language = "", browser = ["en-US"] } = {}) => {
  const doc = installDom(nodes, { bodyData: { language } });
  Object.defineProperty(globalThis, "navigator", { value: { languages: browser }, configurable: true });
  return doc;
};
"""


@needs_node
def test_join_page_speaks_the_browser_language_and_counts_down_a_lockout(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "join.html",
        JOIN
        + """
// The server renders one alert at most: here the lockout, 61 s long.
const doc = join(NODES.filter((n) => n.id !== "join-error"), { language: "en", browser: ["es-MX", "en-US"] });
const $ = (id) => doc.getElementById(id);
$("join-locked").dataset.lockedFor = "61";
const { browserLanguage, lockoutText } = await import("./join.js");
assert.equal(doc.documentElement.lang, "es");
const title = [...doc.body.descendants()].find((n) => n.tagName === "TITLE");
assert.equal(title.textContent, "Entrar al Asistente del pizarrón");
assert.equal($("join-btn").textContent, "Entrar");
assert.equal($("join-locked").textContent, "Demasiados PIN incorrectos. Vuelve a intentarlo en 2 minutos.");
// Guesses during a lockout are refused, so the button waits for the countdown.
assert.equal($("join-btn").disabled, true);
ticks[0]();
assert.equal($("join-locked").textContent, "Demasiados PIN incorrectos. Vuelve a intentarlo en 1 minuto.");
ticks[0]();
assert.equal($("join-locked").textContent, "Demasiados PIN incorrectos. Vuelve a intentarlo en 59 segundos.");
for (let i = 0; i < 58; i++) ticks[0]();
assert.equal($("join-locked").textContent, "Demasiados PIN incorrectos. Vuelve a intentarlo en 1 segundo.");
assert.equal($("join-btn").disabled, true);
ticks[0]();
assert.equal($("join-locked").hidden, true);
assert.equal($("join-btn").disabled, false);
assert.equal($("join-msg").textContent, "Ya puedes volver a intentarlo.");
assert.ok(stopped);
// The browser's language first, then the one the server rendered, then English.
assert.equal(browserLanguage(["EN-gb", "es"], "es"), "en");
assert.equal(browserLanguage(["fr-FR", "de"], "es"), "es");
assert.equal(browserLanguage(["fr-FR"], "xx"), "en");
assert.equal(browserLanguage([undefined], undefined), "en");
assert.equal(browserLanguage(undefined, "es"), "es");
assert.equal(lockoutText(60), "Demasiados PIN incorrectos. Vuelve a intentarlo en 1 minuto.");
assert.equal(lockoutText(125), "Demasiados PIN incorrectos. Vuelve a intentarlo en 3 minutos.");
""",
    )


@needs_node
def test_join_page_after_a_wrong_pin(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "join.html",
        JOIN
        + """
const doc = join(NODES.filter((n) => n.id !== "join-locked"), { language: "es", browser: ["en-US"] });
const $ = (id) => doc.getElementById(id);
$("pin").value = "48291";
await import("./join.js");
// An English phone gets English, whatever the class language.
assert.equal(doc.documentElement.lang, "en");
assert.equal($("join-error").textContent, "That PIN didn't work. Check it on the laptop and try again.");
assert.equal($("join-btn").disabled, false);
assert.equal($("pin").value, "", "the next try starts from an empty box");
assert.equal(ticks.length, 0, "no countdown without a lockout");
""",
    )


@needs_node
def test_help_texts_never_assume_a_terminal_window() -> None:
    for table in strings().values():
        for key, text in table.items():
            for phrase in ("Terminal window", "ventana de Terminal", "Ctrl+C"):
                assert phrase not in text, f"{key} assumes a Terminal window: {text}"
    en, es = strings()["en"], strings()["es"]
    for key in ("hint.offline", "conn.unreachable", "lang.failed"):
        assert "restart the assistant" in en[key], key
        assert "vuelve a iniciar el asistente" in es[key], key
    # The desktop app on either system, not only a Terminal on a Mac.
    assert "On a Mac" in en["check.camera.fix"] and "Let desktop apps access your camera" in en["check.camera.fix"]
    assert "En una Mac" in es["check.camera.fix"] and "En Windows" in es["check.camera.fix"]
    for key in ("check.camera.fixMac", "check.camera.fixWindows", "check.camera.fixOther"):
        assert "restart the assistant" in en[key], key
        assert "vuelve a iniciar el asistente" in es[key], key


def css_block(css: str, opener: str) -> str:
    """The body of the `opener { ... }` block, nested braces included."""
    start = css.index(opener)
    depth, i = 0, css.index("{", start)
    for j in range(i, len(css)):
        depth += {"{": 1, "}": -1}.get(css[j], 0)
        if depth == 0:
            return css[i + 1 : j]
    raise AssertionError(f"{opener} is not closed")


def test_dashboard_steps_fit_one_laptop_screen() -> None:
    css = (STATIC / "css" / "dashboard.css").read_text(encoding="utf-8")
    # A fixed panel height forced page scroll on short laptop screens.
    assert not re.search(r"\.panel\s*\{[^}]*min-height:\s*\d{3}px", css)
    shell = re.sub(r"/\*.*?\*/", "", css_block(css, "@media (min-width: 761px) and (min-height: 480px)"), flags=re.S)
    rules = {sel.strip(): body for sel, body in re.findall(r"([^{}]+)\{([^{}]*)\}", shell)}
    assert "height: 100vh" in rules[".page"] and "flex-direction: column" in rules[".page"]
    assert "min-height: 0" in rules["main"] and "flex: 1 1 auto" in rules["main"]
    assert "overflow-y: auto" in rules[".panel"] and "min-height: 0" in rules[".panel"]
    # The long parts fill what is left and scroll inside: no "100vh - N px" guesses at the header's height.
    entries = next(body for sel, body in rules.items() if sel.startswith(".panel > .entries"))
    assert "flex: 1 1 0" in entries and "max-height: none" in entries
    assert "overflow-y: auto" in rules[".panel > .files"]
    assert "overflow-y: auto" in rules[".board-side .checks"] and "flex: none" in rules[".board-side .calibrate"]
    assert "overflow-y: auto" in rules[".more-col"] and "overflow-y: auto" in rules[".more-col > .config"]
    assert "100vh -" not in shell
