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

GENERIC_FAMILIES = {"serif", "sans-serif", "monospace", "cursive", "system-ui", "ui-monospace", "ui-sans-serif"}
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
    assert {"font-head", "font-body", "font-hand"} <= set(stacks)
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
const LISTENING = "Listening… ask your question now. It stops by itself after about 5 seconds.";
const text = (node) => (typeof node === "string" ? node : [node.textContent, ...node.children.map(text)].join(" ").trim());
const logText = () => $("log").children.map(text).join("\\n");
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
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
const doc = installDom(IDS, { bodyData: { projectorWidth: "1920", projectorHeight: "1080" } });
const $ = (id) => doc.getElementById(id);
const sock = installIo();
installFetch();
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
sock.fire("disconnect");
const space = { code: "Space", key: " ", repeat: false, preventDefault() {} };
for (const fn of doc.listeners.keydown) fn(space);
assert.equal(label(), "Not connected – question not sent");
assert.deepEqual(sock.emitted, []);
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
const card = $("quiz-card");
installFetch({ "GET /api/tutor": { status: 200, body: { mode: "quiz", learn_topic: null, quiz: QUIZ } } });
sock.fire("connect");
await settle();
// A reconnect asks the server, so a quiz that started while the page was away still shows.
assert.ok(!card.classList.contains("hidden"));
assert.equal($("quiz-card-count").textContent, "Quiz · problem 2 of 8");
assert.equal($("quiz-card-problem").textContent, "3x + 5 = 20");
// Made-up problems have no text: the card keeps the count only.
sock.fire("tutor", { mode: "quiz", learn_topic: null, quiz: { ...QUIZ, number: 3, problem: null } });
assert.equal($("quiz-card-count").textContent, "Quiz · problem 3 of 8");
assert.equal($("quiz-card-problem").textContent, "");
sock.fire("tutor", { mode: "help", learn_topic: null, quiz: null });
assert.ok(card.classList.contains("hidden"));
sock.fire("tutor", { mode: "quiz", learn_topic: null, quiz: { ...QUIZ, active: false } });
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
assert.ok($("quiz-card").classList.contains("hidden"));
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
assert.equal($("status-text").textContent, "");
assert.equal($("conn-text").textContent, "");
assert.notEqual($("badge-ask").textContent, "not connected");
sock.fire("connect_error", { message: "xhr poll error" });
sock.fire("connect_error", { message: "xhr poll error" });
assert.match($("status-text").textContent, /^Can't reach the assistant\\. Check its Terminal window/);
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
  "Couldn't change how the assistant helps. Check the assistant's Terminal window is still open, then try again.");
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
assert.match(checks(), /Camera not found .*not open in another app \\(Zoom, FaceTime\\).*\\(Ctrl\\+C\\) and start it again\\./);
assert.match(checks(), /Demo mode: answers are pretend Ask whoever set up the assistant/);
assert.match(checks(), /Answers are read aloud on the projector page \\(click it once to allow sound\\)/);
renderStatus({ ...base, vlm_backend: "ollama", vlm_model: "qwen2.5vl:3b", tts_backend: "none" });
assert.match(checks(), /AI model chosen: qwen2\\.5vl:3b If an answer says it couldn't reach the AI model, start Ollama and ask again\\./);
assert.doesNotMatch(checks(), /AI model ready/);
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
const hint = "Check the assistant's Terminal window is still open, then try again.";
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


# ---- projector: quiz card and bubble -------------------------------------------------


@needs_node
def test_projector_quiz_card_steps_aside_while_the_board_is_read(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "index.html",
        PROJECTOR
        + f"const QUIZ = {QUIZ};\n"
        + """
const card = $("quiz-card");
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
    rule = re.search(r"#quiz-card\s*\{([^}]*)\}", css)
    assert rule, "no #quiz-card rule"
    body = rule.group(1)
    # A glow or a thick bright outline is picked up as ink by find_ink_regions and read by the VLM.
    assert "box-shadow" not in body and "text-shadow" not in body
    border = re.search(r"border:\s*([\d.]+)px", body)
    assert border and float(border.group(1)) <= 2
    width = re.search(r"max-width:\s*([\d.]+)vw", body)
    assert width and float(width.group(1)) <= 30
    assert re.search(r"#quiz-card\.capturing\s*\{[^}]*visibility:\s*hidden", css)


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
""",
    )
