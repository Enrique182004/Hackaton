import { $, el, formatDuration } from "./api.js";
import { setBadge } from "./tabs.js";

const SENTENCES = {
  idle: "Ready. Ask a question below, or on the board.",
  listening: "Listening… speak now.",
  transcribing: "Writing down what you said…",
  thinking: "Thinking about your question…",
  speaking: "Answering on the board…",
};

const ASK_BADGES = {
  idle: "ready",
  listening: "listening…",
  transcribing: "writing it down…",
  thinking: "thinking…",
  speaking: "answering…",
};

let connected = false;
let lastState = "idle";

function check(tone, what, how = "") {
  const mark = { ok: "✓", warn: "!", bad: "✗" }[tone];
  return el(
    "li",
    { dataset: { tone } },
    el("span", { className: "mark", textContent: mark, "aria-label": tone === "ok" ? "OK" : "Needs attention" }),
    el("span", {}, el("span", { className: "what", textContent: what }), how ? el("span", { className: "how", textContent: how }) : null),
  );
}

function filesCheck(s) {
  const total = s.total_documents;
  const active = s.active_documents ?? total;
  if (total === undefined) {
    return s.doc_chunks > 0 ? check("ok", "Study files are loaded") : check("warn", "No study files yet", "Add some in step 1.");
  }
  if (total === 0) return check("warn", "No study files yet", "Add some in step 1.");
  if (active === 0) return check("warn", "No study files ticked", "Tick at least one file in step 1, or answers won't use your notes.");
  return check("ok", `Using ${active} of ${total} study file${total === 1 ? "" : "s"}`);
}

export function renderStatus(s) {
  $("checks").replaceChildren(
    s.camera_running
      ? check("ok", "Camera is working")
      : check("bad", "Camera not found", "Allow camera access for the app that runs the assistant (on a Mac: System Settings → Privacy & Security → Camera), then restart it."),
    s.calibrated
      ? check("ok", "Camera and projector are lined up")
      : check("warn", "Not lined up yet", "Press “Line up camera and projector” below."),
    filesCheck(s),
    s.vlm_backend === "mock"
      ? check("warn", "Practice mode: no AI model", "Answers are placeholders. Restart without SAR_VLM_BACKEND=mock to use the real model.")
      : check("ok", "AI model ready", s.vlm_model),
    s.audio_available
      ? check("ok", "Microphone ready")
      : check("bad", "No microphone found", "You can still type your questions."),
    s.tts_backend === "none" ? check("warn", "Answers are shown but not read aloud") : check("ok", "Answers are read aloud"),
  );
  $("fps").textContent = s.camera_running ? `${(s.fps ?? 0).toFixed(0)} frames/s` : "–";
  $("latency").textContent = s.last_latency_s == null ? "–" : `${s.last_latency_s.toFixed(1)} s`;
  $("uptime").textContent = formatDuration(s.uptime_s);
  $("viewport").dataset.live = s.camera_running ? "1" : "0";
  if (!s.camera_running) setBadge("board", "camera not found", "bad");
  else if (!s.calibrated) setBadge("board", "not lined up yet", "warn");
  else setBadge("board", "ready", "ok");
}

export function renderState(state, detail) {
  lastState = state;
  if (!connected) return;
  const line = $("status-line");
  line.dataset.state = state;
  let text = SENTENCES[state] || state;
  if (detail === "busy") text += " Still working on the last question, so please wait a moment.";
  $("status-text").textContent = text;
  $("listen-btn").dataset.live = state === "listening" ? "1" : "0";
  setBadge("ask", ASK_BADGES[state] || state, state === "idle" ? "ok" : "warn");
}

// `hint` marks a degraded mode where polling still works, so it reads as a warning rather than a failure.
export function renderConnection(isConnected, hint = "") {
  connected = isConnected || Boolean(hint);
  $("conn-text").textContent = isConnected ? "Connected" : hint ? "Connected (no live updates)" : "Not connected";
  if (!connected) {
    $("status-line").dataset.state = "offline";
    $("status-text").textContent = "Can't reach the assistant. Is the app still running?";
    setBadge("ask", "not connected", "bad");
    return;
  }
  renderState(lastState);
}
