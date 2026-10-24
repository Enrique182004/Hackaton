import { $, el, formatDuration } from "./api.js";
import { words } from "./profile.js";
import { setBadge } from "./tabs.js";

export const LISTENING_NOTE = "Listening… ask your question now. It stops by itself after about 5 seconds.";

const SENTENCES = {
  idle: "Ready. Ask a question below, or on the board.",
  listening: LISTENING_NOTE,
  transcribing: "Writing down what you said…",
  thinking: "Thinking about your question…",
  speaking: "Answering on the board…",
};

const ASK_BADGES = {
  listening: "listening…",
  transcribing: "writing it down…",
  thinking: "thinking…",
  speaking: "answering…",
};

const CAMERA_FIX =
  "Check the camera is plugged in and not open in another app (Zoom, FaceTime). On a Mac, allow Terminal in System Settings → Privacy & Security → Camera. Then stop the assistant in its Terminal window (Ctrl+C) and start it again.";

let connected = false;
let lastState = "idle";
// What the Ask badge says while nothing is happening: the tutor mode once it is known.
let idleBadge = "ready";

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
  const w = words();
  const total = s.total_documents;
  const active = s.active_documents ?? total;
  if (total === undefined) {
    return s.doc_chunks > 0 ? check("ok", "Your files are loaded") : check("warn", w.filesMissing, "Add some in step 1.");
  }
  if (total === 0) return check("warn", w.filesMissing, "Add some in step 1.");
  if (active === 0) return check("warn", w.filesNoneTicked, "Tick at least one file in step 1, or answers won't use them.");
  return check("ok", `Using ${active} of ${total} file${total === 1 ? "" : "s"}`);
}

function modelCheck(s) {
  // "Demo", not "practice": practice is what students call the similar-problem mode.
  if (s.vlm_backend === "mock") {
    return check("warn", "Demo mode: answers are pretend", "Ask whoever set up the assistant to restart it with the real AI model turned on.");
  }
  return check("ok", `AI model chosen: ${s.vlm_model}`, "If an answer says it couldn't reach the AI model, start Ollama and ask again.");
}

export function renderStatus(s) {
  $("checks").replaceChildren(
    s.camera_running ? check("ok", "Camera is working") : check("bad", "Camera not found", CAMERA_FIX),
    s.calibrated
      ? check("ok", "Camera and projector are lined up")
      : check("warn", "Not lined up yet", "Press “Line up camera and projector” below."),
    filesCheck(s),
    modelCheck(s),
    s.audio_available
      ? check("ok", "Microphone ready")
      : check("bad", "No microphone found", "You can still type your questions."),
    s.tts_backend === "none"
      ? check("warn", "Answers are shown but not read aloud")
      : check("ok", "Answers are read aloud on the projector page (click it once to allow sound)"),
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
  if (state === "idle") setBadge("ask", idleBadge, "ok");
  else setBadge("ask", ASK_BADGES[state] || state, "warn");
}

export function setIdleBadge(text) {
  idleBadge = text || "ready";
  if (connected && lastState === "idle") setBadge("ask", idleBadge, "ok");
}

// `hint` marks a degraded mode where polling still works, so it reads as a warning rather than a failure.
export function renderConnection(isConnected, hint = "") {
  connected = isConnected || Boolean(hint);
  $("conn-text").textContent = isConnected ? "Connected" : hint ? "Connected (updates every few seconds)" : "Not connected";
  if (!connected) {
    $("status-line").dataset.state = "offline";
    $("status-text").textContent = "Can't reach the assistant. Check its Terminal window is still open; this page reconnects by itself.";
    setBadge("ask", "not connected", "bad");
    return;
  }
  renderState(lastState);
}
