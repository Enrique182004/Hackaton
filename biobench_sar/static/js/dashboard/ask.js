import { $, el, request, setNote } from "./api.js";
import { logEvent } from "./log.js";
import { currentProfile, onProfileChange, words } from "./profile.js";
import { LISTENING_NOTE } from "./status.js";
import { MODE_LABELS } from "./tutor.js";

let pending = null;
let serverState = "idle";
// The push-to-talk phase the ask note currently shows, so status events never clear an unrelated note.
let micNote = null;

// A 202 from /api/listen only queues the recording (in wake mode it starts after the current chunk),
// so "speak now" waits for the server's own "listening" status.
const MIC_NOTES = {
  starting: ["Getting the microphone ready…", ""],
  listening: [LISTENING_NOTE, "ok"],
  transcribing: ["Writing down what you said…", ""],
};

function modeNote(item) {
  if (currentProfile() !== "math") return null;
  if (item.task_kind === "learn") return "learning something new";
  return MODE_LABELS[item.mode] || null;
}

function marginNote(item) {
  const parts = [];
  const mode = modeNote(item);
  if (mode) parts.push(mode);
  // "all correct" / "pointed at" / "a mistake" only mean something when the assistant actually saw a drawing.
  const sawDrawing = item.had_frame && item.ink_regions > 0;
  const work = words().work;
  if (item.animation_state === "error") {
    parts.push(sawDrawing ? `found a mistake in ${work}` : "something went wrong, so this isn't a real answer");
  } else if (sawDrawing && item.animation_state === "pointing") {
    parts.push(`pointed at ${work}`);
  } else if (sawDrawing && item.animation_state === "celebrating") {
    parts.push("all correct!");
  }
  if (item.sources.length) parts.push(`read ${item.sources.join(", ")}`);
  else parts.push("didn't use your files");
  if (!item.had_frame) parts.push("couldn't see the board");
  parts.push(`took ${Math.round(item.latency_s)} s`);
  return parts.join(" · ");
}

function turnNode(item) {
  return el(
    "li",
    { className: "entry", dataset: { state: item.animation_state } },
    el("p", { className: "who", textContent: "You asked" }),
    el("p", { className: "q", textContent: item.query }),
    el("p", { className: "who", textContent: "Assistant" }),
    el("p", { className: "a", textContent: item.spoken_text }),
    el("span", { className: "margin-note", textContent: marginNote(item) }),
  );
}

function showEmpty() {
  $("timeline-empty").hidden = $("timeline").children.length > 0;
}

export function addInteraction(item) {
  if (pending) {
    pending.remove();
    pending = null;
  }
  $("timeline").prepend(turnNode(item));
  showEmpty();
}

export function showPending(text) {
  if (pending) pending.remove();
  pending = el(
    "li",
    { className: "entry pending" },
    el("p", { className: "who", textContent: "You asked" }),
    el("p", { className: "q", textContent: text }),
    el("p", { className: "a", textContent: "Thinking…" }),
  );
  $("timeline").prepend(pending);
  showEmpty();
}

export async function loadHistory() {
  const { interactions } = await request("/api/history");
  $("timeline").replaceChildren(...interactions.map(turnNode));
  showEmpty();
}

const MIC_START_TIMEOUT_MS = 10000;
let micTimer = null;

function note(text, tone) {
  micNote = null;
  clearTimeout(micTimer);
  setNote($("ask-msg"), text, tone);
}

function showMicNote(phase) {
  note(...MIC_NOTES[phase]);
  micNote = phase;
  // "Getting ready" waits for the server's "listening" status; if that never comes, don't leave it up forever.
  if (phase === "starting") {
    micTimer = setTimeout(() => {
      if (micNote === "starting") note("The microphone didn't start. Try Talk again, or type your question.", "warn");
    }, MIC_START_TIMEOUT_MS);
  }
}

// Called with every server status (socket push, or the status poll while the socket is down).
export function onListenState(state) {
  serverState = state;
  if (state === "listening" || state === "transcribing") showMicNote(state);
  // Only idle ends a recording: a query still running can emit other states mid-recording, and
  // "starting" survives idle because a previous query or wake chunk may finish before recording begins.
  else if (state === "idle" && (micNote === "listening" || micNote === "transcribing")) note("");
}

function listenError(err) {
  if (err.status === 503) return "No microphone found. Plug one in and restart the app, or type your question.";
  if (err.status === 409) return "Already listening.";
  logEvent("listen", err.message);
  return "The microphone isn't working right now. Try Talk again, or type your question.";
}

function renderExamples() {
  const input = $("ask-input");
  const buttons = words().examples.map((text) => {
    const button = el("button", { type: "button", className: "example", textContent: text });
    button.addEventListener("click", () => {
      input.value = text;
      $("ask-form").requestSubmit();
    });
    return button;
  });
  $("examples").replaceChildren(el("span", { className: "scribble", textContent: "try:" }), ...buttons);
}

export function initAsk() {
  const input = $("ask-input");
  renderExamples();
  onProfileChange(() => {
    renderExamples();
    // Margin notes depend on the subject, and history can load before /api/config answers.
    loadHistory().catch(() => {});
  });
  $("ask-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return input.focus();
    $("ask-btn").disabled = true;
    try {
      await request("/api/query", { method: "POST", json: { text } });
      input.value = "";
      note("Sent! Watch the board for the answer.", "ok");
    } catch (err) {
      if (err.status === 409) {
        note("Still answering the last question. Try again in a moment.", "warn");
      } else {
        logEvent("query", err.message);
        note("Couldn't send your question. Check the assistant's Terminal window is still open, then try again.", "warn");
      }
    } finally {
      $("ask-btn").disabled = false;
    }
  });
  $("listen-btn").addEventListener("click", async () => {
    try {
      await request("/api/listen", { method: "POST" });
      // The status push can beat this response; never step back from a recording already under way.
      showMicNote(serverState === "listening" || serverState === "transcribing" ? serverState : "starting");
    } catch (err) {
      note(listenError(err), err.status === 503 ? "bad" : "warn");
    }
  });
  showEmpty();
}
