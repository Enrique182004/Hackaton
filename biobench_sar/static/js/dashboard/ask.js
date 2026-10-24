import { $, el, request, setNote } from "./api.js";

let pending = null;
let serverState = "idle";
// The push-to-talk phase the ask note currently shows, so status events never clear an unrelated note.
let micNote = null;

// A 202 from /api/listen only queues the recording (in wake mode it starts after the current chunk),
// so "speak now" waits for the server's own "listening" status.
const MIC_NOTES = {
  starting: ["Getting the microphone ready…", ""],
  listening: ["Listening… speak now.", "ok"],
  transcribing: ["Writing down what you said…", ""],
};

const STATE_NOTES = {
  pointing: "pointed at your drawing",
  error: "spotted a problem",
  celebrating: "all correct!",
};

function marginNote(item) {
  const parts = [];
  // "all correct" / "pointed at" only mean something when the assistant actually saw a drawing.
  const sawDrawing = item.had_frame && item.ink_regions > 0;
  if (STATE_NOTES[item.animation_state] && (sawDrawing || item.animation_state === "error")) parts.push(STATE_NOTES[item.animation_state]);
  if (item.sources.length) parts.push(`read ${item.sources.join(", ")}`);
  else parts.push("didn't use your files");
  if (!item.had_frame) parts.push("couldn't see the board");
  parts.push(`${Math.round(item.latency_s)} s`);
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

function note(text, tone) {
  micNote = null;
  setNote($("ask-msg"), text, tone);
}

function showMicNote(phase) {
  note(...MIC_NOTES[phase]);
  micNote = phase;
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
  return `The microphone isn't working right now (${err.message}).`;
}

export function initAsk() {
  const input = $("ask-input");
  for (const button of document.querySelectorAll(".example")) {
    button.addEventListener("click", () => {
      input.value = button.textContent;
      $("ask-form").requestSubmit();
    });
  }
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
      note(err.status === 409 ? "Still answering the last question. Try again in a moment." : `Couldn't send your question (${err.message}).`, "warn");
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
