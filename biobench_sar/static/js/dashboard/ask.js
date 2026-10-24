import { $, el, request, setNote } from "./api.js";

let pending = null;
let serverState = "idle";
// The push-to-talk phase the ask note currently shows, so status events never clear an unrelated note.
let micNote = null;

// A 202 from /api/listen only queues the recording (in wake mode it starts after the current chunk),
// so "speak now" waits for the server's own "listening" status.
const MIC_NOTES = {
  starting: ["Starting microphone…", ""],
  listening: ["Listening — speak now.", "ok"],
  transcribing: ["Transcribing…", ""],
};

function sourcesTag(sources) {
  return sources.length ? el("span", { className: "tag", textContent: `📄 ${sources.join(", ")}` }) : null;
}

function turnNode(item) {
  return el(
    "li",
    { className: "turn", dataset: { state: item.animation_state } },
    el("p", { className: "q", textContent: item.query }),
    el("p", { className: "a", textContent: item.spoken_text }),
    el(
      "div",
      { className: "meta" },
      el("span", { className: "tag tag-state", textContent: item.animation_state }),
      el("span", { textContent: `${item.latency_s.toFixed(1)}s` }),
      el("span", { textContent: `→ (${item.target_x}, ${item.target_y})` }),
      el("span", { textContent: item.had_frame ? `${item.ink_regions} ink regions` : "no camera frame" }),
      sourcesTag(item.sources),
    ),
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
  pending = el("li", { className: "turn pending" }, el("p", { className: "q", textContent: text }), el("p", { className: "a", textContent: "Thinking…" }));
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
  if (err.status === 503) return "No microphone on the server machine. Plug one in and restart the server, or type your question.";
  if (err.status === 409) return "Already recording.";
  return `Microphone unavailable: ${err.message}`;
}

export function initAsk() {
  const input = $("ask-input");
  $("ask-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return input.focus();
    $("ask-btn").disabled = true;
    try {
      await request("/api/query", { method: "POST", json: { text } });
      input.value = "";
      note("Sent — watch the projector.", "ok");
    } catch (err) {
      note(err.status === 409 ? "Still answering the last question — try again in a moment." : `Could not send: ${err.message}`, "warn");
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
