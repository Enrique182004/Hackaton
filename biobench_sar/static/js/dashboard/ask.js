import { $, el, request, setNote } from "./api.js";
import { has, onLanguageChange, t } from "../i18n.js";
import { logEvent } from "./log.js";
import { currentProfile, modeLabel, onProfileChange, words } from "./profile.js";
import { listeningNote } from "./status.js";

let pending = null;
let pendingText = "";
// The turns on screen, newest first, so a language switch can re-write their notes.
let turns = [];
let serverState = "idle";
// The push-to-talk phase the ask note currently shows, so status events never clear an unrelated note.
let micNote = null;

// A 202 from /api/listen only queues the recording (in wake mode it starts after the current chunk),
// so "speak now" waits for the server's own "listening" status.
const MIC_NOTES = {
  starting: [() => t("mic.starting"), ""],
  listening: [listeningNote, "ok"],
  transcribing: [() => t("state.transcribing"), ""],
};

function modeNote(item) {
  if (currentProfile() !== "math") return null;
  if (item.task_kind === "learn") return t("note.learning");
  return has(`mode.${item.mode}`) ? modeLabel(item.mode) : null;
}

// The server's own texts for a model call or pipeline that failed, for servers that send no `failed` flag.
const FAILURE_TEXT = /couldn['’]t reach the AI model|took too long to answer|sent an unexpected reply|something went wrong|couldn['’]t understand my own answer|unknown VLM backend/i;

// A failed call is no answer at all, whatever is on the board; "error" alone also means "the model found a mistake".
function failedTurn(item) {
  if (typeof item.failed === "boolean") return item.failed;
  return FAILURE_TEXT.test(item.spoken_text || "");
}

function errorNote(item, sawDrawing) {
  if (failedTurn(item)) return t("note.failed");
  // A quiz answer can be given by voice, with nothing on the board.
  if (item.task_kind === "quiz") return t("note.quizWrong");
  return sawDrawing ? t("note.mistakeIn", { work: words().work }) : t("note.mistake");
}

function marginNote(item) {
  const parts = [];
  const mode = modeNote(item);
  if (mode) parts.push(mode);
  // "all correct" / "pointed at" only mean something when the assistant actually saw a drawing.
  const sawDrawing = item.had_frame && item.ink_regions > 0;
  const work = words().work;
  if (item.animation_state === "error") {
    parts.push(errorNote(item, sawDrawing));
  } else if (sawDrawing && item.animation_state === "pointing") {
    parts.push(t("note.pointed", { work }));
  } else if (sawDrawing && item.animation_state === "celebrating") {
    parts.push(t("note.correct"));
  }
  if (item.sources.length) parts.push(t("note.read", { sources: item.sources.join(", ") }));
  else parts.push(t("note.noFiles"));
  if (!item.had_frame) parts.push(t("note.noBoard"));
  parts.push(t("note.took", { n: Math.round(item.latency_s) }));
  return parts.join(" · ");
}

function turnNode(item) {
  // In math, spoken_text spells notation out for the voice; display_text keeps "3x + 5 = 20" for reading.
  const answer = el("p", { className: "a", textContent: item.display_text || item.spoken_text });
  // The answer's own language, so a screen reader reads a Spanish answer in Spanish on an English page.
  if (item.language) answer.setAttribute("lang", item.language);
  return el(
    "li",
    { className: "entry", dataset: { state: item.animation_state } },
    el("p", { className: "who", textContent: t("turn.you") }),
    el("p", { className: "q", textContent: item.query }),
    el("p", { className: "who", textContent: t("turn.assistant") }),
    answer,
    el("span", { className: "margin-note", textContent: marginNote(item) }),
  );
}

function pendingNode(text) {
  return el(
    "li",
    { className: "entry pending" },
    el("p", { className: "who", textContent: t("turn.you") }),
    el("p", { className: "q", textContent: text }),
    el("p", { className: "a", textContent: t("turn.thinking") }),
  );
}

function showEmpty() {
  $("timeline-empty").hidden = $("timeline").children.length > 0;
}

function renderTimeline() {
  pending = pending ? pendingNode(pendingText) : null;
  $("timeline").replaceChildren(...(pending ? [pending] : []), ...turns.map(turnNode));
  showEmpty();
}

export function addInteraction(item) {
  if (pending) {
    pending.remove();
    pending = null;
  }
  turns.unshift(item);
  $("timeline").prepend(turnNode(item));
  showEmpty();
}

export function showPending(text) {
  if (pending) pending.remove();
  pendingText = text;
  pending = pendingNode(text);
  $("timeline").prepend(pending);
  showEmpty();
}

export async function loadHistory() {
  const { interactions } = await request("/api/history");
  turns = interactions;
  $("timeline").replaceChildren(...interactions.map(turnNode));
  showEmpty();
}

const MIC_START_TIMEOUT_MS = 10000;
let micTimer = null;

// The Ask step's one note line. Writing it ends whatever push-to-talk phase it showed.
function note(text, tone) {
  micNote = null;
  clearTimeout(micTimer);
  setNote($("ask-msg"), text, tone);
}

// The same line for the mode picker and quiz buttons, so the step stays one screen tall. A microphone that is
// still starting keeps its watchdog, so one that never comes up is still reported.
export function sideNote(text, tone) {
  if (micNote !== "starting") micNote = null;
  setNote($("ask-msg"), text, tone);
}

function showMicNote(phase) {
  note(...MIC_NOTES[phase]);
  micNote = phase;
  // "Getting ready" waits for the server's "listening" status; if that never comes, don't leave it up forever.
  if (phase === "starting") {
    micTimer = setTimeout(() => {
      if (micNote === "starting") note(() => t("mic.didntStart"), "warn");
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
  if (err.status === 503) return () => t("mic.none");
  if (err.status === 409) return () => t("mic.already");
  logEvent("listen", err.message);
  return () => t("mic.broken");
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
  $("examples").replaceChildren(el("span", { className: "scribble", textContent: t("ask.try") }), ...buttons);
}

export function initAsk() {
  const input = $("ask-input");
  renderExamples();
  onProfileChange(() => {
    renderExamples();
    // Margin notes depend on the subject, and history can load before /api/config answers.
    loadHistory().catch(() => {});
  });
  onLanguageChange(() => {
    renderExamples();
    renderTimeline();
  });
  $("ask-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return input.focus();
    $("ask-btn").disabled = true;
    try {
      await request("/api/query", { method: "POST", json: { text } });
      input.value = "";
      note(() => t("ask.sent"), "ok");
    } catch (err) {
      if (err.status === 409) {
        note(() => t("ask.stillAnswering"), "warn");
      } else {
        logEvent("query", err.message);
        note(() => t("ask.sendFailed", { hint: t("hint.offline") }), "warn");
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
