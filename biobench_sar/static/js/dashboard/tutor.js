import { $, request } from "./api.js";
import { note } from "./ask.js";
import { logEvent } from "./log.js";
import { MODE_LABELS, currentProfile, onProfileChange } from "./profile.js";
import { setIdleBadge } from "./status.js";

export const MODES = ["help", "practice", "quiz"];

const CHOSEN_NOTES = {
  help: "Now write your problem on the board and ask for the next step.",
  practice: "Now write a problem on the board and ask for one like it.",
  quiz: "Press Start quiz, or just say or type “I'm ready”.",
};
const QUIZ_SOURCE = "Problems come from the worksheets you tick in step 1, or the assistant makes some up.";
const OFFLINE_HINT = "Check the assistant's Terminal window is still open, then try again.";

// Bumped on every render, so a GET that was in flight during a newer change can't roll the UI back.
let version = 0;
let last = null;

function quizSummary(quiz, active) {
  if (active) return `Problem ${quiz.number} of ${quiz.total} · ${quiz.score} right so far`;
  // A quiz that never got past its first problem has nothing worth reporting.
  if (quiz && (quiz.score > 0 || quiz.number > 1)) return `Last quiz: ${quiz.score} right. Press Start quiz to go again.`;
  return QUIZ_SOURCE;
}

export function renderTutor(data) {
  version += 1;
  last = data;
  // Tutor modes are a math thing; the biology assistant answers every question the same way.
  const math = currentProfile() === "math";
  const mode = MODES.includes(data.mode) ? data.mode : "help";
  const quiz = data.quiz || null;
  const active = Boolean(quiz && quiz.active);
  $("modes").hidden = !math;
  for (const m of MODES) $(`mode-${m}`).setAttribute("aria-pressed", String(m === mode));
  $("quiz-bar").hidden = !math || (!active && mode !== "quiz");
  $("quiz-progress").textContent = quizSummary(quiz, active);
  const problem = $("quiz-problem");
  problem.textContent = active && quiz.problem ? quiz.problem : "";
  problem.hidden = !problem.textContent;
  $("quiz-start-btn").hidden = active;
  $("quiz-stop-btn").hidden = !active;
  const learn = $("learn-line");
  learn.textContent = data.learn_topic ? `Learning: ${data.learn_topic}` : "";
  learn.hidden = !math || !data.learn_topic;
  if (!math) setIdleBadge("ready");
  else setIdleBadge(active ? `quiz · ${quiz.number} of ${quiz.total}` : MODE_LABELS[mode]);
}

export async function refreshTutor() {
  const seen = version;
  const data = await request("/api/tutor");
  if (seen === version) renderTutor(data);
}

function failed(what, err) {
  logEvent("tutor", `${what}: ${err.message}`);
  note(`Couldn't ${what}. ${OFFLINE_HINT}`, "warn");
}

async function chooseMode(mode) {
  const buttons = MODES.map((m) => $(`mode-${m}`));
  for (const b of buttons) b.disabled = true;
  try {
    const data = await request("/api/tutor/mode", { method: "PUT", json: { mode } });
    renderTutor(data);
    const running = data.quiz && data.quiz.active;
    note(running ? "A quiz is already running. Write your answer on the board." : CHOSEN_NOTES[mode], "ok");
  } catch (err) {
    failed("change how the assistant helps", err);
  } finally {
    for (const b of buttons) b.disabled = false;
  }
}

// The assistant reads each problem out before it checks an answer, so the first step is always asking for it.
const STARTED_NOTE = "Quiz started! Say or type “I'm ready” and the assistant will read out the first problem.";

async function quizAction(kind) {
  const button = $(`quiz-${kind}-btn`);
  button.disabled = true;
  try {
    const data = await request(`/api/tutor/quiz/${kind}`, { method: "POST" });
    renderTutor(data);
    note(kind === "start" ? STARTED_NOTE : "Quiz stopped.", "ok");
  } catch (err) {
    failed(kind === "start" ? "start the quiz" : "stop the quiz", err);
  } finally {
    button.disabled = false;
  }
}

export function initTutor() {
  for (const mode of MODES) $(`mode-${mode}`).addEventListener("click", () => chooseMode(mode));
  $("quiz-start-btn").addEventListener("click", () => quizAction("start"));
  $("quiz-stop-btn").addEventListener("click", () => quizAction("stop"));
  onProfileChange(() => {
    if (last) renderTutor(last);
  });
  // An older server has no tutor API: the picker simply stays hidden.
  refreshTutor().catch(() => {});
}
