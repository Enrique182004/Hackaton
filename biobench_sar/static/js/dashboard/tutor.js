import { $, request } from "./api.js";
import { sideNote } from "./ask.js";
import { onLanguageChange, t } from "../i18n.js";
import { logEvent } from "./log.js";
import { currentProfile, modeLabel, onProfileChange, profileKnown } from "./profile.js";
import { setIdleBadge } from "./status.js";

export const MODES = ["help", "practice", "quiz"];

// Bumped on every render, so a GET that was in flight during a newer change can't roll the UI back.
let version = 0;
let last = null;

// The server's score counts only answers right on the first try: a fixed answer moves on but scores nothing.
function quizSummary(quiz, active) {
  if (active) return t("tutor.progress", { n: quiz.number, total: quiz.total, score: quiz.score });
  // A quiz that never got past its first problem has nothing worth reporting.
  if (quiz && (quiz.score > 0 || quiz.number > 1)) return t("tutor.last", { score: quiz.score });
  return t("tutor.quizSource");
}

export function renderTutor(data) {
  version += 1;
  last = data;
  // Tutor modes are a math thing; the biology assistant answers every question the same way. The picker's
  // buttons change the server's state, so it waits until the server has said it runs the math subject.
  const math = currentProfile() === "math" && profileKnown();
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
  learn.textContent = data.learn_topic ? t("tutor.learning", { topic: data.learn_topic }) : "";
  learn.hidden = !math || !data.learn_topic;
  if (!math) setIdleBadge(null);
  else if (active) setIdleBadge(() => t("tutor.badge.quiz", { n: quiz.number, total: quiz.total }));
  else setIdleBadge(() => modeLabel(mode));
}

export async function refreshTutor() {
  const seen = version;
  const data = await request("/api/tutor");
  if (seen === version) renderTutor(data);
}

// `what` is the tutor.fail.* key; the log keeps the server's raw reason.
function failed(what, err) {
  logEvent("tutor", `${what}: ${err.message}`);
  sideNote(() => t(`tutor.fail.${what}`, { hint: t("hint.offline") }), "warn");
}

async function chooseMode(mode) {
  const buttons = MODES.map((m) => $(`mode-${m}`));
  for (const b of buttons) b.disabled = true;
  try {
    const data = await request("/api/tutor/mode", { method: "PUT", json: { mode } });
    renderTutor(data);
    const running = data.quiz && data.quiz.active;
    sideNote(() => t(running ? "tutor.running" : `tutor.chosen.${mode}`), "ok");
  } catch (err) {
    failed("mode", err);
  } finally {
    for (const b of buttons) b.disabled = false;
  }
}

async function quizAction(kind) {
  const button = $(`quiz-${kind}-btn`);
  button.disabled = true;
  try {
    const data = await request(`/api/tutor/quiz/${kind}`, { method: "POST" });
    renderTutor(data);
    // The assistant reads each problem out before it checks an answer, so the first step is always asking for it.
    sideNote(() => t(kind === "start" ? "tutor.started" : "tutor.stopped"), "ok");
  } catch (err) {
    failed(kind, err);
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
  onLanguageChange(() => {
    if (last) renderTutor(last);
  });
  // An older server has no tutor API: the picker simply stays hidden.
  refreshTutor().catch(() => {});
}
