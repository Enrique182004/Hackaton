import { $, el, formatDuration } from "./api.js";
import { bindText, has, onLanguageChange, plural, t } from "../i18n.js";
import { words } from "./profile.js";
import { setBadge } from "./tabs.js";

export const listeningNote = () => t("state.listening");

let connected = false;
let lastState = "idle";
let lastStatus = null;
// What the Ask badge says while nothing is happening: the tutor mode once it is known.
let idleBadge = () => t("badge.ready");

function check(tone, what, how = "") {
  const mark = { ok: "✓", warn: "!", bad: "✗" }[tone];
  return el(
    "li",
    { dataset: { tone } },
    el("span", { className: "mark", textContent: mark, "aria-label": t(tone === "ok" ? "check.ok" : "check.attention") }),
    el("span", {}, el("span", { className: "what", textContent: what }), how ? el("span", { className: "how", textContent: how }) : null),
  );
}

function filesCheck(s) {
  const w = words();
  const total = s.total_documents;
  const active = s.active_documents ?? total;
  if (total === undefined) {
    return s.doc_chunks > 0 ? check("ok", t("check.files.loaded")) : check("warn", w.filesMissing, t("check.files.addSome"));
  }
  if (total === 0) return check("warn", w.filesMissing, t("check.files.addSome"));
  if (active === 0) return check("warn", w.filesNoneTicked, t("check.files.tickOne"));
  return check("ok", plural("check.files.using", total, { active, total }));
}

function modelCheck(s) {
  // "Demo", not "practice": practice is what students call the similar-problem mode.
  if (s.vlm_backend === "mock") return check("warn", t("check.model.demo"), t("check.model.demoFix"));
  return check("ok", t("check.model.ok", { model: s.vlm_model }), t("check.model.hint"));
}

// /api/status.glare is null without a camera and missing on servers without the glare check.
function glareOf(s) {
  const g = s.glare;
  if (!s.camera_running || !g || typeof g !== "object") return null;
  if (g.severe) return "severe";
  return Array.isArray(g.hotspots) && g.hotspots.length ? "small" : "none";
}

function glareCheck(glare) {
  if (glare === "severe") return check("bad", t("check.glare.severe"), t("check.glare.severeHow"));
  if (glare === "small") return check("warn", t("check.glare.small"), t("check.glare.smallHow"));
  return glare === "none" ? check("ok", t("check.glare.ok")) : null;
}

export function renderStatus(s) {
  lastStatus = s;
  const glare = glareOf(s);
  $("checks").replaceChildren(
    ...[
      s.camera_running ? check("ok", t("check.camera.ok")) : check("bad", t("check.camera.bad"), t("check.camera.fix")),
      glareCheck(glare),
      s.calibrated ? check("ok", t("check.calib.ok")) : check("warn", t("check.calib.warn"), t("check.calib.fix")),
      filesCheck(s),
      modelCheck(s),
      s.audio_available ? check("ok", t("check.mic.ok")) : check("bad", t("check.mic.bad"), t("check.mic.fix")),
      s.tts_backend === "none" ? check("warn", t("check.voice.off")) : check("ok", t("check.voice.ok")),
    ].filter(Boolean),
  );
  $("fps").textContent = s.camera_running ? t("board.fps", { n: (s.fps ?? 0).toFixed(0) }) : "–";
  $("latency").textContent = s.last_latency_s == null ? "–" : t("board.latency", { n: s.last_latency_s.toFixed(1) });
  $("uptime").textContent = formatDuration(s.uptime_s);
  $("viewport").dataset.live = s.camera_running ? "1" : "0";
  // Glare comes before lining up: a hotspot over the checkerboard is one reason lining up fails.
  if (!s.camera_running) setBadge("board", () => t("badge.board.noCamera"), "bad");
  else if (glare === "severe") setBadge("board", () => t("badge.board.glare"), "bad");
  else if (!s.calibrated) setBadge("board", () => t("badge.board.notLinedUp"), "warn");
  else setBadge("board", () => t("badge.board.ready"), "ok");
}

function sentence(state, detail) {
  const text = has(`state.${state}`) ? t(`state.${state}`) : state;
  return detail === "busy" ? `${text} ${t("state.busy")}` : text;
}

export function renderState(state, detail) {
  lastState = state;
  if (!connected) return;
  const line = $("status-line");
  line.dataset.state = state;
  bindText($("status-text"), () => sentence(state, detail));
  $("listen-btn").dataset.live = state === "listening" ? "1" : "0";
  if (state === "idle") setBadge("ask", idleBadge, "ok");
  else setBadge("ask", () => (has(`badge.ask.${state}`) ? t(`badge.ask.${state}`) : state), "warn");
}

// `label` is a function of the current language, or nothing for "ready".
export function setIdleBadge(label) {
  idleBadge = label || (() => t("badge.ready"));
  if (connected && lastState === "idle") setBadge("ask", idleBadge, "ok");
}

// `hint` marks a degraded mode where polling still works, so it reads as a warning rather than a failure.
export function renderConnection(isConnected, hint = "") {
  connected = isConnected || Boolean(hint);
  bindText($("conn-text"), () => t(isConnected ? "conn.connected" : hint ? "conn.polling" : "conn.offline"));
  if (!connected) {
    $("status-line").dataset.state = "offline";
    bindText($("status-text"), () => t("conn.unreachable"));
    setBadge("ask", () => t("badge.offline"), "bad");
    return;
  }
  renderState(lastState);
}

// The checklist is rebuilt from the last status; the status line and badges re-render themselves.
onLanguageChange(() => {
  if (lastStatus) renderStatus(lastStatus);
});
