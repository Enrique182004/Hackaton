import { bindText, languageFor, setLanguage, t } from "./i18n.js";

const body = document.body;
const PROJ_W = parseFloat(body.dataset.projectorWidth) || 1920;
const PROJ_H = parseFloat(body.dataset.projectorHeight) || 1080;
const FRAME = 160;
// Pointing sprite: fingertip sits at (153,56) in the 160px frame; other states anchor at the centre.
const POINT_TIP = { x: 153, y: 56 };
const CENTER = { x: FRAME / 2, y: FRAME / 2 };
const EDGE = 24; // min gap to the window edge; leaves room for the celebrate hop rotating the bubble
const STATES = ["idle", "walking", "pointing", "celebrating", "error"];
const MOVE_MS = 1500;
// While the student asks and the assistant reads the board, the problem card steps aside: projected text in the
// camera picture looks like ink, and the model would read the printed problem as the student's work.
const CAPTURE_STATES = ["listening", "transcribing", "thinking"];
// Server states with a proj.<state> label; anything else is shown as sent.
const LABELLED_STATES = ["idle", "listening", "transcribing", "thinking", "speaking"];
// The ink an answer is about ([x, y, w, h], projector px) gets a glowing outline that pulses for about 2 s and fades
// after HIGHLIGHT_MS, or sooner when the next answer or question comes.
const HIGHLIGHT_MS = 8000;
const HIGHLIGHT_FADE_MS = 800; // style.css #highlight.fading
// Projector px between the ink and the outline, so the outline frames the writing and never lights it up.
const HIGHLIGHT_GAP = 14;
const HIGHLIGHT_BORDER = 3; // CSS px: style.css #highlight border width
const HIGHLIGHT_BORDER_CONTRAST = 6; // the same in high contrast (body[data-contrast="on"] #highlight)
const HIGHLIGHT_TONES = { error: "error", celebrating: "celebrate" };
// Projected UI looks like ink to the camera, so the server is told where it is ("ui_rects", projector px) and leaves
// out ink regions mostly under it. Sent on every change and at least once a second; the server ignores rects older
// than 3 s, so a closed or hidden page stops masking by itself.
const UI_TICK_MS = 250;
const UI_HEARTBEAT_MS = 1000;
const UI_PAD = 10; // CSS px: the glow and the idle bob reach past each element's own box
const UI_JITTER = 12; // projector px: moving less than this (the idle bob) is not a change worth sending
// The room's accessibility settings (accessibility.py) live on <body> as data-* attributes, which style.css reads:
// data-text-size, data-captions ("stay" | "timed"), data-calm and data-contrast ("on" | "off").
const TEXT_SIZES = ["normal", "large", "xlarge"];

const $ = (id) => document.getElementById(id);
const avatar = $("avatar");
const bubble = $("bubble");
const statusEl = $("status");
const statusLabel = $("status-label");
const transcriptEl = $("transcript");
const textOverlay = $("text-overlay");
const textInput = $("text-input");
const debugEl = $("debug");
const debugConn = $("debug-conn");
const debugEvent = $("debug-event");
const crosshair = $("crosshair");
const problemCard = $("problem-card");
const problemLabel = $("problem-card-label");
const problemText = $("problem-card-text");
const highlightEl = $("highlight");

let currentState = "idle";
let pendingState = "idle";
let moveTimer = null;
let lastTarget = null;
let audio = null;
let bubbleTimer = null;
let transcriptTimer = null;
let connected = false;
let statusSinceConnect = false; // a server status arrived since the last disconnect
let offlineTimer = null;
let offlineHintTimer = null;
let tutorVersion = 0; // bumped per "tutor" render so an older GET can't roll the card back
let languageVersion = 0; // the same for "language"
let accessVersion = 0; // and for "accessibility"
let highlight = null; // [x, y, w, h] in projector px while the outline is on (not fading)
let highlightTimer = null;
let sentRects = null; // the last "ui_rects" sent, and when
let sentAt = 0;
let uiQueued = false;

// English until the server says which language it answers in.
setLanguage("en");

function setState(state) {
  if (!STATES.includes(state)) state = "idle";
  currentState = state;
  STATES.forEach((s) => avatar.classList.remove("state-" + s));
  avatar.classList.add("state-" + state);
}

function toPixels(x, y) {
  return { x: (x / PROJ_W) * window.innerWidth, y: (y / PROJ_H) * window.innerHeight };
}

function placeAvatar(state, tx, ty) {
  const anchor = state === "pointing" ? POINT_TIP : CENTER;
  const p = toPixels(tx, ty);
  avatar.style.left = Math.round(p.x - anchor.x) + "px";
  avatar.style.top = Math.round(p.y - anchor.y) + "px";
  // Bubble sits above the avatar; flip/shift it so it never leaves the projected area.
  fitBubble();
}

function clampX(x, width) {
  return Math.max(EDGE, Math.min(x, window.innerWidth - width - EDGE));
}

// Uses the avatar's destination (style.top), not its mid-transition position, plus the real bubble height.
function fitBubble() {
  const top = parseFloat(avatar.style.top) || 0;
  const h = bubble.classList.contains("hidden") ? 120 : bubble.offsetHeight;
  let below = top - h - 8 < 0;
  // Centre the bubble over the avatar, then clamp it inside the window: the avatar itself may hang off-screen
  // when it points near an edge, so aligning to the avatar's edge is not enough.
  const left = parseFloat(avatar.style.left) || 0;
  const width = bubble.classList.contains("hidden") ? 420 : bubble.offsetWidth;
  let x = clampX(left + avatar.offsetWidth / 2 - width / 2, width);
  // Never over the highlighted writing: beside it, else on the avatar's other side; if nothing fits, as it was.
  const hl = highlight && highlightBox(highlight);
  if (hl) {
    const yOf = (under) => (under ? top + avatar.offsetHeight + 4 : top - h - 4);
    const hits = (bx, under) => {
      const by = yOf(under);
      return bx < hl.left + hl.width && bx + width > hl.left && by < hl.top + hl.height && by + h > hl.top;
    };
    const fits = (under) => (under ? yOf(true) + h <= window.innerHeight : yOf(false) >= 0);
    if (hits(x, below)) {
      const options = [
        [clampX(hl.left - width - EDGE, width), below],
        [clampX(hl.left + hl.width + EDGE, width), below],
        [x, !below],
      ];
      const free = options.find(([bx, under]) => fits(under) && !hits(bx, under));
      if (free) [x, below] = free;
    }
  }
  avatar.classList.toggle("bubble-below", below);
  bubble.style.left = Math.round(x - left) + "px";
}

function isRect(r) {
  return Array.isArray(r) && r.length === 4 && r.every(Number.isFinite) && r[2] > 0 && r[3] > 0;
}

function highlightBorder() {
  return body.dataset.contrast === "on" ? HIGHLIGHT_BORDER_CONTRAST : HIGHLIGHT_BORDER;
}

// The outline's box in window px: the ink plus the gap, plus the border (drawn inside the box), kept on screen.
function highlightBox([x, y, w, h]) {
  const border = highlightBorder();
  const a = toPixels(x - HIGHLIGHT_GAP, y - HIGHLIGHT_GAP);
  const b = toPixels(x + w + HIGHLIGHT_GAP, y + h + HIGHLIGHT_GAP);
  const left = Math.max(0, a.x - border);
  const top = Math.max(0, a.y - border);
  const right = Math.min(window.innerWidth, b.x + border);
  const bottom = Math.min(window.innerHeight, b.y + border);
  return { left, top, width: Math.max(0, right - left), height: Math.max(0, bottom - top) };
}

function placeHighlight() {
  if (!highlight) return;
  const box = highlightBox(highlight);
  for (const side of ["left", "top", "width", "height"]) highlightEl.style[side] = Math.round(box[side]) + "px";
}

function showHighlight(rect, state) {
  const same = highlight !== null && rect.every((v, i) => v === highlight[i]);
  highlight = rect.slice();
  placeHighlight();
  const tone = HIGHLIGHT_TONES[state];
  if (tone) highlightEl.dataset.tone = tone;
  else delete highlightEl.dataset.tone;
  highlightEl.classList.remove("hidden", "fading");
  if (!same) {
    // A new region pulses again: the class comes off, a reflow, and it goes back on.
    highlightEl.classList.remove("pulse");
    void highlightEl.offsetWidth;
    highlightEl.classList.add("pulse");
  }
  clearTimeout(highlightTimer);
  highlightTimer = setTimeout(() => clearHighlight(), HIGHLIGHT_MS);
  fitBubble();
  uiChanged();
}

function hideHighlight() {
  highlightEl.classList.add("hidden");
  highlightEl.classList.remove("fading", "pulse");
  uiChanged();
}

// Fades out, or with `now` goes at once: a question is starting and the camera is about to read the board.
function clearHighlight(now = false) {
  clearTimeout(highlightTimer);
  highlightTimer = null;
  highlight = null;
  if (highlightEl.classList.contains("hidden")) return;
  if (now) return hideHighlight();
  highlightEl.classList.add("fading");
  highlightTimer = setTimeout(hideHighlight, HIGHLIGHT_FADE_MS);
}

// Servers without highlights send no "highlight" key at all; those answers leave the outline alone.
function applyHighlight(d, state) {
  if (!Object.hasOwn(d, "highlight")) return;
  if (isRect(d.highlight)) showHighlight(d.highlight, state);
  else clearHighlight();
}

// Visible on the board: not display:none (itself or a parent), and not the problem card stepping aside.
function onBoard(node) {
  for (let n = node; n && n !== body; n = n.parentNode) if (n.classList.contains("hidden")) return false;
  return !node.classList.contains("capturing");
}

// The outline goes as its four edges, not its whole box: the server drops ink mostly under a UI rect, and the
// writing inside the outline is exactly what the student is working on.
function edges(r) {
  const band = highlightBorder();
  return [
    { left: r.left, top: r.top, width: r.width, height: band },
    { left: r.left, top: r.top + r.height - band, width: r.width, height: band },
    { left: r.left, top: r.top, width: band, height: r.height },
    { left: r.left + r.width - band, top: r.top, width: band, height: r.height },
  ];
}

// [x, y, w, h] in projector px for everything the projector is drawing right now.
function uiRects() {
  if ($("calibration-img")) return []; // the checkerboard covers the whole board
  const sx = PROJ_W / window.innerWidth;
  const sy = PROJ_H / window.innerHeight;
  const boxes = [];
  // [what is drawn, what decides whether it shows]: the typing box shows with its overlay.
  const parts = [[avatar], [bubble], [problemCard], [highlightEl], [transcriptEl], [statusEl], [textInput, textOverlay], [debugEl]];
  for (const [node, holder = node] of parts) {
    if (!onBoard(holder) || !onBoard(node)) continue;
    const r = node.getBoundingClientRect();
    if (!(r.width > 0 && r.height > 0)) continue;
    boxes.push(...(node === highlightEl ? edges(r) : [r]));
  }
  const out = [];
  for (const r of boxes) {
    const x0 = Math.max(0, (r.left - UI_PAD) * sx);
    const y0 = Math.max(0, (r.top - UI_PAD) * sy);
    const x1 = Math.min(PROJ_W, (r.left + r.width + UI_PAD) * sx);
    const y1 = Math.min(PROJ_H, (r.top + r.height + UI_PAD) * sy);
    if (x1 - x0 >= 1 && y1 - y0 >= 1) out.push([Math.round(x0), Math.round(y0), Math.round(x1 - x0), Math.round(y1 - y0)]);
  }
  return out;
}

function sameRects(a, b) {
  return a.length === b.length && a.every((r, i) => r.every((v, j) => Math.abs(v - b[i][j]) <= UI_JITTER));
}

// `force` sends even an unchanged list. Never while disconnected: socket.io would replay stale rects later.
function reportUi(force = false) {
  if (!socket || !connected || document.hidden) return;
  const rects = uiRects();
  const now = Date.now();
  if (!force && sentRects && sameRects(rects, sentRects) && now - sentAt < UI_HEARTBEAT_MS) return;
  sentRects = rects;
  sentAt = now;
  socket.emit("ui_rects", { rects });
}

// Several changes in one event (avatar, bubble, outline) go out as one message, once the layout has them all.
function uiChanged() {
  if (uiQueued) return;
  uiQueued = true;
  queueMicrotask(() => {
    uiQueued = false;
    reportUi(true);
  });
}

// Captions held until the next question (accessibility "captions_stay"): no timer takes the bubble down.
function captionsStay() {
  return body.dataset.captions === "stay";
}

// Single owner of bubbleTimer: every (re)arm clears the previous one so stale timers can't hide a newer bubble.
function armBubbleTimer(ms) {
  clearTimeout(bubbleTimer);
  bubbleTimer = ms && !captionsStay() ? setTimeout(hideBubbleUnlessSpeaking, ms) : null;
}

function showBubble(text, holdMs) {
  armBubbleTimer(0);
  if (!text) { bubble.classList.add("hidden"); return; }
  bubble.textContent = text;
  bubble.classList.remove("hidden");
  fitBubble();
  armBubbleTimer(holdMs);
  uiChanged();
}

// Clear `audio` before touching the element: src="" fires an "error" event on it, and the
// listeners ignore events from any element that is no longer the current one.
function releaseAudio() {
  const a = audio;
  audio = null;
  if (!a) return;
  a.pause();
  a.src = "";
}

// A timer armed before captions were set to stay leaves the caption up too.
function hideBubbleUnlessSpeaking() {
  if (captionsStay() || (audio && !audio.paused && !audio.ended)) return;
  bubble.classList.add("hidden");
  uiChanged();
}

// A held caption goes when a real new question starts (its words, or the board being read), so it never covers
// the board in the camera picture. Timed captions keep their own timer.
function dropHeldCaption() {
  if (!captionsStay() || bubble.classList.contains("hidden")) return;
  bubble.classList.add("hidden");
  uiChanged();
}

function onAvatarUpdate(d) {
  const finalState = d.animation_state || "idle";
  const hasTarget = Number.isFinite(d.target_x) && Number.isFinite(d.target_y);
  pendingState = finalState;
  clearTimeout(moveTimer);
  // Before the bubble, so the bubble can keep clear of the highlighted writing.
  applyHighlight(d, finalState);

  if (hasTarget) {
    lastTarget = { x: d.target_x, y: d.target_y };
    setState("walking");
    placeAvatar(finalState, d.target_x, d.target_y);
    // transitionend can be skipped (no movement / hidden tab), so use a timer as the single source of truth.
    moveTimer = setTimeout(() => {
      setState(pendingState);
      uiChanged();
    }, MOVE_MS + 50);
    updateCrosshair();
    uiChanged();
  } else {
    setState(finalState);
  }
  // Held captions: the student's words stay up until this answer arrives.
  if (captionsStay()) {
    clearTranscript();
    uiChanged();
  }
  // display_text keeps math in normal notation (3x + 5 = 20); spoken_text spells it out for the voice.
  const text = d.display_text || d.spoken_text || "";
  showBubble(text, text ? Math.max(4000, text.length * 80) : 0);
}

function onSpeak(d) {
  releaseAudio();
  applyHighlight(d, pendingState);
  const text = d.display_text || d.text || d.spoken_text;
  if (text) showBubble(text); else armBubbleTimer(0);
  if (!d.url) { armBubbleTimer(4000); return; }
  const a = new Audio(d.url);
  audio = a;
  const done = () => {
    if (audio !== a) return; // superseded by a newer speak or a resync
    releaseAudio();
    armBubbleTimer(1200);
  };
  a.addEventListener("ended", done);
  a.addEventListener("error", done);
  a.play().catch(done); // autoplay may be blocked until the first key press
}

function onStatus(d) {
  statusSinceConnect = true;
  renderStatus(d);
}

function renderStatus(d) {
  const s = d.state || "idle";
  clearTimeout(offlineHintTimer);
  statusEl.dataset.state = s;
  // The corner is read from across the room, so it says what to do rather than the internal state name.
  bindText(statusLabel, () => (d.detail === "busy" ? t("proj.busy") : LABELLED_STATES.includes(s) ? t(`proj.${s}`) : s));
  const capturing = CAPTURE_STATES.includes(s);
  problemCard.classList.toggle("capturing", capturing);
  // A new question: the last answer's outline is stale, and the camera must not see it as writing.
  if (capturing) clearHighlight(true);
  // Like the quiz card: the camera reads the board right after "thinking", so a held caption goes as soon as a
  // question starts (the microphone opening gives seconds of margin; "thinking" alone may come too late).
  if (capturing) dropHeldCaption();
  uiChanged();
}

function onTutor(d) {
  tutorVersion += 1;
  const quiz = d.quiz && d.quiz.active ? d.quiz : null;
  // Outside a quiz the card shows the problem the assistant dictated or read off the board (tutor.py show_problem).
  const shown = !quiz && d.problem && d.problem.text ? d.problem : null;
  problemCard.classList.toggle("hidden", !quiz && !shown);
  if (quiz) {
    bindText(problemLabel, () => t("proj.quiz", { n: quiz.number, total: quiz.total }));
    // No problem text means the assistant makes problems up and reads them out itself.
    problemText.textContent = quiz.problem || "";
  } else if (shown) {
    bindText(problemLabel, () => t(shown.source === "board" ? "proj.problem.board" : "proj.problem.dictated"));
    problemText.textContent = shown.text;
  }
  uiChanged();
}

// The corner and the problem card speak the language the assistant answers in; the bubble is the answer itself.
function onLanguage(d) {
  languageVersion += 1;
  const lang = languageFor(d);
  if (lang) setLanguage(lang);
}

function refreshLanguage() {
  const seen = languageVersion;
  if (typeof fetch !== "function") return;
  fetch("/api/language")
    .then((r) => (r.ok ? r.json() : null))
    .then((d) => { if (d && seen === languageVersion) onLanguage(d); })
    .catch(() => {});
}

// Unknown values are ignored, so a newer server can't put the page in a state style.css doesn't know.
function applyAccessibility(d) {
  const held = captionsStay();
  if (TEXT_SIZES.includes(d.text_size)) body.dataset.textSize = d.text_size;
  if (typeof d.captions_stay === "boolean") body.dataset.captions = d.captions_stay ? "stay" : "timed";
  if (typeof d.calm === "boolean") body.dataset.calm = d.calm ? "on" : "off";
  if (typeof d.high_contrast === "boolean") body.dataset.contrast = d.high_contrast ? "on" : "off";
  // Captions switched off while one is held: it goes like an answer without audio, the student's words after 5 s.
  if (held && !captionsStay()) {
    if (!bubble.classList.contains("hidden")) armBubbleTimer(4000);
    if (!transcriptEl.classList.contains("hidden")) armTranscriptTimer();
  }
  // Bigger text and a thicker outline change the boxes: place them again and tell the server at once.
  placeHighlight();
  fitBubble();
  uiChanged();
}

// Text size, held captions, calm mode and high contrast, set for the whole room from the dashboard.
function onAccessibility(d) {
  accessVersion += 1;
  applyAccessibility(d);
}

// Changes made while the page was away come with the GET on every connect; Flask painted the first ones.
function refreshAccessibility() {
  const seen = accessVersion;
  if (typeof fetch !== "function") return;
  fetch("/api/accessibility")
    .then((r) => (r.ok ? r.json() : null))
    .then((d) => { if (d && seen === accessVersion) onAccessibility(d); })
    .catch(() => {});
}

// "tutor" events missed while disconnected would leave a stale card, so ask the server once per connect.
function refreshTutor() {
  const seen = tutorVersion;
  problemCard.classList.add("hidden");
  if (typeof fetch !== "function") return;
  fetch("/api/tutor")
    .then((r) => (r.ok ? r.json() : null))
    .then((d) => { if (d && seen === tutorVersion) onTutor(d); })
    .catch(() => {});
}

function clearTranscript() {
  clearTimeout(transcriptTimer);
  transcriptEl.textContent = "";
  transcriptEl.classList.add("hidden");
}

// The student's words go after 5 s, or with captions held when the answer arrives (onAvatarUpdate).
function armTranscriptTimer() {
  clearTimeout(transcriptTimer);
  transcriptTimer = null;
  if (captionsStay()) return;
  transcriptTimer = setTimeout(() => {
    transcriptEl.classList.add("hidden");
    uiChanged();
  }, 5000);
}

function onTranscript(d) {
  transcriptEl.textContent = d.text || "";
  transcriptEl.classList.toggle("hidden", !d.text);
  if (d.text) dropHeldCaption();
  armTranscriptTimer();
  uiChanged();
}

// Black is invisible on the projector, so "offline" is a bright amber frame plus a status-corner label.
function setOffline(key = "proj.offline") {
  clearTimeout(offlineTimer);
  clearTimeout(offlineHintTimer);
  body.classList.add("offline");
  statusEl.dataset.state = "offline";
  bindText(statusLabel, () => t(key));
}

function hintOffline() {
  setOffline("proj.notSent");
  offlineHintTimer = setTimeout(() => {
    if (!connected) bindText(statusLabel, () => t("proj.offline"));
  }, 2000);
}

// A reconnect may follow missed events (e.g. hide_calibration), so drop all transient UI. The
// server pushes its real status on connect and that push can land before this handler runs, so
// "idle" is only a placeholder until it arrives, never an overwrite.
function resync() {
  body.classList.remove("offline");
  hideCalibration();
  clearTranscript();
  releaseAudio();
  armBubbleTimer(0);
  bubble.classList.add("hidden");
  clearTimeout(moveTimer);
  pendingState = "idle";
  setState("idle");
  clearHighlight(true);
  if (!statusSinceConnect) renderStatus({ state: "idle" });
  refreshTutor();
  refreshLanguage();
  refreshAccessibility();
}

function showCalibration(d) {
  hideCalibration();
  const img = document.createElement("img");
  img.id = "calibration-img";
  img.alt = "";
  img.src = d.url || "/calibration.png";
  document.body.appendChild(img);
  uiChanged();
}

function hideCalibration() {
  const img = $("calibration-img");
  if (img) img.remove();
  uiChanged();
}

function updateCrosshair() {
  if (!lastTarget) return;
  const p = toPixels(lastTarget.x, lastTarget.y);
  crosshair.style.left = p.x + "px";
  crosshair.style.top = p.y + "px";
  crosshair.classList.toggle("hidden", debugEl.classList.contains("hidden"));
}

function logEvent(name, data) {
  bindText(debugEvent, name + " " + JSON.stringify(data, null, 1));
}

function setConn(key, vars) {
  bindText(debugConn, () => t(key, vars));
}

// Socket.IO may fail to load (offline venue); the page still renders and keys are no-ops.
const socket = typeof io === "function" ? io() : null;
if (socket) {
  socket.on("connect", () => {
    connected = true;
    setConn("proj.debug.connected");
    resync();
    uiChanged();
  });
  socket.on("disconnect", () => { connected = false; statusSinceConnect = false; setConn("proj.debug.disconnected"); setOffline(); });
  socket.on("connect_error", (e) => {
    connected = false;
    statusSinceConnect = false;
    setConn("proj.debug.error", { message: e && e.message });
    setOffline();
  });
  // Server-emitted "interaction" events are for the dashboard; the projector deliberately has no handler.
  const handlers = {
    avatar_update: onAvatarUpdate, speak: onSpeak, status: onStatus, transcript: onTranscript,
    show_calibration: showCalibration, hide_calibration: hideCalibration, tutor: onTutor, language: onLanguage,
    accessibility: onAccessibility,
  };
  Object.keys(handlers).forEach((name) => {
    socket.on(name, (data) => {
      logEvent(name, data);
      try { handlers[name](data || {}); } catch (err) { console.error(name, err); }
    });
  });
} else {
  setConn("proj.debug.noClient");
  setOffline();
}

// Never emit while disconnected: socket.io would buffer and replay stale presses/queries on reconnect.
function emit(name, payload) {
  if (!socket || !connected) { hintOffline(); return false; }
  socket.emit(name, payload || {});
  return true;
}

// Grace period so the initial handshake doesn't flash the offline frame.
if (socket) offlineTimer = setTimeout(() => { if (!connected) setOffline(); }, 1500);
// Catches what no handler announces (the avatar mid-walk) and keeps the server's copy fresh.
setInterval(reportUi, UI_TICK_MS);
document.addEventListener("visibilitychange", () => { if (!document.hidden) uiChanged(); });

function openTextInput() {
  // A typed question has no "listening" state: the held caption goes now, before the board is read.
  dropHeldCaption();
  textOverlay.classList.remove("hidden");
  textInput.value = "";
  textInput.focus();
  uiChanged();
}

function closeTextInput() {
  textOverlay.classList.add("hidden");
  textInput.blur();
  uiChanged();
}

textInput.addEventListener("keydown", (e) => {
  e.stopPropagation(); // keep hotkeys (Space, C, T) from firing while typing
  if (e.key === "Enter") {
    if (e.isComposing) return;
    const text = textInput.value.trim();
    // Keep the overlay (and the typed text) open if offline so the question isn't lost.
    if (text && !emit("text_query", { text })) return;
    closeTextInput();
  } else if (e.key === "Escape") {
    closeTextInput();
  }
});

document.addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || e.isComposing) return;
  if (document.activeElement === textInput) return;
  const k = typeof e.key === "string" ? e.key.toLowerCase() : "";
  if (e.code === "Space" || k === " ") { e.preventDefault(); if (!e.repeat) emit("trigger_listen"); }
  else if (e.repeat) return;
  else if (k === "t") { e.preventDefault(); openTextInput(); }
  else if (k === "c") {
    debugEl.classList.toggle("hidden");
    if (debugEl.classList.contains("hidden")) crosshair.classList.add("hidden"); else updateCrosshair();
    uiChanged();
  } else if (k === "f") {
    if (document.documentElement.requestFullscreen) document.documentElement.requestFullscreen().catch(() => {});
  }
});

window.addEventListener("resize", () => {
  placeHighlight();
  if (lastTarget) placeAvatar(currentState, lastTarget.x, lastTarget.y);
  updateCrosshair();
  uiChanged();
});

// Start parked at the centre of the screen.
placeAvatar("idle", PROJ_W / 2, PROJ_H / 2);
setState("idle");
