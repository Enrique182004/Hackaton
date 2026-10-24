(function () {
  "use strict";

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
  // The corner is read from across the room, so it says what to do rather than the internal state name.
  const STATUS_LABELS = {
    idle: "Ready – press Space to talk",
    listening: "Listening…",
    transcribing: "Writing it down…",
    thinking: "Thinking…",
    speaking: "Answering…",
  };
  const BUSY_LABEL = "Still busy – wait a moment";
  const OFFLINE_LABEL = "Not connected";

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
  const quizCard = $("quiz-card");
  const quizCount = $("quiz-card-count");
  const quizProblem = $("quiz-card-problem");

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

  // Uses the avatar's destination (style.top), not its mid-transition position, plus the real bubble height.
  function fitBubble() {
    const top = parseFloat(avatar.style.top) || 0;
    const h = bubble.classList.contains("hidden") ? 120 : bubble.offsetHeight;
    avatar.classList.toggle("bubble-below", top - h - 8 < 0);
    // Centre the bubble over the avatar, then clamp it inside the window: the avatar itself may hang off-screen
    // when it points near an edge, so aligning to the avatar's edge is not enough.
    const left = parseFloat(avatar.style.left) || 0;
    const width = bubble.classList.contains("hidden") ? 420 : bubble.offsetWidth;
    const ideal = left + avatar.offsetWidth / 2 - width / 2;
    const clamped = Math.max(EDGE, Math.min(ideal, window.innerWidth - width - EDGE));
    bubble.style.left = Math.round(clamped - left) + "px";
  }

  // Single owner of bubbleTimer: every (re)arm clears the previous one so stale timers can't hide a newer bubble.
  function armBubbleTimer(ms) {
    clearTimeout(bubbleTimer);
    bubbleTimer = ms ? setTimeout(hideBubbleUnlessSpeaking, ms) : null;
  }

  function showBubble(text, holdMs) {
    armBubbleTimer(0);
    if (!text) { bubble.classList.add("hidden"); return; }
    bubble.textContent = text;
    bubble.classList.remove("hidden");
    fitBubble();
    armBubbleTimer(holdMs);
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

  function hideBubbleUnlessSpeaking() {
    if (audio && !audio.paused && !audio.ended) return;
    bubble.classList.add("hidden");
  }

  function onAvatarUpdate(d) {
    const finalState = d.animation_state || "idle";
    const hasTarget = Number.isFinite(d.target_x) && Number.isFinite(d.target_y);
    pendingState = finalState;
    clearTimeout(moveTimer);

    if (hasTarget) {
      lastTarget = { x: d.target_x, y: d.target_y };
      setState("walking");
      placeAvatar(finalState, d.target_x, d.target_y);
      // transitionend can be skipped (no movement / hidden tab), so use a timer as the single source of truth.
      moveTimer = setTimeout(() => setState(pendingState), MOVE_MS + 50);
      updateCrosshair();
    } else {
      setState(finalState);
    }
    showBubble(d.spoken_text || "", d.spoken_text ? Math.max(4000, d.spoken_text.length * 80) : 0);
  }

  function onSpeak(d) {
    releaseAudio();
    if (d.text) showBubble(d.text); else armBubbleTimer(0);
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
    statusLabel.textContent = d.detail === "busy" ? BUSY_LABEL : STATUS_LABELS[s] || s;
  }

  function onTutor(d) {
    tutorVersion += 1;
    const quiz = d.quiz;
    const active = Boolean(quiz && quiz.active);
    quizCard.classList.toggle("hidden", !active);
    if (!active) return;
    quizCount.textContent = "Quiz · problem " + quiz.number + " of " + quiz.total;
    // No problem text means the assistant makes problems up and reads them out itself.
    quizProblem.textContent = quiz.problem || "";
  }

  // "tutor" events missed while disconnected would leave a stale card, so ask the server once per connect.
  function refreshTutor() {
    const seen = tutorVersion;
    quizCard.classList.add("hidden");
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

  function onTranscript(d) {
    clearTimeout(transcriptTimer);
    transcriptEl.textContent = d.text || "";
    transcriptEl.classList.toggle("hidden", !d.text);
    transcriptTimer = setTimeout(() => transcriptEl.classList.add("hidden"), 5000);
  }

  // Black is invisible on the projector, so "offline" is a bright amber frame plus a status-corner label.
  function setOffline(text) {
    clearTimeout(offlineTimer);
    clearTimeout(offlineHintTimer);
    body.classList.add("offline");
    statusEl.dataset.state = "offline";
    statusLabel.textContent = text || OFFLINE_LABEL;
  }

  function hintOffline() {
    setOffline("Not connected – question not sent");
    offlineHintTimer = setTimeout(() => {
      if (!connected) statusLabel.textContent = OFFLINE_LABEL;
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
    if (!statusSinceConnect) renderStatus({ state: "idle" });
    refreshTutor();
  }

  function showCalibration(d) {
    hideCalibration();
    const img = document.createElement("img");
    img.id = "calibration-img";
    img.alt = "";
    img.src = d.url || "/calibration.png";
    document.body.appendChild(img);
  }

  function hideCalibration() {
    const img = $("calibration-img");
    if (img) img.remove();
  }

  function updateCrosshair() {
    if (!lastTarget) return;
    const p = toPixels(lastTarget.x, lastTarget.y);
    crosshair.style.left = p.x + "px";
    crosshair.style.top = p.y + "px";
    crosshair.classList.toggle("hidden", debugEl.classList.contains("hidden"));
  }

  function logEvent(name, data) {
    debugEvent.textContent = name + " " + JSON.stringify(data, null, 1);
  }

  function setConn(text) {
    debugConn.textContent = "socket: " + text;
  }

  // Socket.IO may fail to load (offline venue); the page still renders and keys are no-ops.
  const socket = typeof io === "function" ? io() : null;
  if (socket) {
    socket.on("connect", () => { connected = true; setConn("connected"); resync(); });
    socket.on("disconnect", () => { connected = false; statusSinceConnect = false; setConn("disconnected"); setOffline(); });
    socket.on("connect_error", (e) => {
      connected = false;
      statusSinceConnect = false;
      setConn("error: " + (e && e.message));
      setOffline();
    });
    // Server-emitted "interaction" events are for the dashboard; the projector deliberately has no handler.
    const handlers = {
      avatar_update: onAvatarUpdate, speak: onSpeak, status: onStatus, transcript: onTranscript,
      show_calibration: showCalibration, hide_calibration: hideCalibration, tutor: onTutor,
    };
    Object.keys(handlers).forEach((name) => {
      socket.on(name, (data) => {
        logEvent(name, data);
        try { handlers[name](data || {}); } catch (err) { console.error(name, err); }
      });
    });
  } else {
    setConn("socket.io client not loaded");
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

  function openTextInput() {
    textOverlay.classList.remove("hidden");
    textInput.value = "";
    textInput.focus();
  }

  function closeTextInput() {
    textOverlay.classList.add("hidden");
    textInput.blur();
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
    } else if (k === "f") {
      if (document.documentElement.requestFullscreen) document.documentElement.requestFullscreen().catch(() => {});
    }
  });

  window.addEventListener("resize", () => {
    if (lastTarget) placeAvatar(currentState, lastTarget.x, lastTarget.y);
    updateCrosshair();
  });

  // Start parked at the centre of the screen.
  placeAvatar("idle", PROJ_W / 2, PROJ_H / 2);
  setState("idle");
})();
