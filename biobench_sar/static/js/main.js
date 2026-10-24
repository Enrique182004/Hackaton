(function () {
  "use strict";

  const body = document.body;
  const PROJ_W = parseFloat(body.dataset.projectorWidth) || 1920;
  const PROJ_H = parseFloat(body.dataset.projectorHeight) || 1080;
  const FRAME = 160;
  // Pointing sprite: fingertip sits at (153,56) in the 160px frame; other states anchor at the centre.
  const POINT_TIP = { x: 153, y: 56 };
  const CENTER = { x: FRAME / 2, y: FRAME / 2 };
  const STATES = ["idle", "walking", "pointing", "celebrating", "error"];
  const MOVE_MS = 1500;

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

  let currentState = "idle";
  let pendingState = "idle";
  let moveTimer = null;
  let lastTarget = null;
  let audio = null;
  let bubbleTimer = null;
  let transcriptTimer = null;
  let connected = false;

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
    const mid = (parseFloat(avatar.style.left) || 0) + avatar.offsetWidth / 2;
    const half = (bubble.classList.contains("hidden") ? 420 : bubble.offsetWidth) / 2;
    avatar.classList.toggle("bubble-right", mid - half < 0);
    avatar.classList.toggle("bubble-left", mid + half > window.innerWidth);
  }

  function showBubble(text, holdMs) {
    clearTimeout(bubbleTimer);
    if (!text) { bubble.classList.add("hidden"); return; }
    bubble.textContent = text;
    bubble.classList.remove("hidden");
    fitBubble();
    if (holdMs) bubbleTimer = setTimeout(hideBubbleUnlessSpeaking, holdMs);
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
    if (d.text) showBubble(d.text);
    if (audio) { audio.pause(); audio = null; }
    if (!d.url) { bubbleTimer = setTimeout(hideBubbleUnlessSpeaking, 4000); return; }
    audio = new Audio(d.url);
    const done = () => { bubbleTimer = setTimeout(hideBubbleUnlessSpeaking, 1200); };
    audio.addEventListener("ended", done);
    audio.addEventListener("error", done);
    audio.play().catch(done); // autoplay may be blocked until the first key press
  }

  function onStatus(d) {
    const s = d.state || "idle";
    statusEl.dataset.state = s;
    statusLabel.textContent = d.detail ? s + " - " + d.detail : s;
  }

  function onTranscript(d) {
    transcriptEl.textContent = d.text || "";
    transcriptEl.classList.toggle("hidden", !d.text);
    clearTimeout(transcriptTimer);
    transcriptTimer = setTimeout(() => transcriptEl.classList.add("hidden"), 5000);
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
    socket.on("connect", () => { connected = true; setConn("connected"); });
    socket.on("disconnect", () => { connected = false; setConn("disconnected"); });
    socket.on("connect_error", (e) => { connected = false; setConn("error: " + (e && e.message)); });
    const handlers = {
      avatar_update: onAvatarUpdate, speak: onSpeak, status: onStatus, transcript: onTranscript,
      show_calibration: showCalibration, hide_calibration: hideCalibration,
    };
    Object.keys(handlers).forEach((name) => {
      socket.on(name, (data) => {
        logEvent(name, data);
        try { handlers[name](data || {}); } catch (err) { console.error(name, err); }
      });
    });
  } else {
    setConn("socket.io client not loaded");
  }

  function emit(name, payload) {
    if (socket) socket.emit(name, payload || {});
  }

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
      const text = textInput.value.trim();
      if (text) emit("text_query", { text });
      closeTextInput();
    } else if (e.key === "Escape") {
      closeTextInput();
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key.toLowerCase();
    if (e.code === "Space") { e.preventDefault(); if (!e.repeat) emit("trigger_listen"); }
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
