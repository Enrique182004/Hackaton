import { $, request, every, setNote } from "./api.js";
import { initAsk, addInteraction, showPending, loadHistory, onListenState } from "./ask.js";
import { initCamera, setTarget } from "./camera.js";
import { initFiles, refreshFiles, renderFiles } from "./files.js";
import { renderStatus, renderState, renderConnection } from "./status.js";
import { initTabs } from "./tabs.js";
import { logEvent, renderConfig } from "./log.js";
import { setProfile } from "./profile.js";
import { initTutor, refreshTutor, renderTutor } from "./tutor.js";

const STATUS_MS = 2000;
// The first handshake takes a moment; "Connecting…" stays up until then instead of flashing "not connected".
const CONNECT_GRACE_MS = 3000;

let socketLive = false;

function applyState(state, detail) {
  renderState(state, detail);
  onListenState(state);
}

function initSocket() {
  // The vendored client can fail to load; every other panel only needs polling, so keep booting without it.
  if (typeof io !== "function") {
    renderConnection(false, "no live updates");
    logEvent("socket", "socket.io client not loaded; polling only");
    return;
  }
  const socket = io();
  // Reconnect attempts fail every few seconds while the server is down; log the first one only.
  let errorLogged = false;
  setTimeout(() => {
    if (!socketLive) renderConnection(false);
  }, CONNECT_GRACE_MS);
  socket.on("connect", () => {
    socketLive = true;
    errorLogged = false;
    renderConnection(true);
    logEvent("connect", "");
    loadHistory().catch(() => {});
    refreshFiles().catch(() => {});
    refreshTutor().catch(() => {});
  });
  socket.on("disconnect", () => {
    socketLive = false;
    renderConnection(false);
    logEvent("disconnect", "");
  });
  socket.on("connect_error", (err) => {
    socketLive = false;
    renderConnection(false);
    if (!errorLogged) logEvent("connect_error", err?.message || "");
    errorLogged = true;
  });
  socket.on("status", (d) => {
    applyState(d.state, d.detail);
    logEvent("status", d);
  });
  socket.on("transcript", (d) => {
    showPending(d.text);
    logEvent("transcript", d.text);
  });
  socket.on("avatar_update", (d) => {
    setTarget(d.target_x, d.target_y);
    logEvent("avatar_update", d);
  });
  socket.on("interaction", (d) => addInteraction(d));
  socket.on("speak", (d) => logEvent("speak", d.url || "(local voice)"));
  socket.on("documents_changed", (d) => renderFiles(d));
  socket.on("tutor", (d) => {
    renderTutor(d);
    logEvent("tutor", d);
  });
}

function calibrationError(res) {
  const reason = res.error || "";
  if (res.status === 503 || /no camera frame/i.test(reason)) return "The camera isn't working yet, so it can't line up. Fix the camera first (see the checklist).";
  if (res.status === 409) return "Already lining up, one moment…";
  if (/not detected/i.test(reason)) {
    return "Couldn't see the checkerboard. Check the projector page is open on the projector in full screen (press F there), the camera sees the whole board and the room isn't too bright, then try again.";
  }
  logEvent("calibrate", reason || "unknown problem");
  return "Couldn't line up. Check the projector page is open on the projector in full screen and the assistant's Terminal window is still open, then try again.";
}

function initCalibration() {
  const btn = $("calibrate-btn");
  const msg = $("calib-msg");
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    setNote(msg, "Showing the checkerboard on the board… keep the camera still.");
    try {
      const res = await request("/api/calibrate", { method: "POST", timeout: 15000 });
      if (res.ok && res.error) {
        logEvent("calibrate", res.error);
        setNote(msg, "Lined up, but it couldn't be saved, so do this again after restarting the assistant.", "warn");
      } else if (res.ok) {
        setNote(msg, "Lined up! The assistant now knows where the board is.", "ok");
      } else {
        setNote(msg, calibrationError(res), "bad");
      }
    } catch (err) {
      setNote(msg, calibrationError({ status: err.status, error: err.body?.error || err.message }), "bad");
    } finally {
      btn.disabled = false;
    }
  });
}

initTabs();
initSocket();
loadHistory().catch(() => {});
initCamera();
initAsk();
initFiles();
initTutor();
initCalibration();
every(STATUS_MS, async () => {
  const status = await request("/api/status");
  renderStatus(status);
  // Without a live socket the poll is the only source of state changes (e.g. the push-to-talk note,
  // or the student switching mode by voice).
  if (!socketLive && status.state) applyState(status.state);
  if (!socketLive) refreshTutor().catch(() => {});
});
request("/api/config")
  .then((config) => {
    renderConfig(config);
    setProfile(config.profile);
  })
  .catch(() => {});
