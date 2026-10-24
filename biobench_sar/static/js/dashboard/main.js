import { $, request, every, setNote } from "./api.js";
import { initAsk, addInteraction, showPending, loadHistory, onListenState } from "./ask.js";
import { initCamera, setTarget } from "./camera.js";
import { initFiles, refreshFiles, renderFiles } from "./files.js";
import { renderStatus, renderState, renderConnection } from "./status.js";
import { initTabs } from "./tabs.js";
import { logEvent, renderConfig } from "./log.js";

const STATUS_MS = 2000;

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
  socket.on("connect", () => {
    socketLive = true;
    renderConnection(true);
    logEvent("connect", "");
    loadHistory().catch(() => {});
    refreshFiles().catch(() => {});
  });
  socket.on("disconnect", () => {
    socketLive = false;
    renderConnection(false);
    logEvent("disconnect", "");
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
}

function calibrationError(res) {
  const reason = res.error || "";
  if (res.status === 503 || /no camera frame/i.test(reason)) return "The camera isn't working yet, so it can't line up. Fix the camera first (see the list).";
  if (res.status === 409) return "Already lining up, one moment…";
  if (/not detected/i.test(reason)) return "Couldn't see the checkerboard. Make sure the camera sees the whole board and the room isn't too bright, then try again.";
  return `Couldn't line up (${reason || "unknown problem"}).`;
}

function initCalibration() {
  const btn = $("calibrate-btn");
  const msg = $("calib-msg");
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    setNote(msg, "Showing the checkerboard on the board… keep the camera still.");
    try {
      const res = await request("/api/calibrate", { method: "POST", timeout: 15000 });
      if (res.ok) setNote(msg, res.error ? `Lined up, but it couldn't be saved for next time (${res.error}).` : "Lined up! The assistant now knows where the board is.", res.error ? "warn" : "ok");
      else setNote(msg, calibrationError(res), "bad");
    } catch (err) {
      setNote(msg, calibrationError({ status: err.status, error: err.body?.error || err.message }), "bad");
    } finally {
      btn.disabled = false;
    }
  });
}

initTabs();
renderConnection(false);
initSocket();
loadHistory().catch(() => {});
initCamera();
initAsk();
initFiles();
initCalibration();
every(STATUS_MS, async () => {
  const status = await request("/api/status");
  renderStatus(status);
  // Without a live socket the poll is the only source of state changes (e.g. the push-to-talk note).
  if (!socketLive && status.state) applyState(status.state);
});
request("/api/config").then(renderConfig).catch(() => {});
