import { $, request, every, setNote } from "./api.js";
import { initAsk, addInteraction, showPending, loadHistory, onListenState } from "./ask.js";
import { initCamera, setTarget } from "./camera.js";
import { renderStatus, renderState, renderConnection } from "./health.js";
import { initKnowledge } from "./knowledge.js";
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
    renderConnection(false, "socket.io did not load: no live events, panels refresh by polling.");
    logEvent("socket", "socket.io client not loaded; polling only");
    return;
  }
  const socket = io();
  socket.on("connect", () => {
    socketLive = true;
    renderConnection(true);
    logEvent("connect", "");
    loadHistory().catch(() => {});
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
}

function initCalibration() {
  const btn = $("calibrate-btn");
  const msg = $("calib-msg");
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    setNote(msg, "Projecting checkerboard…");
    try {
      const res = await request("/api/calibrate", { method: "POST", timeout: 15000 });
      if (res.ok) setNote(msg, res.error ? `Calibrated, but ${res.error}` : "Calibrated.", res.error ? "warn" : "ok");
      else setNote(msg, res.error || "Calibration failed.", "bad");
    } catch (err) {
      setNote(msg, err.message, "bad");
    } finally {
      btn.disabled = false;
    }
  });
}

renderConnection(false);
initSocket();
initCamera();
initAsk();
initKnowledge();
initCalibration();
every(STATUS_MS, async () => {
  const status = await request("/api/status");
  renderStatus(status);
  // Without a live socket the poll is the only source of state changes (e.g. the push-to-talk note).
  if (!socketLive && status.state) applyState(status.state);
});
request("/api/config").then(renderConfig).catch(() => {});
