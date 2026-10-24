import { $, request, every, setNote } from "./api.js";
import { initAsk, addInteraction, showPending, loadHistory, onListenState } from "./ask.js";
import { initCamera, setHighlight, setTarget } from "./camera.js";
import { initFiles, refreshFiles, renderFiles } from "./files.js";
import { t } from "../i18n.js";
import { initLanguage, languageVersion, refreshLanguage, renderLanguage } from "./language.js";
import { renderStatus, renderState, renderConnection } from "./status.js";
import { initTabs } from "./tabs.js";
import { logEvent, renderConfig } from "./log.js";
import { setProfile } from "./profile.js";
import { initTutor, refreshTutor, renderTutor } from "./tutor.js";
import { initPhone, initServiceWorker, refreshPhone } from "./phone.js";

const STATUS_MS = 2000;
// The first handshake takes a moment; "Connecting…" stays up until then instead of flashing "not connected".
const CONNECT_GRACE_MS = 3000;

let socketLive = false;
let configLoaded = false;
let configRequest = null;

// The subject decides the wording and whether the math mode picker shows, so it is fetched again on every
// reconnect (the server may have restarted with another SAR_PROFILE) and retried until it first succeeds.
function loadConfig() {
  configRequest ??= request("/api/config")
    .then((config) => {
      configLoaded = true;
      renderConfig(config);
      setProfile(config.profile);
    })
    .finally(() => {
      configRequest = null;
    });
  return configRequest;
}

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
    loadConfig().catch(() => {});
    loadHistory().catch(() => {});
    refreshFiles().catch(() => {});
    refreshTutor().catch(() => {});
    refreshLanguage().catch(() => {});
    // A restarted server has a new PIN.
    refreshPhone().catch(() => {});
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
    if (Object.hasOwn(d, "highlight")) setHighlight(d.highlight);
    logEvent("avatar_update", d);
  });
  socket.on("interaction", (d) => addInteraction(d));
  socket.on("speak", (d) => logEvent("speak", d.url || "(local voice)"));
  socket.on("documents_changed", (d) => renderFiles(d));
  socket.on("tutor", (d) => {
    renderTutor(d);
    logEvent("tutor", d);
  });
  socket.on("language", (d) => {
    renderLanguage(d);
    logEvent("language", d);
  });
}

// The calib.* key for a failed lining-up; the raw reason goes to the activity log.
function calibrationError(res) {
  const reason = res.error || "";
  if (res.status === 503 || /no camera frame/i.test(reason)) return "calib.noCamera";
  if (res.status === 409) return "calib.busy";
  if (/not detected/i.test(reason)) return "calib.notDetected";
  logEvent("calibrate", reason || "unknown problem");
  return "calib.failed";
}

function initCalibration() {
  const btn = $("calibrate-btn");
  const msg = $("calib-msg");
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    const say = (key, tone) => setNote(msg, () => t(key), tone);
    say("calib.showing");
    try {
      const res = await request("/api/calibrate", { method: "POST", timeout: 15000 });
      if (res.ok && res.error) {
        logEvent("calibrate", res.error);
        say("calib.notSaved", "warn");
      } else if (res.ok) {
        say("calib.ok", "ok");
      } else {
        say(calibrationError(res), "bad");
      }
    } catch (err) {
      say(calibrationError({ status: err.status, error: err.body?.error || err.message }), "bad");
    } finally {
      btn.disabled = false;
    }
  });
}

initLanguage();
initTabs();
initSocket();
loadHistory().catch(() => {});
initCamera();
initAsk();
initFiles();
initTutor();
initCalibration();
initPhone();
initServiceWorker();
every(STATUS_MS, async () => {
  const seenLanguage = languageVersion();
  const status = await request("/api/status");
  // The poll also catches a language change whose socket event was missed; an older reply never undoes a newer one.
  if (seenLanguage === languageVersion()) renderLanguage({ setting: status.language, current: status.current_language });
  // /api/status names the subject too, so a failed /api/config can't leave the wrong one showing.
  if (status.profile) setProfile(status.profile);
  if (!configLoaded) loadConfig().catch(() => {});
  renderStatus(status);
  // Without a live socket the poll is the only source of state changes (e.g. the push-to-talk note,
  // or the student switching mode by voice).
  if (!socketLive && status.state) applyState(status.state);
  if (!socketLive) refreshTutor().catch(() => {});
});
loadConfig().catch(() => {});
refreshLanguage().catch(() => {});
