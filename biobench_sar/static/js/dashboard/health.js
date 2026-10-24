import { $, el, formatDuration } from "./api.js";

function row(label, value, tone) {
  return el("li", { dataset: { tone } }, el("span", { className: "dot" }), el("span", { className: "label", textContent: label }), el("span", { className: "value", textContent: value }));
}

export function renderStatus(s) {
  const fps = s.fps ?? 0;
  const cameraTone = !s.camera_running ? "bad" : fps >= 28 ? "ok" : "warn";
  const model = `${s.vlm_backend} · ${s.vlm_model}`;
  $("health").replaceChildren(
    row("Camera", s.camera_running ? `${fps.toFixed(1)} fps` : "no signal", cameraTone),
    row("Calibration", s.calibrated ? "calibrated" : "not calibrated", s.calibrated ? "ok" : "warn"),
    row("Knowledge", `${s.doc_chunks} chunks`, s.doc_chunks > 0 ? "ok" : "warn"),
    row("Model", model, s.vlm_backend === "mock" ? "warn" : "ok"),
    row("Microphone", s.audio_available ? `${s.listen_mode} mode` : "unavailable", s.audio_available ? "ok" : "bad"),
    row("Voice", s.tts_backend, s.tts_backend === "none" ? "warn" : "ok"),
  );
  $("fps").textContent = s.camera_running ? fps.toFixed(1) : "–";
  $("latency").textContent = s.last_latency_s == null ? "–" : `${s.last_latency_s.toFixed(1)}s`;
  $("uptime").textContent = formatDuration(s.uptime_s);
  $("kb-total").textContent = s.doc_chunks;
  $("viewport").dataset.live = s.camera_running ? "1" : "0";
}

export function renderState(state, detail) {
  const chip = $("state-chip");
  chip.dataset.state = state;
  chip.querySelector("span").textContent = detail ? `${state} · ${detail}` : state;
  $("listen-btn").dataset.live = state === "listening" ? "1" : "0";
}

// `hint` marks a degraded mode (polling still works), so it reads as a warning rather than a failure.
export function renderConnection(connected, hint = "") {
  const chip = $("conn-chip");
  chip.dataset.tone = connected ? "ok" : hint ? "warn" : "bad";
  chip.title = hint;
  chip.querySelector("span").textContent = connected ? "live" : hint ? "offline · polling" : "offline";
}
