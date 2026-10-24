import { $, el } from "./api.js";

const MAX_LINES = 200;

export function logEvent(name, data) {
  const list = $("log");
  const time = new Date().toLocaleTimeString([], { hour12: false });
  const detail = typeof data === "string" ? data : JSON.stringify(data);
  list.prepend(el("li", {}, el("time", { textContent: time }), el("span", { className: "ev", textContent: name }), detail || ""));
  while (list.children.length > MAX_LINES) list.lastChild.remove();
}

const LABELS = {
  vlm_model: "AI model",
  vlm_backend: "AI model runs on",
  vlm_base_url: "AI model address",
  embed_model: "File search model",
  rag_top_k: "File passages per answer",
  whisper_model: "Speech-to-text model",
  listen_mode: "Microphone mode",
  wake_word: "Wake word",
  tts_backend: "Voice",
  tts_voice: "Voice name",
  camera_index: "Camera number",
  camera_size: "Camera picture size",
  projector_size: "Projector size",
  checkerboard: "Checkerboard squares",
};

export function renderConfig(config) {
  const dl = $("config");
  dl.replaceChildren();
  for (const [key, value] of Object.entries(config)) {
    const label = LABELS[key] || key.replaceAll("_", " ");
    dl.append(el("dt", { textContent: label }), el("dd", { textContent: Array.isArray(value) ? value.join(" × ") : String(value) }));
  }
}
