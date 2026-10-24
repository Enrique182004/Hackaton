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
  profile: "Subject",
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
  checkerboard: "Checkerboard inner corners",
};

// Plain words for the values a student might wonder about; anything else is shown as configured.
const VALUES = {
  profile: { math: "Math", biology: "Biology" },
  listen_mode: {
    push: () => "Press Talk (or Space on the projector page)",
    wake: (config) => `Always listening: say “${config.wake_word || "assistant"}” first`,
  },
  tts_backend: {
    edge: "Online voice (needs internet)",
    pyttsx3: "Computer voice (works offline)",
    say: "Mac voice (works offline)",
    none: "Off (answers are only written)",
  },
};

function shown(key, value, config) {
  const table = VALUES[key];
  const plain = table && Object.hasOwn(table, value) ? table[value] : null;
  if (typeof plain === "function") return plain(config);
  if (plain) return plain;
  return Array.isArray(value) ? value.join(" × ") : String(value);
}

export function renderConfig(config) {
  const dl = $("config");
  dl.replaceChildren();
  for (const [key, value] of Object.entries(config)) {
    // The wake word only matters when the microphone is always listening.
    if (key === "wake_word" && config.listen_mode !== "wake") continue;
    const label = LABELS[key] || key.replaceAll("_", " ");
    dl.append(el("dt", { textContent: label }), el("dd", { textContent: shown(key, value, config) }));
  }
}
