import { $, el } from "./api.js";
import { has, onLanguageChange, t } from "../i18n.js";

const MAX_LINES = 200;

export function logEvent(name, data) {
  const list = $("log");
  const time = new Date().toLocaleTimeString([], { hour12: false });
  const detail = typeof data === "string" ? data : JSON.stringify(data);
  list.prepend(el("li", {}, el("time", { textContent: time }), el("span", { className: "ev", textContent: name }), detail || ""));
  while (list.children.length > MAX_LINES) list.lastChild.remove();
}

// Plain words for the values a student might wonder about (value.<key>.<value>); anything else is shown as configured.
const PLAIN_VALUES = ["profile", "language", "current_language", "listen_mode", "tts_backend"];

let lastConfig = null;

function shown(key, value, config) {
  const table = key === "current_language" ? "language" : key;
  const plain = `value.${table}.${value}`;
  if (PLAIN_VALUES.includes(key) && has(plain)) return t(plain, { word: config.wake_word || "assistant" });
  return Array.isArray(value) ? value.join(" × ") : String(value);
}

// What a teacher looks for first; anything else follows in the server's order.
const ORDER = [
  "profile", "language", "current_language", "listen_mode", "wake_word", "tts_backend", "tts_voice", "vlm_model",
  "vlm_backend", "vlm_base_url", "camera_index", "camera_size", "projector_size", "checkerboard",
];

function rank(key) {
  const i = ORDER.indexOf(key);
  return i < 0 ? ORDER.length : i;
}

export function renderConfig(config) {
  lastConfig = config;
  const dl = $("config");
  dl.replaceChildren();
  const entries = Object.entries(config).sort(([a], [b]) => rank(a) - rank(b));
  for (const [key, value] of entries) {
    // The wake word only matters when the microphone is always listening.
    if (key === "wake_word" && config.listen_mode !== "wake") continue;
    const label = has(`config.${key}`) ? t(`config.${key}`) : key.replaceAll("_", " ");
    dl.append(el("dt", { textContent: label }), el("dd", { textContent: shown(key, value, config) }));
  }
}

onLanguageChange(() => {
  if (lastConfig) renderConfig(lastConfig);
});
