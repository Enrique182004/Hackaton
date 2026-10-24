import { $, request, setNote } from "./api.js";
import { languageFor, setLanguage, t } from "../i18n.js";
import { logEvent } from "./log.js";

// The header switch "English | Español | Auto". Auto shows the page in the language of the latest answer.
const SETTINGS = ["en", "es", "auto"];
const STORAGE_KEY = "sar.dashboard.language";

// Bumped on every render, so a GET or status poll that was in flight during a newer change can't roll it back.
let version = 0;

function remembered() {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

function remember(lang) {
  try {
    localStorage.setItem(STORAGE_KEY, lang);
  } catch {
    // Private mode or blocked storage: the next visit starts in English until the server answers.
  }
}

export function languageVersion() {
  return version;
}

// A /api/language body ({setting, current}) from GET, PUT, the "language" event or /api/status.
export function renderLanguage(body) {
  if (!body || !SETTINGS.includes(body.setting)) return;
  version += 1;
  $("lang-switch").hidden = false;
  for (const s of SETTINGS) $(`lang-${s}`).setAttribute("aria-pressed", String(s === body.setting));
  const lang = languageFor(body);
  if (!lang) return;
  setLanguage(lang);
  remember(lang);
}

export async function refreshLanguage() {
  const seen = version;
  const body = await request("/api/language");
  if (seen === version) renderLanguage(body);
}

async function choose(setting) {
  const buttons = SETTINGS.map((s) => $(`lang-${s}`));
  for (const b of buttons) b.disabled = true;
  try {
    renderLanguage(await request("/api/language", { method: "PUT", json: { language: setting } }));
    setNote($("lang-msg"), "");
  } catch (err) {
    logEvent("language", err.message);
    setNote($("lang-msg"), () => t("lang.failed"), "bad");
  } finally {
    for (const b of buttons) b.disabled = false;
  }
}

// First paint in the language this browser showed last, so a Spanish class never flashes English.
export function initLanguage() {
  setLanguage(remembered());
  for (const s of SETTINGS) $(`lang-${s}`).addEventListener("click", () => choose(s));
}
