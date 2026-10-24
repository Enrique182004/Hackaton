import { $, request, setNote } from "./api.js";
import { t } from "../i18n.js";
import { logEvent } from "./log.js";

// "Accessibility" under More tools. The settings belong to the room, not to this browser: the projector, the voice
// and every open dashboard follow them (the "accessibility" event), so a joined phone may change them too.
const SIZES = ["normal", "large", "xlarge"];
const RATES = ["slower", "normal", "faster"];
// The on/off settings and their checkboxes.
const BOXES = { captions_stay: "access-captions", calm: "access-calm", high_contrast: "access-contrast" };

// Bumped on every render, so a GET that was in flight during a newer change or push can't roll the card back.
let version = 0;
// What the room has, as the server last said; a failed change shows this again.
let last = null;

function controls() {
  return [
    ...Object.values(BOXES).map((id) => $(id)),
    ...SIZES.map((s) => $(`access-size-${s}`)),
    ...RATES.map((r) => $(`access-rate-${r}`)),
  ];
}

// GET, PUT and the event all send the five settings; anything less (another server version) is left alone.
function complete(body) {
  return (
    Boolean(body) &&
    Object.keys(BOXES).every((key) => typeof body[key] === "boolean") &&
    SIZES.includes(body.text_size) &&
    RATES.includes(body.speech_rate)
  );
}

function draw(body) {
  $("access-card").hidden = false;
  for (const [key, id] of Object.entries(BOXES)) $(id).checked = body[key];
  for (const s of SIZES) $(`access-size-${s}`).setAttribute("aria-pressed", String(s === body.text_size));
  for (const r of RATES) $(`access-rate-${r}`).setAttribute("aria-pressed", String(r === body.speech_rate));
}

// A /api/accessibility body from GET, PUT or the "accessibility" event.
export function renderAccessibility(body) {
  if (!complete(body)) return;
  version += 1;
  last = body;
  draw(body);
}

export async function refreshAccessibility() {
  const seen = version;
  const body = await request("/api/accessibility");
  if (seen === version) renderAccessibility(body);
}

// Only the setting that changed is sent, so two people changing different settings never undo each other.
async function change(key, value) {
  if (last && last[key] === value) return;
  const all = controls();
  // Disabling the focused control drops keyboard focus to the page; it goes back once the change is done.
  const focused = all.find((c) => c === document.activeElement);
  for (const c of all) c.disabled = true;
  try {
    renderAccessibility(await request("/api/accessibility", { method: "PUT", json: { [key]: value } }));
    setNote($("access-msg"), "");
  } catch (err) {
    logEvent("accessibility", err.message);
    setNote($("access-msg"), () => t("access.failed", { hint: t("hint.offline") }), "warn");
    // A checkbox has already ticked itself: show what the room really has.
    if (last) draw(last);
  } finally {
    for (const c of all) c.disabled = false;
    focused?.focus();
  }
}

export function initAccessibility() {
  for (const [key, id] of Object.entries(BOXES)) $(id).addEventListener("change", () => change(key, $(id).checked));
  for (const s of SIZES) $(`access-size-${s}`).addEventListener("click", () => change("text_size", s));
  for (const r of RATES) $(`access-rate-${r}`).addEventListener("click", () => change("speech_rate", r));
  // An older server has no accessibility API: the card simply stays hidden.
  refreshAccessibility().catch(() => {});
}
