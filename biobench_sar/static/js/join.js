import { LANGUAGES, bindText, plural, setLanguage, t } from "./i18n.js";

// The PIN page a phone sees before it joins. It can't ask the server anything yet, so it speaks the browser's
// language, else the one the server rendered it in, else English.
const TICK_MS = 1000;

export function browserLanguage(preferred, fallback) {
  for (const tag of preferred || []) {
    const base = String(tag || "").toLowerCase().split("-")[0];
    if (LANGUAGES.includes(base)) return base;
  }
  return LANGUAGES.includes(fallback) ? fallback : "en";
}

// Seconds left -> the countdown sentence: minutes (rounded up) from one minute on, else seconds.
export function lockoutText(seconds) {
  return seconds >= 60 ? plural("join.lockedMinutes", Math.ceil(seconds / 60)) : plural("join.lockedSeconds", seconds);
}

const $ = (id) => document.getElementById(id);
const nav = globalThis.navigator || {};
setLanguage(browserLanguage(nav.languages?.length ? nav.languages : [nav.language], document.body.dataset.language));

const input = $("pin");
const button = $("join-btn");
const locked = $("join-locked");

// Guesses during a lockout don't count but aren't tried either, so the button waits for the countdown.
if (locked) {
  let left = Math.ceil(parseFloat(locked.dataset.lockedFor) || 0);
  const tick = () => {
    if (left <= 0) {
      clearInterval(timer);
      locked.hidden = true;
      button.disabled = false;
      bindText($("join-msg"), () => t("join.ready"));
      $("join-msg").dataset.tone = "ok";
      input.focus();
      return;
    }
    const now = left;
    bindText(locked, () => lockoutText(now));
    left -= 1;
  };
  button.disabled = true;
  const timer = setInterval(tick, TICK_MS);
  tick();
}

// A wrong PIN comes back with the form: start the next try from an empty box.
if ($("join-error") && input) {
  input.value = "";
  input.focus();
}
