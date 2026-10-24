import { $ } from "./api.js";
import { bindText } from "../i18n.js";

const STEPS = ["files", "board", "ask", "more"];
const STORAGE_KEY = "sar.dashboard.step";

let chosenByUser = false;

function remembered() {
  try {
    const step = localStorage.getItem(STORAGE_KEY);
    return STEPS.includes(step) ? step : null;
  } catch {
    return null;
  }
}

export function selectStep(step, { focus = false, byUser = false } = {}) {
  for (const name of STEPS) {
    const tab = $(`tab-${name}`);
    const panel = $(`panel-${name}`);
    const active = name === step;
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
    panel.hidden = !active;
  }
  if (focus) $(`tab-${step}`).focus();
  if (byUser) {
    chosenByUser = true;
    try {
      localStorage.setItem(STORAGE_KEY, step);
    } catch {
      // Private mode or blocked storage: the choice just isn't remembered.
    }
  }
}

// `text` may be a function of the current language (see bindText).
export function setBadge(step, text, tone = "") {
  bindText($(`badge-${step}`), text);
  $(`tab-${step}`).dataset.tone = tone;
}

// First visit with an empty library: start where the work is. Never overrides a tab the user picked.
export function suggestStep(step) {
  if (!chosenByUser && !remembered()) selectStep(step);
}

export function initTabs() {
  for (const tab of document.querySelectorAll(".step-tab")) {
    tab.addEventListener("click", () => selectStep(tab.dataset.step, { byUser: true }));
    tab.addEventListener("keydown", (e) => {
      const offset = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
      if (!offset) return;
      e.preventDefault();
      const next = STEPS[(STEPS.indexOf(tab.dataset.step) + offset + STEPS.length) % STEPS.length];
      selectStep(next, { focus: true, byUser: true });
    });
  }
  const start = remembered();
  if (start) chosenByUser = true;
  selectStep(start || "ask");
}
