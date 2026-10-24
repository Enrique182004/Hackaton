import { $, el } from "./api.js";

const MAX_LINES = 200;

export function logEvent(name, data) {
  const list = $("log");
  const time = new Date().toLocaleTimeString([], { hour12: false });
  const detail = typeof data === "string" ? data : JSON.stringify(data);
  list.prepend(el("li", {}, el("time", { textContent: time }), el("span", { className: "ev", textContent: name }), detail || ""));
  while (list.children.length > MAX_LINES) list.lastChild.remove();
}

export function renderConfig(config) {
  const dl = $("config");
  dl.replaceChildren();
  for (const [key, value] of Object.entries(config)) {
    dl.append(el("dt", { textContent: key }), el("dd", { textContent: Array.isArray(value) ? value.join(" × ") : String(value) }));
  }
}
