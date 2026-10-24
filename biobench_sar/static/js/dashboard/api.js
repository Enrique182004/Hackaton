const TIMEOUT_MS = 4000;

export class ApiError extends Error {
  constructor(message, status, body) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

export async function request(path, { method = "GET", json, body, timeout = TIMEOUT_MS } = {}) {
  const init = { method, signal: AbortSignal.timeout(timeout) };
  if (json !== undefined) {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(json);
  } else if (body !== undefined) {
    init.body = body;
  }
  const res = await fetch(path, init);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new ApiError(data.error || `HTTP ${res.status}`, res.status, data);
  return data;
}

export function $(id) {
  return document.getElementById(id);
}

export function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "dataset") Object.assign(node.dataset, value);
    else if (key in node) node[key] = value;
    else node.setAttribute(key, value);
  }
  node.append(...children.filter((c) => c !== null && c !== undefined));
  return node;
}

export function setNote(node, text, tone = "") {
  node.textContent = text;
  node.dataset.tone = tone;
}

// Self-scheduling loop: the next tick waits for the previous one, so slow responses never pile up.
export function every(ms, fn, { backoffMs = ms * 4 } = {}) {
  let stopped = false;
  const tick = async () => {
    if (stopped) return;
    let delay = ms;
    if (!document.hidden) {
      try {
        await fn();
      } catch {
        delay = backoffMs;
      }
    }
    setTimeout(tick, delay);
  };
  tick();
  return () => { stopped = true; };
}

export function formatDuration(seconds) {
  if (seconds == null) return "–";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  if (m < 60) return `${m}m ${Math.round(seconds % 60)}s`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}
