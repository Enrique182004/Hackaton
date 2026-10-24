import { $, every, request, setNote } from "./api.js";
import { bindText, onLanguageChange, plural, t } from "../i18n.js";
import { logEvent } from "./log.js";

// "Use a phone or tablet" under More tools. Phone mode (SAR_LAN) renders the card and data-lan="1"; GET /api/lan
// answers only the laptop itself, so a joined phone never sees the card, the PIN or the QR code.
const POLL_MS = 5000;
const SVG_NS = "http://www.w3.org/2000/svg";
const QUIET_ZONE = 4; // modules of white around the code, as scanners expect
const ACTIONS = {
  rotate: { confirm: "phone.confirmRotate", done: "phone.rotated", failed: "phone.rotateFailed" },
  disconnect: { confirm: "phone.confirmDisconnect", done: "phone.disconnected", failed: "phone.disconnectFailed" },
};

let last = null;
// The URL the code on screen encodes; undefined until the first render, so a first reply with no URL hides the box.
let drawnUrl;
// Bumped on every render, so a GET that was in flight during New PIN can't bring the old PIN back.
let version = 0;
// GET /api/lan said no (a phone, or a server without phone mode): stop asking.
let refused = false;

export function lanEnabled() {
  return document.body.dataset.lan === "1";
}

// "123456" -> "123 456": easier to read out and to type.
export function groupPin(pin) {
  const digits = String(pin ?? "");
  return /^\d{6}$/.test(digits) ? `${digits.slice(0, 3)} ${digits.slice(3)}` : digits;
}

// The address to type: the QR link without its ?pin=, so typing it opens the PIN form.
export function typedUrl(url) {
  return String(url).replace(/[?#].*$/, "");
}

// One SVG path for every dark module; black on white whatever the page theme, or phones may not read it.
export function qrSvg(url) {
  const make = globalThis.qrcode;
  if (typeof make !== "function") return null;
  const qr = make(0, "M"); // 0 = the smallest version that fits
  qr.addData(url);
  qr.make();
  const n = qr.getModuleCount();
  const size = n + 2 * QUIET_ZONE;
  let d = "";
  for (let row = 0; row < n; row++) {
    for (let col = 0; col < n; col++) if (qr.isDark(row, col)) d += `M${col + QUIET_ZONE} ${row + QUIET_ZONE}h1v1h-1z`;
  }
  const svg = document.createElementNS(SVG_NS, "svg");
  for (const [k, v] of Object.entries({ viewBox: `0 0 ${size} ${size}`, "shape-rendering": "crispEdges" })) svg.setAttribute(k, v);
  const paper = document.createElementNS(SVG_NS, "rect");
  for (const [k, v] of Object.entries({ width: size, height: size, fill: "#ffffff" })) paper.setAttribute(k, v);
  const ink = document.createElementNS(SVG_NS, "path");
  for (const [k, v] of Object.entries({ d, fill: "#000000" })) ink.setAttribute(k, v);
  svg.append(paper, ink);
  return svg;
}

function drawQr(url) {
  const box = $("phone-qr");
  if (url === drawnUrl) return;
  drawnUrl = url;
  const svg = url ? qrSvg(url) : null;
  // No code (no network, or the vendored script didn't load): the address and the PIN still work.
  box.hidden = !svg;
  box.replaceChildren(...(svg ? [svg] : []));
}

function render(body) {
  version += 1;
  last = body;
  $("phone-card").hidden = false;
  const urls = (Array.isArray(body.urls) ? body.urls : []).filter((u) => typeof u === "string" && u);
  drawQr(urls[0] || null);
  bindText($("phone-pin"), body.pin ? groupPin(body.pin) : "–");
  bindText($("phone-url"), urls.length ? typedUrl(urls[0]) : () => t("phone.noNetwork"));
  $("phone-url").dataset.tone = urls.length ? "" : "warn";
  const others = urls.slice(1).map(typedUrl);
  bindText($("phone-other"), others.length ? () => t("phone.otherUrls", { urls: others.join(" · ") }) : "");
  $("phone-other").hidden = others.length === 0;
  const phones = Number.isFinite(body.phones) ? body.phones : 0;
  bindText($("phone-count"), () => (phones > 0 ? plural("phone.count", phones) : t("phone.count.none")));
  const wait = Number.isFinite(body.locked_for) ? body.locked_for : 0;
  if (wait > 0) setNote($("phone-msg"), () => plural("phone.locked", Math.ceil(wait / 60)), "warn");
  else if ($("phone-msg").dataset.tone === "warn") setNote($("phone-msg"), "");
}

function hide() {
  version += 1;
  last = null;
  $("phone-card").hidden = true;
}

export async function refreshPhone() {
  if (!lanEnabled() || !$("phone-card") || refused) return;
  const seen = version;
  let body;
  try {
    body = await request("/api/lan");
  } catch (err) {
    // 403: this is a phone. 404: a server without phone mode. Anything else may pass, so keep what is shown.
    if (err.status === 403 || err.status === 404) {
      refused = true;
      hide();
    }
    throw err;
  }
  if (seen !== version) return;
  if (body && body.enabled === true) render(body);
  else hide();
}

async function act(kind) {
  const texts = ACTIONS[kind];
  if (!confirm(t(texts.confirm))) return;
  const button = $(`phone-${kind}-btn`);
  button.disabled = true;
  try {
    const body = await request(`/api/lan/${kind}`, { method: "POST" });
    if (body && body.enabled === true && Array.isArray(body.urls)) render(body);
    else await refreshPhone();
    setNote($("phone-msg"), () => t(texts.done), "ok");
  } catch (err) {
    logEvent("lan", `${kind}: ${err.message}`);
    setNote($("phone-msg"), () => t(texts.failed, { hint: t("hint.offline") }), "bad");
  } finally {
    button.disabled = false;
  }
}

export function initPhone() {
  if (!lanEnabled() || !$("phone-card")) return;
  $("phone-rotate-btn").addEventListener("click", () => act("rotate"));
  $("phone-disconnect-btn").addEventListener("click", () => act("disconnect"));
  onLanguageChange(() => {
    if (last) render(last);
  });
  // The first tick always asks; after that only while More tools is open (the phone count, a lockout).
  let first = true;
  every(POLL_MS, async () => {
    if (!first && $("panel-more").hidden) return;
    first = false;
    await refreshPhone();
  });
}

// Phone mode registers the service worker; otherwise a worker left from an earlier phone-mode run on this port is
// removed with its caches, so edited JS and CSS show up on the next reload. Plain-HTTP phones have no serviceWorker.
export function initServiceWorker() {
  const workers = globalThis.navigator?.serviceWorker;
  if (!workers) return;
  if (lanEnabled()) {
    workers.register("/sw.js", { scope: "/" }).catch(() => {});
    return;
  }
  workers
    .getRegistrations()
    .then((registrations) => registrations.forEach((r) => r.unregister()))
    .catch(() => {});
  const caches = globalThis.caches;
  if (caches) {
    caches
      .keys()
      .then((names) => names.filter((n) => n.startsWith("sar-")).forEach((n) => caches.delete(n)))
      .catch(() => {});
  }
}
