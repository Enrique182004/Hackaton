import { $, request, every } from "./api.js";

const SVG_NS = "http://www.w3.org/2000/svg";
const FRAME_MS = 500;
const INK_MS = 1500;

let warped = true;
let target = null;

function svgNode(tag, attrs) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}

// The picture lives on the Board tab and the drawing count on More tools; other tabs need neither.
const boardShown = () => !$("panel-board").hidden;
const inkShown = () => boardShown() || !$("panel-more").hidden;

// Chains on load/error instead of reassigning src on a timer, so a slow frame is never cancelled mid-flight.
function startFrames() {
  const img = $("frame");
  const next = (delay) => setTimeout(load, delay);
  let started = false;
  const load = () => {
    // Re-check soon (no fetch) so the picture appears quickly once the Board tab opens.
    if (!boardShown()) return next(FRAME_MS);
    if (started && document.hidden) return next(FRAME_MS * 4);
    started = true;
    img.onload = () => next(FRAME_MS);
    img.onerror = () => next(FRAME_MS * 4);
    img.src = `/api/frame.jpg?warped=${warped ? 1 : 0}&t=${Date.now()}`;
  };
  load();
}

function drawOverlay(regions, size) {
  const svg = $("ink-overlay");
  const show = warped && $("ink-toggle").checked;
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${size[0]} ${size[1]}`);
  if (!show) return;
  for (const r of regions) {
    svg.append(svgNode("rect", { x: r.x, y: r.y, width: r.w, height: r.h, rx: 6 }));
    svg.append(svgNode("circle", { cx: r.cx, cy: r.cy, r: 8 }));
  }
  if (target) svg.append(svgNode("circle", { class: "target", cx: target.x, cy: target.y, r: 26 }));
}

export function setTarget(x, y) {
  target = { x, y };
}

export function initCamera() {
  for (const button of document.querySelectorAll(".seg button")) {
    button.addEventListener("click", () => {
      warped = button.dataset.view === "1";
      for (const b of document.querySelectorAll(".seg button")) b.setAttribute("aria-pressed", String(b === button));
      $("ink-overlay").replaceChildren();
    });
  }
  startFrames();
  every(INK_MS, async () => {
    if (!inkShown()) return;
    if (!warped || !$("ink-toggle").checked) return drawOverlay([], [1920, 1080]);
    try {
      const data = await request("/api/ink");
      $("ink-count").textContent = data.regions.length;
      drawOverlay(data.regions, data.size);
    } catch (err) {
      $("ink-count").textContent = "–";
      throw err;
    }
  });
}
