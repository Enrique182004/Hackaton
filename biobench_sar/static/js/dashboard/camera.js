import { $, request, every } from "./api.js";

const SVG_NS = "http://www.w3.org/2000/svg";
const FRAME_MS = 500;
const INK_MS = 1500;
// As long as the projector shows its outline (main.js HIGHLIGHT_MS).
const HIGHLIGHT_MS = 8000;

let warped = true;
let target = null;
let highlight = null; // { x, y, w, h, at }: the ink the last answer was about

// The projector size from the template (Settings): the lined-up picture, the ink boxes and the avatar's
// target all live in this space. The classroom kit's Epson WXGA is 1280×800 (16:10).
export function projectorSize() {
  const w = parseFloat(document.body.dataset.projectorWidth);
  const h = parseFloat(document.body.dataset.projectorHeight);
  return w > 0 && h > 0 ? [w, h] : [1280, 800];
}

// Where an object-fit: contain picture of imgW×imgH sits inside a boxW×boxH box, in % of the box. The ink
// overlay is put exactly there, so its boxes line up with the drawings even if the picture's shape differs.
export function containedBox(boxW, boxH, imgW, imgH) {
  const box = boxW / boxH;
  const img = imgW > 0 && imgH > 0 ? imgW / imgH : box;
  if (img > box) {
    const height = (box / img) * 100;
    return { left: 0, top: (100 - height) / 2, width: 100, height };
  }
  const width = (img / box) * 100;
  return { left: (100 - width) / 2, top: 0, width, height: 100 };
}

function placeOverlay(img) {
  const [w, h] = projectorSize();
  const fit = containedBox(w, h, img.naturalWidth, img.naturalHeight);
  const svg = $("ink-overlay");
  for (const side of ["left", "top", "width", "height"]) svg.style[side] = `${+fit[side].toFixed(3)}%`;
}

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
    img.onload = () => {
      placeOverlay(img);
      next(FRAME_MS);
    };
    img.onerror = () => next(FRAME_MS * 4);
    img.src = `/api/frame.jpg?warped=${warped ? 1 : 0}&t=${Date.now()}`;
  };
  load();
}

// Marks scale with the board, so they look the same on a 1280- or a 1920-wide projector.
function drawOverlay(regions, size) {
  const svg = $("ink-overlay");
  const show = warped && $("ink-toggle").checked;
  const unit = size[0] / 1920;
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${size[0]} ${size[1]}`);
  if (!show) return;
  for (const r of regions) {
    svg.append(svgNode("rect", { x: r.x, y: r.y, width: r.w, height: r.h, rx: 6 * unit }));
    svg.append(svgNode("circle", { cx: r.cx, cy: r.cy, r: 8 * unit }));
  }
  if (target) svg.append(svgNode("circle", { class: "target", cx: target.x, cy: target.y, r: 26 * unit }));
  if (highlight && Date.now() - highlight.at < HIGHLIGHT_MS) {
    const { x, y, w, h } = highlight;
    svg.append(svgNode("rect", { class: "highlight", x, y, width: w, height: h, rx: 10 * unit }));
  }
}

export function setTarget(x, y) {
  target = Number.isFinite(x) && Number.isFinite(y) ? { x, y } : null;
}

// The avatar_update's "highlight" ([x, y, w, h] projector px, or null). Servers without it send no key at all.
export function setHighlight(rect) {
  const ok = Array.isArray(rect) && rect.length === 4 && rect.every(Number.isFinite) && rect[2] > 0 && rect[3] > 0;
  highlight = ok ? { x: rect[0], y: rect[1], w: rect[2], h: rect[3], at: Date.now() } : null;
}

export function initCamera() {
  const [w, h] = projectorSize();
  // The box takes the projector's shape, so the lined-up picture fills it with no bars.
  $("viewport").style.aspectRatio = `${w} / ${h}`;
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
    if (!warped || !$("ink-toggle").checked) return drawOverlay([], projectorSize());
    try {
      const data = await request("/api/ink");
      $("ink-count").textContent = data.regions.length;
      drawOverlay(data.regions, data.size || projectorSize());
    } catch (err) {
      $("ink-count").textContent = "–";
      throw err;
    }
  });
}
