import { $, el, request, setNote } from "./api.js";

let pending = null;

function sourcesTag(sources) {
  return sources.length ? el("span", { className: "tag", textContent: `📄 ${sources.join(", ")}` }) : null;
}

function turnNode(item) {
  return el(
    "li",
    { className: "turn", dataset: { state: item.animation_state } },
    el("p", { className: "q", textContent: item.query }),
    el("p", { className: "a", textContent: item.spoken_text }),
    el(
      "div",
      { className: "meta" },
      el("span", { className: "tag tag-state", textContent: item.animation_state }),
      el("span", { textContent: `${item.latency_s.toFixed(1)}s` }),
      el("span", { textContent: `→ (${item.target_x}, ${item.target_y})` }),
      el("span", { textContent: item.had_frame ? `${item.ink_regions} ink regions` : "no camera frame" }),
      sourcesTag(item.sources),
    ),
  );
}

function showEmpty() {
  $("timeline-empty").hidden = $("timeline").children.length > 0;
}

export function addInteraction(item) {
  if (pending) {
    pending.remove();
    pending = null;
  }
  $("timeline").prepend(turnNode(item));
  showEmpty();
}

export function showPending(text) {
  if (pending) pending.remove();
  pending = el("li", { className: "turn pending" }, el("p", { className: "q", textContent: text }), el("p", { className: "a", textContent: "Thinking…" }));
  $("timeline").prepend(pending);
  showEmpty();
}

export async function loadHistory() {
  const { interactions } = await request("/api/history");
  $("timeline").replaceChildren(...interactions.map(turnNode));
  showEmpty();
}

export function initAsk() {
  const input = $("ask-input");
  const msg = $("ask-msg");
  $("ask-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return input.focus();
    $("ask-btn").disabled = true;
    try {
      await request("/api/query", { method: "POST", json: { text } });
      input.value = "";
      setNote(msg, "Sent — watch the projector.", "ok");
    } catch (err) {
      setNote(msg, err.status === 409 ? "Still answering the last question — try again in a moment." : `Could not send: ${err.message}`, "warn");
    } finally {
      $("ask-btn").disabled = false;
    }
  });
  $("listen-btn").addEventListener("click", async () => {
    try {
      await request("/api/listen", { method: "POST" });
      setNote(msg, "Listening — speak now.", "ok");
    } catch (err) {
      setNote(msg, err.status === 409 ? "Already recording." : `Microphone unavailable: ${err.message}`, "warn");
    }
  });
  showEmpty();
}
