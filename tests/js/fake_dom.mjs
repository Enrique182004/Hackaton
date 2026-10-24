// Just enough DOM, fetch and socket.io for the vanilla static/js pages to boot under node.
// Elements exist only for what the real template declares (its ids and its data-i18n texts, with their tags and
// data-* attributes), so an id typo in the JS fails the test.

class FakeClassList {
  #names = new Set();
  add(...names) { for (const n of names) this.#names.add(n); }
  remove(...names) { for (const n of names) this.#names.delete(n); }
  contains(name) { return this.#names.has(name); }
  toggle(name, force) {
    const on = force === undefined ? !this.#names.has(name) : Boolean(force);
    if (on) this.#names.add(name); else this.#names.delete(name);
    return on;
  }
}

export class FakeElement {
  #text = "";

  constructor(tag) {
    this.tagName = tag.toUpperCase();
    this.id = "";
    this.children = [];
    this.parentNode = null;
    this.dataset = {};
    this.style = {};
    this.attributes = {};
    this.listeners = {};
    this.classList = new FakeClassList();
    this.textContent = "";
    this.className = "";
    this.title = "";
    this.type = "";
    this.value = "";
    this.src = "";
    this.alt = "";
    this.hidden = false;
    this.disabled = false;
    this.checked = false;
    this.htmlFor = "";
    this.placeholder = "";
    this.files = [];
    this.offsetWidth = 100;
    this.offsetHeight = 40;
  }

  get lastChild() { return this.children.at(-1) ?? null; }

  // As in a browser, setting the text replaces whatever was inside.
  get textContent() { return this.#text; }
  set textContent(value) {
    for (const c of this.children) if (c instanceof FakeElement) c.parentNode = null;
    this.children = [];
    this.#text = value == null ? "" : String(value);
  }

  #adopt(node, atFront) {
    if (node instanceof FakeElement) {
      node.remove();
      node.parentNode = this;
    }
    if (atFront) this.children.unshift(node); else this.children.push(node);
  }

  append(...nodes) { for (const n of nodes) this.#adopt(n, false); }
  appendChild(node) { this.#adopt(node, false); return node; }
  prepend(...nodes) { for (const n of [...nodes].reverse()) this.#adopt(n, true); }
  replaceChildren(...nodes) {
    for (const c of [...this.children]) if (c instanceof FakeElement) c.remove();
    this.children = [];
    this.#text = "";
    this.append(...nodes);
  }
  remove() {
    if (!this.parentNode) return;
    this.parentNode.children = this.parentNode.children.filter((c) => c !== this);
    this.parentNode = null;
  }

  // A test may pin the box with `node.rect = {left, top, width, height}`; otherwise it comes from the inline style
  // (left/top/width/height in px), with offsetWidth/offsetHeight for a size the style doesn't give.
  getBoundingClientRect() {
    const r = this.rect ?? {
      left: parseFloat(this.style.left) || 0,
      top: parseFloat(this.style.top) || 0,
      width: parseFloat(this.style.width) || this.offsetWidth,
      height: parseFloat(this.style.height) || this.offsetHeight,
    };
    return { ...r, x: r.left, y: r.top, right: r.left + r.width, bottom: r.top + r.height };
  }

  setAttribute(name, value) { this.attributes[name] = String(value); }
  getAttribute(name) { return this.attributes[name] ?? null; }
  focus() {}
  blur() {}

  // Tag selectors only. querySelector creates the child on first use, as the template markup would provide it.
  querySelector(tag) {
    const want = tag.toUpperCase();
    let found = this.children.find((c) => c instanceof FakeElement && c.tagName === want);
    if (!found) this.appendChild((found = new FakeElement(tag)));
    return found;
  }

  querySelectorAll(tag) {
    const want = tag.toUpperCase();
    const out = [];
    for (const c of this.children) {
      if (!(c instanceof FakeElement)) continue;
      if (c.tagName === want) out.push(c);
      out.push(...c.querySelectorAll(tag));
    }
    return out;
  }

  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  requestSubmit() { return this.fire("submit"); }
  // Resolves once every (possibly async) listener has finished.
  fire(type, extra = {}) {
    const event = { type, target: this, preventDefault() {}, stopPropagation() {}, ...extra };
    return Promise.all((this.listeners[type] || []).map((fn) => fn(event)));
  }

  *descendants() {
    for (const c of this.children) {
      if (!(c instanceof FakeElement)) continue;
      yield c;
      yield* c.descendants();
    }
  }

  find(id) {
    for (const c of this.children) {
      if (!(c instanceof FakeElement)) continue;
      if (c.id === id) return c;
      const deep = c.find(id);
      if (deep) return deep;
    }
    return null;
  }
}

// `nodes` are ids, or {id, tag, data, hidden} as test_frontend.template_nodes reads them from the template.
export function installDom(nodes, { bodyData = {} } = {}) {
  const body = new FakeElement("body");
  Object.assign(body.dataset, bodyData);
  for (const spec of nodes) {
    const { id = "", tag = "div", data = {}, hidden = false } = typeof spec === "string" ? { id: spec } : spec;
    const node = new FakeElement(tag);
    node.id = id;
    node.hidden = hidden;
    Object.assign(node.dataset, data);
    body.appendChild(node);
  }
  const doc = {
    body,
    documentElement: new FakeElement("html"),
    hidden: false,
    activeElement: null,
    listeners: {},
    getElementById: (id) => body.find(id),
    createElement: (tag) => new FakeElement(tag),
    createElementNS: (_ns, tag) => new FakeElement(tag),
    // "[data-foo-bar]" selectors only (what i18n.js asks for); anything else finds nothing, as before.
    querySelectorAll(selector) {
      const attr = /^\[data-([a-z0-9-]+)\]$/.exec(selector);
      if (!attr) return [];
      const key = attr[1].replace(/-([a-z0-9])/g, (_, c) => c.toUpperCase());
      return [...body.descendants()].filter((node) => Object.hasOwn(node.dataset, key));
    },
    addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); },
  };
  globalThis.document = doc;
  globalThis.window = { innerWidth: 1920, innerHeight: 1080, addEventListener() {} };
  globalThis.Audio = class { play() { return Promise.resolve(); } pause() {} addEventListener() {} };
  globalThis.confirm = () => true;
  return doc;
}

const STATUS = {
  fps: 30, camera_running: true, calibrated: false, doc_chunks: 0, vlm_backend: "mock", vlm_model: "mock",
  audio_available: true, listen_mode: "push", tts_backend: "none", last_latency_s: null, uptime_s: 1, state: "idle",
  language: "auto", current_language: "en", glare: { fraction: 0, severe: false, hotspots: [] },
};

const DEFAULT_ROUTES = {
  "GET /api/status": { status: 200, body: STATUS },
  "GET /api/documents": { status: 200, body: { documents: [], total_chunks: 0 } },
  "GET /api/history": { status: 200, body: { interactions: [] } },
  "GET /api/config": { status: 200, body: {} },
  "GET /api/ink": { status: 200, body: { regions: [], size: [1920, 1080] } },
  "GET /api/tutor": { status: 200, body: { mode: "help", learn_topic: null, quiz: null } },
  "GET /api/language": { status: 200, body: { setting: "auto", current: "en" } },
  "POST /api/query": { status: 202, body: { accepted: true } },
  "POST /api/listen": { status: 202, body: { accepted: true } },
};

// A route is a {status, body} reply or a function of the fetch init returning one (or a promise of one).
export function installFetch(routes = {}) {
  const table = { ...structuredClone(DEFAULT_ROUTES), ...routes };
  const calls = [];
  globalThis.fetch = async (url, init = {}) => {
    const path = String(url).split("?")[0];
    const method = init.method || "GET";
    calls.push({ method, path, body: init.body });
    const route = table[`${method} ${path}`] ?? { status: 404, body: { error: "not found" } };
    const reply = await (typeof route === "function" ? route(init) : route);
    return { ok: reply.status >= 200 && reply.status < 300, status: reply.status, json: async () => reply.body };
  };
  return { calls, setStatus: (patch) => Object.assign(table["GET /api/status"].body, patch) };
}

export function installIo() {
  const handlers = {};
  const emitted = [];
  const socket = {
    on(name, fn) { (handlers[name] ||= []).push(fn); },
    emit(name, payload) { emitted.push([name, payload]); },
  };
  globalThis.io = () => socket;
  return { emitted, fire: (name, data) => { for (const fn of handlers[name] || []) fn(data); } };
}

// Lets every pending promise chain (fetch -> json -> handler) run to completion.
export async function settle() {
  for (let i = 0; i < 5; i++) await new Promise((resolve) => setImmediate(resolve));
}
