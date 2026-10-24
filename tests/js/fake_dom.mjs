// Just enough DOM, fetch and socket.io for the vanilla static/js pages to boot under node.
// Elements exist only for ids the real template declares, so an id typo in the JS fails the test.

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
    this.files = [];
    this.offsetWidth = 100;
    this.offsetHeight = 40;
  }

  get lastChild() { return this.children.at(-1) ?? null; }

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
    this.append(...nodes);
  }
  remove() {
    if (!this.parentNode) return;
    this.parentNode.children = this.parentNode.children.filter((c) => c !== this);
    this.parentNode = null;
  }

  setAttribute(name, value) { this.attributes[name] = String(value); }
  getAttribute(name) { return this.attributes[name] ?? null; }
  focus() {}
  blur() {}

  // Tag selectors only; the child is created on first use, as the template markup would provide it.
  querySelector(tag) {
    const want = tag.toUpperCase();
    let found = this.children.find((c) => c instanceof FakeElement && c.tagName === want);
    if (!found) this.appendChild((found = new FakeElement(tag)));
    return found;
  }

  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  // Resolves once every (possibly async) listener has finished.
  fire(type, extra = {}) {
    const event = { type, target: this, preventDefault() {}, stopPropagation() {}, ...extra };
    return Promise.all((this.listeners[type] || []).map((fn) => fn(event)));
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

export function installDom(ids, { bodyData = {} } = {}) {
  const body = new FakeElement("body");
  Object.assign(body.dataset, bodyData);
  for (const id of ids) {
    const node = new FakeElement("div");
    node.id = id;
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
    querySelectorAll: () => [],
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
};

const DEFAULT_ROUTES = {
  "GET /api/status": { status: 200, body: STATUS },
  "GET /api/documents": { status: 200, body: { documents: [], total_chunks: 0 } },
  "GET /api/history": { status: 200, body: { interactions: [] } },
  "GET /api/config": { status: 200, body: {} },
  "GET /api/ink": { status: 200, body: { regions: [], size: [1920, 1080] } },
  "POST /api/query": { status: 202, body: { accepted: true } },
  "POST /api/listen": { status: 202, body: { accepted: true } },
};

// A route is a {status, body} reply or a function returning one (or a promise of one).
export function installFetch(routes = {}) {
  const table = { ...DEFAULT_ROUTES, ...routes };
  const calls = [];
  globalThis.fetch = async (url, init = {}) => {
    const path = String(url).split("?")[0];
    const method = init.method || "GET";
    calls.push({ method, path });
    const route = table[`${method} ${path}`] ?? { status: 404, body: { error: "not found" } };
    const reply = await (typeof route === "function" ? route() : route);
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
