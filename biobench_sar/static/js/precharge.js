(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const ACCEPT = [".pdf", ".txt", ".md", ".csv", ".docx", ".pptx", ".png", ".jpg", ".jpeg"];

  async function api(method, url, body) {
    const opts = { method, headers: {} };
    if (body !== undefined) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(body); }
    const res = await fetch(url, opts);
    let data = null;
    try { data = await res.json(); } catch (_) { /* empty body is fine */ }
    return { ok: res.ok, status: res.status, data: data };
  }

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }

  // ---- upload ----
  const drop = $("drop");
  const fileInput = $("file-input");

  function validExt(name) {
    const i = name.lastIndexOf(".");
    return i >= 0 && ACCEPT.includes(name.slice(i).toLowerCase());
  }

  function addResult(name, text, isErr) {
    const li = el("li");
    li.appendChild(el("span", "", name));
    li.appendChild(el("span", isErr ? "err-text" : "ok-text", text));
    $("upload-results").prepend(li);
  }

  async function upload(fileList) {
    const files = Array.from(fileList);
    const good = files.filter((f) => {
      const ok = validExt(f.name);
      if (!ok) addResult(f.name, "unsupported type", true);
      return ok;
    });
    if (!good.length) return;
    const fd = new FormData();
    good.forEach((f) => fd.append("files", f));
    $("upload-total").textContent = "Uploading " + good.length + " file(s)...";
    try {
      const res = await fetch("/api/upload", { method: "POST", body: fd });
      const data = await res.json().catch(() => null);
      if (!res.ok || !data) throw new Error("HTTP " + res.status);
      (data.files || []).forEach((f) =>
        f.error ? addResult(f.name, f.error, true) : addResult(f.name, f.chunks + " chunks", false));
      $("upload-total").textContent = "Total chunks: " + data.total_chunks;
    } catch (err) {
      good.forEach((f) => addResult(f.name, "upload failed: " + err.message, true));
      $("upload-total").textContent = "";
    }
  }

  drop.addEventListener("click", () => fileInput.click());
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } });
  fileInput.addEventListener("change", () => { upload(fileInput.files); fileInput.value = ""; });
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => upload(e.dataTransfer.files));
  // Stop the browser from navigating to a file dropped outside the zone.
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  // ---- status polling ----
  function renderStatus(rows) {
    const dl = $("status-list");
    dl.replaceChildren();
    rows.forEach(([k, v]) => { dl.appendChild(el("dt", "", k)); dl.appendChild(el("dd", "", String(v))); });
  }

  async function pollStatus() {
    try {
      const r = await api("GET", "/api/status");
      if (!r.ok || !r.data) throw new Error("HTTP " + r.status);
      const s = r.data;
      renderStatus([
        ["state", s.state], ["fps", typeof s.fps === "number" ? s.fps.toFixed(1) : s.fps],
        ["camera", s.camera_running ? "running" : "stopped"], ["calibrated", s.calibrated ? "yes" : "no"],
        ["doc chunks", s.doc_chunks], ["VLM", (s.vlm_backend || "?") + " / " + (s.vlm_model || "?")],
        ["audio", s.audio_available ? "available" : "unavailable"],
      ]);
    } catch (err) {
      renderStatus([["server", "unreachable (" + err.message + ")"]]);
    }
  }
  pollStatus();
  setInterval(pollStatus, 2000);

  // ---- camera preview ----
  function refreshFrames() {
    const t = Date.now();
    $("frame-raw").src = "/api/frame.jpg?warped=0&t=" + t;
    $("frame-warped").src = "/api/frame.jpg?warped=1&t=" + t;
  }
  refreshFrames();
  setInterval(refreshFrames, 500);

  // ---- controls ----
  function showResult(text) { $("control-result").textContent = text; }

  async function guarded(btn, fn) {
    btn.disabled = true;
    try { await fn(); } catch (err) { showResult("Request failed: " + err.message); }
    btn.disabled = false;
  }

  $("btn-calibrate").addEventListener("click", (e) => guarded(e.currentTarget, async () => {
    showResult("Calibrating...");
    const r = await api("POST", "/api/calibrate");
    const d = r.data || {};
    showResult(d.ok ? "Calibration OK" : "Calibration failed: " + (d.error || "HTTP " + r.status));
  }));

  $("btn-listen").addEventListener("click", (e) => guarded(e.currentTarget, async () => {
    const r = await api("POST", "/api/listen");
    showResult(r.ok ? "Listening triggered" : "Listen failed: HTTP " + r.status);
  }));

  $("ask-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const input = $("ask-text");
    const text = input.value.trim();
    if (!text) return;
    guarded(e.submitter || e.currentTarget.querySelector("button"), async () => {
      const r = await api("POST", "/api/query", { text });
      showResult(r.ok ? "Query sent" : "Query failed: HTTP " + r.status);
      if (r.ok) input.value = "";
    });
  });

  // ---- event log ----
  const log = $("log");
  function logLine(name, data) {
    const t = new Date().toLocaleTimeString();
    log.textContent += "[" + t + "] " + name + " " + JSON.stringify(data) + "\n";
    if (log.textContent.length > 20000) log.textContent = log.textContent.slice(-15000);
    log.scrollTop = log.scrollHeight;
  }

  const conn = $("conn");
  function setConn(text, ok) { conn.textContent = "socket: " + text; conn.classList.toggle("ok", !!ok); }

  if (typeof io === "function") {
    const socket = io();
    socket.on("connect", () => setConn("connected", true));
    socket.on("disconnect", () => setConn("disconnected", false));
    socket.on("connect_error", () => setConn("error", false));
    ["avatar_update", "status", "transcript", "speak"].forEach((n) => socket.on(n, (d) => logEvent(n, d)));
  } else {
    setConn("client not loaded", false);
  }
  function logEvent(n, d) { logLine(n, d); }
})();
