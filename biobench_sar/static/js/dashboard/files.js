import { $, el, request, setNote } from "./api.js";
import { bindText, onLanguageChange, plural, t } from "../i18n.js";
import { logEvent } from "./log.js";
import { onProfileChange, words } from "./profile.js";
import { setBadge, suggestStep } from "./tabs.js";

const UPLOAD_TIMEOUT_MS = 300000;
const offlineHint = () => t("hint.offline");
// File extension (or the ingestor's type) -> the files.kind.* word for it.
const KINDS = {
  pdf: "pdf",
  docx: "docx",
  pptx: "pptx",
  csv: "csv",
  md: "md",
  txt: "txt",
  png: "photo",
  jpg: "photo",
  jpeg: "photo",
  image: "photo",
};

let documents = [];
// Until the first /api/documents reply, `documents` is only the page's placeholder, not "no files".
let loaded = false;

// The extension first: the ingestor reports every photo as type "image".
function kindOf(doc) {
  const ext = doc.source.includes(".") ? doc.source.split(".").pop().toLowerCase() : "";
  const kind = Object.hasOwn(KINDS, ext) ? KINDS[ext] : KINDS[(doc.type || "").toLowerCase()];
  return t(`files.kind.${kind || "other"}`);
}

function isActive(doc) {
  return doc.active !== false;
}

function summary() {
  const total = documents.length;
  const active = documents.filter(isActive).length;
  $("use-all-btn").hidden = total === 0 || active === total;
  if (total === 0) return words().noFiles;
  if (active === total) return plural("files.usingAll", total);
  if (active === 0) return words().noneTicked;
  return t("files.usingSome", { active, total });
}

// Ids from the row index: names that differ only in punctuation would otherwise share one.
function fileRow(doc, index) {
  const id = `use-${index}`;
  const box = el("input", { type: "checkbox", id, checked: isActive(doc) });
  box.addEventListener("change", () => saveSelection());
  const remove = el("button", {
    type: "button",
    className: "remove-btn",
    textContent: t("files.remove"),
    "aria-label": t("files.removeLabel", { name: doc.source }),
  });
  remove.addEventListener("click", () => removeFile(doc.source, remove));
  const sections = plural("files.sections", doc.chunks);
  return el(
    "li",
    { dataset: { active: isActive(doc) ? "1" : "0", source: doc.source } },
    box,
    el("label", { htmlFor: id, className: "file-name", textContent: doc.source }, el("span", { className: "file-meta", textContent: t("files.meta", { kind: kindOf(doc), sections }) })),
    remove,
  );
}

function draw() {
  $("files").replaceChildren(...documents.map((doc, i) => fileRow(doc, i)));
  bindText($("files-summary"), summary());
  const total = documents.length;
  const active = documents.filter(isActive).length;
  if (total === 0) {
    setBadge("files", words().filesBadge, "warn");
  } else if (active === 0) {
    setBadge("files", t("files.badge.noneTicked"), "warn");
  } else {
    setBadge("files", active === total ? t("files.badge.all", { n: total }) : t("files.badge.some", { active, total }), "ok");
  }
}

export function renderFiles(body) {
  loaded = true;
  documents = body.documents || [];
  draw();
  if (documents.length === 0) suggestStep("files");
}

export async function refreshFiles() {
  renderFiles(await request("/api/documents"));
}

async function saveSelection(all = false) {
  const ticked = [...$("files").querySelectorAll("li")].filter((li) => li.querySelector("input").checked).map((li) => li.dataset.source);
  const payload = all || ticked.length === documents.length ? { all: true } : { sources: ticked };
  try {
    renderFiles(await request("/api/documents/selection", { method: "PUT", json: payload }));
  } catch (err) {
    logEvent("selection", err.message);
    setNote($("upload-msg"), () => t("files.selectionFailed", { hint: offlineHint() }), "bad");
    await refreshFiles().catch(() => {});
  }
}

async function removeFile(source, button) {
  if (!confirm(t("files.confirmRemove", { name: source }))) return;
  button.disabled = true;
  try {
    await request(`/api/documents/${encodeURIComponent(source)}`, { method: "DELETE" });
    setNote($("upload-msg"), () => t("files.removed", { name: source }), "ok");
  } catch (err) {
    logEvent("remove", `${source}: ${err.message}`);
    setNote($("upload-msg"), () => t("files.removeFailed", { name: source, hint: offlineHint() }), "bad");
  }
  await refreshFiles().catch(() => {});
}

// The reason's key only: logging happens once, when the reply arrives, not on every language change.
function failureKey(f) {
  if (/unsupported/i.test(f.error)) return "files.fail.unsupported";
  // Saving a scan again as a PDF still gives a picture of text, so that advice would not help.
  if (/no readable text/i.test(f.error)) return "files.fail.scan";
  logEvent("upload", `${f.name}: ${f.error}`);
  return "files.fail.other";
}

function uploadNote(added, failed) {
  const parts = [];
  if (added.length) parts.push(plural("files.added", added.length, { names: added.join(", ") }));
  if (failed.length) {
    const list = failed.map(([name, key]) => t(key, { name })).join("; ");
    parts.push(plural("files.unread", failed.length, { list }));
  }
  return parts.join(" ");
}

async function upload(files) {
  if (!files.length) return;
  const msg = $("upload-msg");
  const form = new FormData();
  for (const f of files) form.append("files", f);
  const count = files.length;
  setNote(msg, () => plural("files.reading", count));
  try {
    const res = await request("/api/upload", { method: "POST", body: form, timeout: UPLOAD_TIMEOUT_MS });
    const failed = res.files.filter((f) => f.error).map((f) => [f.name, failureKey(f)]);
    const added = res.files.filter((f) => !f.error).map((f) => f.name);
    setNote(msg, () => uploadNote(added, failed), failed.length ? "warn" : "ok");
  } catch (err) {
    if (err.status !== 413) logEvent("upload", err.message);
    setNote(msg, () => (err.status === 413 ? t("files.tooBig") : t("files.uploadFailed", { hint: offlineHint() })), "bad");
  }
  await refreshFiles().catch(() => {});
}

async function search(q) {
  const hits = $("hits");
  if (!q) return hits.replaceChildren();
  const { results } = await request(`/api/search?q=${encodeURIComponent(q)}&k=4`);
  hits.replaceChildren(
    ...(results.length
      ? results.map((r) => el("li", {}, el("span", { className: "src", textContent: r.source }), r.text))
      : [el("li", { textContent: t("search.none") })]),
  );
}

export function initFiles() {
  const drop = $("drop");
  const input = $("file-input");
  input.addEventListener("change", () => {
    upload([...input.files]);
    input.value = "";
  });
  for (const ev of ["dragenter", "dragover"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.dataset.over = "1"; });
  for (const ev of ["dragleave", "drop"]) drop.addEventListener(ev, () => { drop.dataset.over = "0"; });
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    upload([...e.dataTransfer.files]);
  });
  $("use-all-btn").addEventListener("click", () => saveSelection(true));
  // Only re-word a list that has arrived: an empty placeholder would send a first visit to this tab.
  onProfileChange(() => {
    if (loaded) renderFiles({ documents });
  });
  onLanguageChange(() => {
    if (loaded) draw();
  });
  $("search-form").addEventListener("submit", (e) => {
    e.preventDefault();
    // Shown in the results list: the upload note lives on the files tab, out of sight from here.
    search($("search-input").value.trim()).catch((err) => {
      logEvent("search", err.message);
      $("hits").replaceChildren(el("li", { textContent: t("search.failed", { hint: offlineHint() }) }));
    });
  });
  refreshFiles().catch(() => {});
}
