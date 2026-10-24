import { $, OFFLINE_HINT, el, request, setNote } from "./api.js";
import { logEvent } from "./log.js";
import { onProfileChange, words } from "./profile.js";
import { setBadge, suggestStep } from "./tabs.js";

const UPLOAD_TIMEOUT_MS = 300000;
const PHOTO = "Photo (text read from the image)";
const KINDS = {
  pdf: "PDF document",
  docx: "Word document",
  pptx: "Slides",
  csv: "Spreadsheet (CSV)",
  md: "Notes",
  txt: "Text file",
  png: PHOTO,
  jpg: PHOTO,
  jpeg: PHOTO,
  image: PHOTO,
};

let documents = [];
// Until the first /api/documents reply, `documents` is only the page's placeholder, not "no files".
let loaded = false;

// The extension first: the ingestor reports every photo as type "image".
function kindOf(doc) {
  const ext = doc.source.includes(".") ? doc.source.split(".").pop().toLowerCase() : "";
  return KINDS[ext] || KINDS[(doc.type || "").toLowerCase()] || "File";
}

function isActive(doc) {
  return doc.active !== false;
}

function summary() {
  const total = documents.length;
  const active = documents.filter(isActive).length;
  $("use-all-btn").hidden = total === 0 || active === total;
  if (total === 0) return words().noFiles;
  if (active === total) return `The assistant uses all ${total} file${total === 1 ? "" : "s"} when answering.`;
  if (active === 0) return words().noneTicked;
  return `The assistant uses ${active} of ${total} files when answering.`;
}

// Ids from the row index: names that differ only in punctuation would otherwise share one.
function fileRow(doc, index) {
  const id = `use-${index}`;
  const box = el("input", { type: "checkbox", id, checked: isActive(doc) });
  box.addEventListener("change", () => saveSelection());
  const remove = el("button", { type: "button", className: "remove-btn", textContent: "Remove", "aria-label": `Remove ${doc.source}` });
  remove.addEventListener("click", () => removeFile(doc.source, remove));
  const sections = `${doc.chunks} section${doc.chunks === 1 ? "" : "s"}`;
  return el(
    "li",
    { dataset: { active: isActive(doc) ? "1" : "0", source: doc.source } },
    box,
    el("label", { htmlFor: id, className: "file-name", textContent: doc.source }, el("span", { className: "file-meta", textContent: `${kindOf(doc)} · ${sections}` })),
    remove,
  );
}

export function renderFiles(body) {
  loaded = true;
  documents = body.documents || [];
  $("files").replaceChildren(...documents.map((doc, i) => fileRow(doc, i)));
  $("files-summary").textContent = summary();
  const total = documents.length;
  const active = documents.filter(isActive).length;
  if (total === 0) {
    setBadge("files", words().filesBadge, "warn");
    suggestStep("files");
  } else if (active === 0) {
    setBadge("files", "none ticked", "warn");
  } else {
    setBadge("files", active === total ? `${total} in use` : `${active} of ${total} in use`, "ok");
  }
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
    setNote($("upload-msg"), `Couldn't save which files are ticked. ${OFFLINE_HINT}`, "bad");
    await refreshFiles().catch(() => {});
  }
}

async function removeFile(source, button) {
  if (!confirm(`Remove “${source}”? The assistant will stop using it.`)) return;
  button.disabled = true;
  try {
    await request(`/api/documents/${encodeURIComponent(source)}`, { method: "DELETE" });
    setNote($("upload-msg"), `Removed ${source}.`, "ok");
  } catch (err) {
    logEvent("remove", `${source}: ${err.message}`);
    setNote($("upload-msg"), `Couldn't remove ${source}. ${OFFLINE_HINT}`, "bad");
  }
  await refreshFiles().catch(() => {});
}

function describeFailure(f) {
  if (/unsupported/i.test(f.error)) return `${f.name} (that type of file can't be read)`;
  // Saving a scan again as a PDF still gives a picture of text, so that advice would not help.
  if (/no readable text/i.test(f.error)) return `${f.name} (it looks like a scan with no text, so type it up or upload a clearer photo)`;
  logEvent("upload", `${f.name}: ${f.error}`);
  return `${f.name} (try saving it again as a PDF or text file)`;
}

async function upload(files) {
  if (!files.length) return;
  const msg = $("upload-msg");
  const form = new FormData();
  for (const f of files) form.append("files", f);
  setNote(msg, `Reading ${files.length} file${files.length === 1 ? "" : "s"}… the first time can take up to a minute.`);
  try {
    const res = await request("/api/upload", { method: "POST", body: form, timeout: UPLOAD_TIMEOUT_MS });
    const failed = res.files.filter((f) => f.error);
    const added = res.files.filter((f) => !f.error).map((f) => f.name);
    const parts = [];
    if (added.length) parts.push(`Added ${added.join(", ")}.`);
    if (failed.length) parts.push(`Couldn't read ${failed.map(describeFailure).join("; ")}.`);
    setNote(msg, parts.join(" "), failed.length ? "warn" : "ok");
  } catch (err) {
    if (err.status !== 413) logEvent("upload", err.message);
    setNote(msg, err.status === 413 ? "That upload is too big (50 MB at most)." : `Upload failed. ${OFFLINE_HINT}`, "bad");
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
      : [el("li", { textContent: "Nothing in your ticked files matches that." })]),
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
  $("search-form").addEventListener("submit", (e) => {
    e.preventDefault();
    // Shown in the results list: the upload note lives on the files tab, out of sight from here.
    search($("search-input").value.trim()).catch((err) => {
      logEvent("search", err.message);
      $("hits").replaceChildren(el("li", { textContent: `Search didn't work. ${OFFLINE_HINT}` }));
    });
  });
  refreshFiles().catch(() => {});
}
