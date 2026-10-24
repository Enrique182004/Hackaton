import { $, el, request, setNote } from "./api.js";

const UPLOAD_TIMEOUT_MS = 300000;
const KINDS = {
  pdf: "PDF document",
  docx: "Word document",
  pptx: "Slides",
  csv: "Spreadsheet (CSV)",
  md: "Notes",
  txt: "Text file",
  png: "Photo (text read from the image)",
  jpg: "Photo (text read from the image)",
  jpeg: "Photo (text read from the image)",
};

let documents = [];

function kindOf(doc) {
  const ext = (doc.type || doc.source.split(".").pop() || "").toLowerCase();
  return KINDS[ext] || "File";
}

function isActive(doc) {
  return doc.active !== false;
}

function summary() {
  const total = documents.length;
  const active = documents.filter(isActive).length;
  $("use-all-btn").hidden = total === 0 || active === total;
  if (total === 0) return "No files yet. The assistant will answer from general knowledge only.";
  if (active === total) return `The assistant uses all ${total} file${total === 1 ? "" : "s"} when answering.`;
  if (active === 0) return "No files ticked. The assistant won't look at your notes.";
  return `The assistant uses ${active} of ${total} files when answering.`;
}

function fileRow(doc) {
  const id = `use-${doc.source.replace(/[^a-z0-9]/gi, "-")}`;
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
  documents = body.documents || [];
  $("files").replaceChildren(...documents.map(fileRow));
  $("files-summary").textContent = summary();
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
    setNote($("upload-msg"), `Couldn't save your choice: ${err.message}`, "bad");
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
    setNote($("upload-msg"), `Couldn't remove ${source}: ${err.message}`, "bad");
  }
  await refreshFiles().catch(() => {});
}

function describeFailure(f) {
  if (/unsupported/i.test(f.error)) return `${f.name} (that type of file can't be read)`;
  return `${f.name} (${f.error})`;
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
    setNote(msg, err.status === 413 ? "That upload is too big (50 MB at most)." : `Upload failed: ${err.message}`, "bad");
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
  $("search-form").addEventListener("submit", (e) => {
    e.preventDefault();
    search($("search-input").value.trim()).catch((err) => setNote($("upload-msg"), `Search failed: ${err.message}`, "bad"));
  });
  refreshFiles().catch(() => {});
}
