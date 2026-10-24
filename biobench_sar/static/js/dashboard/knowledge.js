import { $, el, request, setNote } from "./api.js";

const UPLOAD_TIMEOUT_MS = 300000;

async function refreshDocs() {
  const { documents, total_chunks: total } = await request("/api/documents");
  $("kb-total").textContent = total;
  $("docs").replaceChildren(
    ...documents.map((d) => {
      const remove = el("button", { className: "icon-btn", type: "button", textContent: "remove", title: `Remove ${d.source}` });
      remove.addEventListener("click", () => removeDoc(d.source, remove));
      return el(
        "li",
        {},
        el("span", { className: "type", textContent: d.type || "?" }),
        el("span", { className: "name", textContent: d.source, title: d.source }),
        el("span", { className: "n", textContent: `${d.chunks}` }),
        remove,
      );
    }),
  );
}

async function removeDoc(source, button) {
  if (!confirm(`Remove "${source}" from the knowledge base?`)) return;
  button.disabled = true;
  try {
    await request(`/api/documents/${encodeURIComponent(source)}`, { method: "DELETE" });
    setNote($("upload-msg"), `Removed ${source}.`, "ok");
  } catch (err) {
    setNote($("upload-msg"), `Could not remove ${source}: ${err.message}`, "bad");
  }
  await refreshDocs().catch(() => {});
}

async function upload(files) {
  if (!files.length) return;
  const msg = $("upload-msg");
  const form = new FormData();
  for (const f of files) form.append("files", f);
  setNote(msg, `Embedding ${files.length} file(s)… the first upload also loads the embedding model.`);
  try {
    const res = await request("/api/upload", { method: "POST", body: form, timeout: UPLOAD_TIMEOUT_MS });
    const failed = res.files.filter((f) => f.error);
    const ok = res.files.length - failed.length;
    const summary = `Added ${res.total_chunks} chunks from ${ok} file(s).`;
    setNote(msg, failed.length ? `${summary} Failed: ${failed.map((f) => `${f.name} (${f.error})`).join("; ")}` : summary, failed.length ? "warn" : "ok");
  } catch (err) {
    setNote(msg, err.status === 413 ? "Upload too large (50 MB max)." : `Upload failed: ${err.message}`, "bad");
  }
  await refreshDocs().catch(() => {});
}

async function search(q) {
  const hits = $("hits");
  if (!q) return hits.replaceChildren();
  const { results } = await request(`/api/search?q=${encodeURIComponent(q)}&k=4`);
  hits.replaceChildren(
    ...(results.length
      ? results.map((r) =>
          el("li", {}, el("div", { className: "src" }, el("span", { textContent: r.source }), el("span", { textContent: r.score.toFixed(2) })), el("p", { textContent: r.text })),
        )
      : [el("li", { textContent: "No matching chunks." })]),
  );
}

export function initKnowledge() {
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
  $("search-form").addEventListener("submit", (e) => {
    e.preventDefault();
    search($("search-input").value.trim()).catch((err) => setNote($("upload-msg"), `Search failed: ${err.message}`, "bad"));
  });
  refreshDocs().catch(() => {});
}
