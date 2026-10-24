from __future__ import annotations

import os
import re
import shutil
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from biobench_sar.rag.quiz_bank import is_answer_column, strip_document_answers

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".pdf", ".txt", ".md", ".csv", ".docx", ".pptx", ".png", ".jpg", ".jpeg"}
)
TESSERACT_HINT = (
    "Tesseract OCR, which reads the text in images, was not found. Install it, then upload the image again: "
    "Windows `winget install UB-Mannheim.TesseractOCR` (or https://github.com/UB-Mannheim/tesseract/wiki), "
    "macOS `brew install tesseract`, Ubuntu `sudo apt install tesseract-ocr`. Besides PATH, the assistant looks in "
    "C:\\Program Files\\Tesseract-OCR, %LOCALAPPDATA%\\Programs\\Tesseract-OCR, /opt/homebrew/bin and /usr/local/bin."
)
# Homebrew (Apple silicon, then Intel), MacPorts and the distro package.
_POSIX_TESSERACT_DIRS = ("/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin", "/usr/bin")


# A whole document's texts (pages, slides or the one text) -> the same texts cleaned, before chunking.
Cleaner = Callable[[list[str]], list[str]]


class UnsupportedFileType(ValueError):
    pass


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    metadata: dict[str, str | int]


def _split_long(text: str, max_chars: int) -> list[str]:
    parts: list[str] = []
    rest = text.strip()
    while len(rest) > max_chars:
        cut = max(rest.rfind(sep, 0, max_chars) for sep in (". ", "\n", " "))
        if cut < max_chars // 2:
            cut = max_chars
        else:
            cut += 1
        parts.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        parts.append(rest)
    return parts


def chunk_paragraphs(text: str, max_chars: int = 1000) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(para) > max_chars:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(_split_long(para, max_chars))
        elif buf and len(buf) + 2 + len(para) <= max_chars:
            buf = f"{buf}\n\n{para}"
        else:
            if buf:
                chunks.append(buf)
            buf = para
    if buf:
        chunks.append(buf)
    return chunks


def _mk(texts: list[str], name: str, ftype: str, **meta: str | int) -> list[Chunk]:
    base: dict[str, str | int] = {"source": name, "type": ftype, **meta}
    return [Chunk(t.strip(), name, dict(base)) for t in texts if t and t.strip()]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _cleaned(parts: list[str], clean: Cleaner | None) -> list[str]:
    return clean(parts) if clean is not None else parts


def page_paragraphs(page: Any) -> str:
    """A PDF page's text with a blank line between its text blocks, the paragraphs PyMuPDF finds from the layout.

    page.get_text() puts only single newlines between paragraphs, so chunk_paragraphs would see one paragraph per
    page. A block keeps its own line breaks, which answer stripping reads line by line.
    """
    blocks = page.get_text("blocks")
    # (x0, y0, x1, y1, text, block_no, block_type); type 1 is an image, whose "text" only describes it.
    return "\n\n".join(text for b in blocks if b[6] == 0 and (text := b[4].strip()))


def _ingest_pdf(path: Path, clean: Cleaner | None = None) -> list[Chunk]:
    import pymupdf

    with pymupdf.open(str(path)) as doc:
        pages = [page_paragraphs(page) for page in doc]
    out: list[Chunk] = []
    for i, text in enumerate(_cleaned(pages, clean), start=1):
        out += _mk(chunk_paragraphs(text), path.name, "pdf", page=i)
    return out


def _ingest_csv(path: Path, max_chars: int = 1000, *, drop_answers: bool = False) -> list[Chunk]:
    import pandas as pd

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if drop_answers:
        df = df[[col for col in df.columns if not is_answer_column(str(col))]]
        if df.columns.empty:
            return []  # nothing but answers
    out: list[Chunk] = []
    group: list[str] = []
    first = 0

    def flush() -> None:
        if group:
            meta_row = first if len(group) == 1 else f"{first}-{first + len(group) - 1}"
            out.extend(_mk([" ".join(group)], path.name, "csv", row=meta_row))
            group.clear()

    for n, (_, row) in enumerate(df.iterrows(), start=1):
        cells = ", ".join(f"{col} is {val}" for col, val in row.items() if str(val).strip())
        line = f"Row {n}: {cells}."
        if group and len(" ".join(group)) + 1 + len(line) > max_chars:
            flush()
        if not group:
            first = n
        group.append(line)
    flush()
    return out


def _ingest_docx(path: Path, clean: Cleaner | None = None) -> list[Chunk]:
    import docx

    text = "\n\n".join(p.text for p in docx.Document(str(path)).paragraphs if p.text.strip())
    return _mk(chunk_paragraphs(_cleaned([text], clean)[0]), path.name, "docx")


def _ingest_pptx(path: Path, clean: Cleaner | None = None) -> list[Chunk]:
    from pptx import Presentation

    slides: list[str] = []
    for slide in Presentation(str(path)).slides:
        paras = [
            "".join(r.text for r in p.runs) or p.text
            for shape in slide.shapes
            if shape.has_text_frame
            for p in shape.text_frame.paragraphs
        ]
        slides.append("\n\n".join(p for p in paras if p.strip()))
    out: list[Chunk] = []
    for i, text in enumerate(_cleaned(slides, clean), start=1):
        out += _mk(chunk_paragraphs(text), path.name, "pptx", slide=i)
    return out


def tesseract_candidates(platform: str | None = None, environ: Mapping[str, str] | None = None) -> list[str]:
    """Where installers put tesseract, for when PATH doesn't say: the UB-Mannheim Windows installer (for everyone
    or just this user) never adds itself to PATH, and an app started from Finder gets a PATH without Homebrew."""
    platform = sys.platform if platform is None else platform
    environ = os.environ if environ is None else environ
    if platform != "win32":
        return [str(PurePosixPath(folder, "tesseract")) for folder in _POSIX_TESSERACT_DIRS]
    roots = [environ.get(key) for key in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)")]
    folders = [PureWindowsPath(root, "Tesseract-OCR") for root in roots if root]
    folders += [
        PureWindowsPath(r"C:\Program Files\Tesseract-OCR"),
        PureWindowsPath(r"C:\Program Files (x86)\Tesseract-OCR"),
    ]
    local = environ.get("LOCALAPPDATA")
    if local:
        folders += [PureWindowsPath(local, "Programs", "Tesseract-OCR"), PureWindowsPath(local, "Tesseract-OCR")]
    found: list[str] = []
    for folder in folders:
        exe = str(folder / "tesseract.exe")
        if exe.lower() not in (f.lower() for f in found):
            found.append(exe)
    return found


def find_tesseract(
    *,
    configured: str = "tesseract",
    which: Callable[[str], str | None] = shutil.which,
    is_file: Callable[[str], bool] = os.path.isfile,
    platform: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """The tesseract program to run: `configured` (pytesseract's tesseract_cmd) when it is a full path that exists,
    else the one on PATH, else the first install folder that has it; None when it isn't installed."""
    if configured != "tesseract" and is_file(configured):
        return configured
    on_path = which("tesseract")
    if on_path:
        return on_path
    return next((exe for exe in tesseract_candidates(platform, environ) if is_file(exe)), None)


def configure_tesseract() -> str | None:
    """Point pytesseract at the tesseract find_tesseract() finds (it only runs plain `tesseract` from PATH)."""
    import pytesseract

    exe = find_tesseract(configured=str(pytesseract.pytesseract.tesseract_cmd))
    if exe is not None:
        pytesseract.pytesseract.tesseract_cmd = exe
    return exe


def _ingest_image(path: Path, clean: Cleaner | None = None) -> list[Chunk]:
    import pytesseract
    from PIL import Image

    try:
        with Image.open(path) as img:
            text = pytesseract.image_to_string(img)
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(TESSERACT_HINT) from exc
    return _mk(chunk_paragraphs(_cleaned([text], clean)[0]), path.name, "image")


def ingest_file(path: str | Path, *, strip_answers: bool = False) -> list[Chunk]:
    """The file's text as chunks. strip_answers (the math profile): answer keys, "Answer:" lines, answer tables and a
    CSV's answer columns are taken out of the whole document before it is chunked, so no chunk (and no embedding)
    holds an answer. The quiz bank reads its problems and answers from the original file instead."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileType(f"Unsupported file type: {ext or p.name}")
    clean = strip_document_answers if strip_answers else None
    if ext == ".pdf":
        return _ingest_pdf(p, clean)
    if ext in (".txt", ".md"):
        return _mk(chunk_paragraphs(_cleaned([_read_text(p)], clean)[0]), p.name, ext[1:])
    if ext == ".csv":
        return _ingest_csv(p, drop_answers=strip_answers)
    if ext == ".docx":
        return _ingest_docx(p, clean)
    if ext == ".pptx":
        return _ingest_pptx(p, clean)
    if configure_tesseract() is None:
        raise RuntimeError(TESSERACT_HINT)
    return _ingest_image(p, clean)
