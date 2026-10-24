"""Quiz problems pulled out of uploaded worksheets and quiz banks, persisted as JSON per source file."""

from __future__ import annotations

import contextlib
import csv
import json
import logging
import os
import re
import tempfile
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_LEVEL = 1
PROBLEM_COLUMNS = ("problem", "question", "exercise")
ANSWER_COLUMNS = ("answer", "solution")
TOPIC_COLUMNS = ("topic", "subject", "unit")
LEVEL_COLUMNS = ("level", "difficulty")

_NUMBERED = re.compile(
    r"^\s*(?:(?:problem|question|exercise|problema|pregunta|ejercicio|q)\s*#?\s*)?"
    r"(?P<n>\d{1,3})\s*(?:[.)]\s+|:\s*)(?P<text>\S.*)$",
    re.I,
)
_ANSWER_LINE = re.compile(r"^\s*[-*>]*\s*(?:answer|ans|solution|respuesta)\s*[:=]\s*(?P<text>\S.*)$", re.I)
_INLINE_ANSWER = re.compile(r"[(\[]?\s*\b(?:answer|ans|respuesta)\b\s*[:=]\s*(?P<text>.+?)\s*[)\]]?\s*$", re.I)
_ANSWER_KEY = re.compile(r"^\W*(?:answer\s+key|answers|solutions|respuestas|soluciones)\W*$", re.I)
# A tutoring guide's numbered tips can contain math too; only files that present themselves as worksheets count.
_WORKSHEET_HINT = re.compile(
    r"worksheet|quiz|exercise|homework|assignment|practice|problem\s*set|drill|\btest\b|ejercicio|tarea|examen|"
    r"pr[aá]ctica",
    re.I,
)
_LEVEL = re.compile(r"\b(?:level|difficulty)\s*[:\-]?\s*(?P<n>\d+)\b", re.I)
_HEADING = re.compile(r"^\s*#+\s*(?P<text>.+?)\s*#*\s*$")
_TOPIC = re.compile(r"^\s*topic\s*:\s*(?P<text>.+)$", re.I)
_MATH_OPERATOR = re.compile(r"[=+\-*/×÷·^<>≤≥√%]")
_MATH_WORD = re.compile(
    r"\b(?:solve|simplify|evaluate|factor|expand|find|calculate|compute|what\s+is|how\s+much|how\s+many|"
    r"round|convert|write|graph|plot)\b",
    re.I,
)


@dataclass(frozen=True)
class QuizProblem:
    problem: str
    answer: str | None
    topic: str
    level: int
    source: str

    def to_json(self) -> dict[str, Any]:
        return {"problem": self.problem, "answer": self.answer, "topic": self.topic, "level": self.level}


def _looks_like_math(text: str) -> bool:
    # Worksheets mix problems with numbered instructions ("1. Show your work"); only keep lines with math in them.
    return any(ch.isdigit() for ch in text) and bool(_MATH_OPERATOR.search(text) or _MATH_WORD.search(text))


def _clean(text: str) -> str:
    text = text.replace("**", "").replace("`", "")
    return re.sub(r"[\s_]+$", "", text).strip()


def _level(value: Any) -> int:
    try:
        return max(1, round(float(str(value).strip())))
    except (TypeError, ValueError, OverflowError):
        return DEFAULT_LEVEL


def _pick(row: dict[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        value = (row.get(name) or "").strip()
        if value:
            return value
    return ""


def _parse_csv(path: Path, source: str) -> list[QuizProblem]:
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return []
        reader.fieldnames = [(name or "").strip().lower() for name in reader.fieldnames]
        if not any(name in reader.fieldnames for name in PROBLEM_COLUMNS):
            return []
        out: list[QuizProblem] = []
        for row in reader:
            problem = _pick(row, PROBLEM_COLUMNS)
            if not problem:
                continue
            answer = _pick(row, ANSWER_COLUMNS) or None
            level = _level(_pick(row, LEVEL_COLUMNS))
            out.append(QuizProblem(problem, answer, _pick(row, TOPIC_COLUMNS), level, source))
        return out


def parse_worksheet_text(text: str, source: str) -> list[QuizProblem]:
    """Numbered problems ("1. 3x + 5 = 20", "Problem 2: ...") with optional "Answer: ..." lines or an answer key."""
    problems: list[dict[str, Any]] = []
    by_number: dict[int, dict[str, Any]] = {}
    topic, level, in_key = "", DEFAULT_LEVEL, False
    for raw in text.splitlines():
        line = _clean(raw)
        if not line:
            continue
        if _ANSWER_KEY.match(line):
            in_key = True
            continue
        heading = _HEADING.match(line) or _TOPIC.match(line)
        if heading:
            in_key, title = False, heading.group("text")
            found = _LEVEL.search(title)
            if found:
                level = _level(found.group("n"))
                title = _LEVEL.sub("", title).strip(" -—–:")
            if title:
                topic = title
            continue
        found = _LEVEL.fullmatch(line.strip(" -—–:"))
        if found:
            level = _level(found.group("n"))
            continue
        answer = _ANSWER_LINE.match(line)
        if answer:
            if problems and problems[-1]["answer"] is None:
                problems[-1]["answer"] = answer.group("text").strip()
            continue
        numbered = _NUMBERED.match(line)
        if not numbered:
            continue
        number, body = int(numbered.group("n")), numbered.group("text").strip()
        if in_key:
            target = by_number.get(number)
            if target is not None and target["answer"] is None:
                target["answer"] = body
            continue
        inline = _INLINE_ANSWER.search(body)
        given = None
        if inline and inline.start() > 0:
            given, body = inline.group("text").strip(), body[: inline.start()].strip()
        if not _looks_like_math(body):
            continue
        item = {"problem": body, "answer": given or None, "topic": topic, "level": level}
        problems.append(item)
        by_number.setdefault(number, item)
    return [QuizProblem(p["problem"], p["answer"], p["topic"], p["level"], source) for p in problems]


def _document_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".md", ".txt"):
        return path.read_text(encoding="utf-8", errors="replace")
    if ext == ".pdf":
        import pymupdf

        with pymupdf.open(str(path)) as doc:
            return "\n".join(page.get_text() for page in doc)
    if ext == ".docx":
        import docx

        return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    return ""


def is_worksheet(source: str, text: str) -> bool:
    """The file name, first line or a markdown heading calls it a worksheet, quiz, exercise set, homework..."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    titles = [re.sub(r"[_.\-]+", " ", Path(source).stem), *lines[:1], *(line for line in lines if line.startswith("#"))]
    return any(_WORKSHEET_HINT.search(title) for title in titles)


def parse_quiz_file(path: str | Path, source: str) -> list[QuizProblem]:
    """Problems in a quiz-bank CSV or a worksheet (.md/.txt/.pdf/.docx). Never raises: other files have none."""
    p = Path(path)
    try:
        if p.suffix.lower() == ".csv":
            return _parse_csv(p, source)
        text = _document_text(p)
        return parse_worksheet_text(text, source) if is_worksheet(source, text) else []
    except Exception as exc:
        log.warning("could not read quiz problems from %s: %s", source, exc)
        return []


class QuizBank:
    """Every source file's quiz problems, persisted as JSON. Thread-safe; a missing or bad file starts empty."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._sources: dict[str, list[QuizProblem]] = self._load()

    def _load(self) -> dict[str, list[QuizProblem]]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        sources = data.get("sources") if isinstance(data, dict) else None
        if not isinstance(sources, dict):
            return {}
        out: dict[str, list[QuizProblem]] = {}
        for source, items in sources.items():
            if not isinstance(items, list):
                continue
            problems = [
                QuizProblem(
                    str(item["problem"]),
                    None if item.get("answer") is None else str(item["answer"]),
                    str(item.get("topic") or ""),
                    _level(item.get("level", DEFAULT_LEVEL)),
                    str(source),
                )
                for item in items
                if isinstance(item, dict) and str(item.get("problem") or "").strip()
            ]
            if problems:
                out[str(source)] = problems
        return out

    def sources(self) -> list[str]:
        with self._lock:
            return list(self._sources)

    def replace_source(self, source: str, problems: list[QuizProblem]) -> None:
        with self._lock:
            if problems:
                self._sources.pop(source, None)  # re-insert at the end: file order follows upload order
                self._sources[source] = [p if p.source == source else replace(p, source=source) for p in problems]
            elif self._sources.pop(source, None) is None:
                return
            self._save()

    def remove_source(self, source: str) -> None:
        with self._lock:
            if self._sources.pop(source, None) is not None:
                self._save()

    def problems(self, sources: list[str] | None = None) -> list[QuizProblem]:
        """Problems from `sources` (None = every file), easiest level first, then in file order."""
        with self._lock:
            wanted = None if sources is None else set(sources)
            picked = [p for name, items in self._sources.items() if wanted is None or name in wanted for p in items]
        return sorted(picked, key=lambda p: p.level)

    def _save(self) -> None:
        payload = {"sources": {name: [p.to_json() for p in items] for name, items in self._sources.items()}}
        tmp = ""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=f".{self._path.name}.", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp, self._path)
        except OSError as exc:
            log.warning("could not save quiz bank to %s: %s", self._path, exc)
            if tmp:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
