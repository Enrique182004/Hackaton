"""Quiz problems pulled out of uploaded worksheets and quiz banks, persisted as JSON per source file."""

from __future__ import annotations

import contextlib
import csv
import json
import logging
import os
import random
import re
import tempfile
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_LEVEL = 1
# A worksheet without "Level N" headings is taken to get harder as it goes (the usual layout): its problems get
# levels 1..POSITION_LEVELS by position, so they mix with a quiz bank's levels instead of all being level 1.
POSITION_LEVELS = 5
INVENTED_SOURCE = "(invented)"
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
_ANSWER_LABEL = re.compile(r"^\s*[-*>]*\s*(?:answer|ans|solution|respuesta)\s*[:=]\s*$", re.I)
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
    levelled = last_kept = False
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
                level, levelled = _level(found.group("n")), True
                title = _LEVEL.sub("", title).strip(" -—–:")
            if title:
                topic = title
            continue
        found = _LEVEL.fullmatch(line.strip(" -—–:"))
        if found:
            level, levelled = _level(found.group("n")), True
            continue
        answer = _ANSWER_LINE.match(line)
        if answer:
            # Only under the problem it belongs to: a skipped instruction line can have its own answer.
            if last_kept and problems[-1]["answer"] is None:
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
        last_kept = _looks_like_math(body)
        if not last_kept:
            continue
        item = {"problem": body, "answer": given or None, "topic": topic, "level": level}
        problems.append(item)
        by_number.setdefault(number, item)
    if not levelled:
        span = min(len(problems), POSITION_LEVELS)
        for i, item in enumerate(problems):
            item["level"] = DEFAULT_LEVEL + i * span // len(problems)
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


_PREFIXED = re.compile(r"^\s*(?:problem|question|exercise|problema|pregunta|ejercicio)\s*#?\s*\d{1,3}\s*[:.)]", re.I)
# Words that can sit next to a real problem's math; any other word makes a numbered line prose (a tip, a step).
_PROBLEM_WORDS = frozenset(
    {"solve", "simplify", "evaluate", "factor", "expand", "find", "calculate", "compute", "what", "how", "much"}
    | {"many", "round", "convert", "write", "graph", "plot", "and", "the", "for"}
)


def _mostly_math(body: str) -> bool:
    words = [w for w in re.findall(r"[a-zA-Z]{3,}", body) if w.lower() not in _PROBLEM_WORDS]
    return _looks_like_math(body) and len(words) <= 1


def is_worksheet(source: str, text: str) -> bool:
    """Called a worksheet (file name, first line or a heading: worksheet, quiz, exercise, homework...) or laid out
    like one: "Problem N:" lines, "Answer:" lines, or numbered lines that are mostly bare math."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    titles = [re.sub(r"[_.\-]+", " ", Path(source).stem), *lines[:1], *(line for line in lines if line.startswith("#"))]
    if any(_WORKSHEET_HINT.search(title) for title in titles):
        return True
    if any(_PREFIXED.match(line) or _ANSWER_LINE.match(line) for line in lines):
        return True
    numbered = [found.group("text") for found in map(_NUMBERED.match, lines) if found]
    math = sum(1 for body in numbered if _mostly_math(_clean(body)))
    return math >= 2 and 2 * math >= len(numbered)


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


def pick_progressive(problems: Sequence[QuizProblem], length: int) -> list[QuizProblem]:
    """At most `length` of the (easiest-first) problems, spread evenly so the quiz still climbs every level."""
    if length <= 0 or len(problems) <= length:
        return list(problems)
    if length == 1:
        return [problems[0]]
    last = len(problems) - 1
    return [problems[round(i * last / (length - 1))] for i in range(length)]


# ---- answers kept away from the model ------------------------------------------------------------------------
# Math notes are worksheets and quiz banks: with an answer in a prompt the model reads it out, or copies it as the
# student's answer. For the math profile the ingestor strips a whole file's text before chunking and embedding it
# (strip_document_answers), so the store never holds one; strip_chunk_answers is the safety net at prompt time.

_ANSWER_WORD = r"(?:answers?|ans|solutions?|soluci[oó]n(?:es)?|respuestas?)"
_ANSWER_TAIL = re.compile(rf"[(\[]?\s*\b{_ANSWER_WORD}\b\s*[:=].*$", re.I)
# "Answer: x = 6", "Answers: 6, 8, 7", "Solución 3: x = 2", "**Respuesta:** x = 6".
_ANSWER_ROW = re.compile(rf"^\s*[-*•>]*\s*{_ANSWER_WORD}\s*(?:#?\s*\d{{1,3}}\s*)?[:=]\s*\S", re.I)
# "Answer:" or "Solución" on a line of its own: the answer, or a worked solution, is on the lines below it.
_ANSWER_HEAD = re.compile(r"^\s*[-*•>]*\s*(?:answer|ans|solution|soluci[oó]n|respuesta)\s*[:=]?\s*$", re.I)
# A section of answers that runs to a heading over something else: "Answer key", "Answers:", "Respuestas".
_KEY_LINE = re.compile(
    r"^\W*(?:(?:the\s+)?answer\s+key|answers|solutions|soluciones|respuestas|clave(?:\s+de\s+respuestas)?|"
    r"hoja\s+de\s+respuestas)\W*$",
    re.I,
)
# A heading named after answers: "## Answer key", "## Solución"; "## Answers to part A" only when answers follow.
_KEY_TITLE = re.compile(rf"^(?:the\s+)?(?:{_ANSWER_WORD}|answer\s+key|clave(?:\s+de\s+respuestas)?)\W*$", re.I)
_KEY_TITLE_START = re.compile(rf"^(?:the\s+)?(?:{_ANSWER_WORD}|answer\s+key|clave\s+de\s+respuestas)\b", re.I)
_LEVEL_LINE = re.compile(r"^\s*(?:level|nivel|difficulty|dificultad)\s*[:\-]?\s*\d+\b.{0,60}$", re.I)
_ANSWER_COLUMN = re.compile(rf"\b{_ANSWER_WORD}\b", re.I)
_TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?$")
# The ingestor writes a CSV row as "Row 3: problem is 3/4 - 1/2, answer is 1/4, topic is ..., level is 1."
_CSV_ROW = re.compile(r"^\s*Row\s+\d+\s*:")
_CSV_ANSWER = re.compile(rf"(?:,\s*)?\b(?:\w+[ _])?{_ANSWER_WORD}\s+is\s+.*?(?=,\s|\.\s|\.?\s*$)", re.I | re.M)


_VALUE = r"[-−]?\s*(?:\d+\s+\d+\s*/\s*\d+|\d+(?:[.,]\d+)?(?:\s*/\s*\d+)?|\.\d+)"
_SOLVED = rf"(?:[a-z]\s*(?:<=|>=|=|<|>|≤|≥)\s*{_VALUE}|\(\s*{_VALUE}\s*,\s*{_VALUE}\s*\)|{_VALUE})"
# "x = 6", "x = 5 or x = -3", "5/6", "(2, 3)", "x = 6 (subtract nine first)": a value, never a problem to solve.
_ANSWER_SHAPE = re.compile(
    rf"^{_SOLVED}(?:\s*(?:,|;|\bor\b|\band\b|\bo\b|\by\b)\s*{_SOLVED})*"
    r"(?:\s+[(\[]\s*[a-záéíóúñ]{2,}.*|\s+[-–—]\s+[a-záéíóúñ]{2,}.*|\s*:\s+[a-záéíóúñ]{2,}.*)?\.?$",
    re.I,
)
# "x = 6" or "6": never a problem. A bare fraction can be one ("12/16" under "Simplify each fraction").
_LETTER_VALUE = re.compile(rf"^[a-z]\s*(?:<=|>=|=|<|>|≤|≥)\s*{_VALUE}", re.I)
_PLAIN_NUMBER = re.compile(r"^[-−]?\s*\d+(?:[.,]\d+)?\.?$")
# "1) 6  2) 8  3) 7", "1. x = 6, 2. x = 8": a key written on one line.
_ANSWER_LIST = re.compile(rf"^(?:\s*[(\[]?\d{{1,3}}\s*[.):\]]\s*{_SOLVED}\s*[,;]?)+\s*$", re.I)
# "x + 9 = 15 → x = 6", "3/4 - 1/2 => 1/4": the value after an arrow.
_ARROW_ANSWER = re.compile(rf"\s*(?:→|⇒|=>|->)\s*{_SOLVED}\s*\.?\s*$", re.I)
_LINE_PREFIX = re.compile(
    r"^\s*(?:[-*•>]+\s*)?(?:(?:problem|question|exercise|problema|pregunta|ejercicio|q)\s*#?\s*)?"
    r"(?:\d{1,3}\s*[.):]\s*)?",
    re.I,
)
# A lone letter set to a number somewhere in the line ("10 - 3x = 4, so x = 2"); "3x = 15" is a step, not this.
_SOLVED_IN_LINE = re.compile(r"(?:^|[\s,;:(→>])[a-z]\s*=\s*[-−]?\s*\d", re.I)


def is_answer_column(name: str) -> bool:
    """A CSV or table column of answers: "answer", "Correct answer", "solution_steps", "Respuesta", "Solución"."""
    return bool(_ANSWER_COLUMN.search(re.sub(r"[_\-]+", " ", str(name or ""))))


def _answer_body(line: str) -> str:
    return _LINE_PREFIX.sub("", _clean(line), count=1).strip()


def _cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |" if any(cells) else ""


def _is_answer_line(line: str) -> bool:
    cleaned = _clean(line)
    if cleaned.startswith("|"):
        cells = _cells(cleaned)
        values = [cell for cell in cells if cell and not cell.isdigit()]
        return bool(values) and all(_ANSWER_SHAPE.match(cell) for cell in values)
    body = _answer_body(line)
    return bool(body) and bool(_ANSWER_SHAPE.match(body))


def _is_heading_line(cleaned: str) -> bool:
    return bool(
        _HEADING.match(cleaned) or _TOPIC.match(cleaned) or _LEVEL_LINE.match(cleaned) or _KEY_LINE.match(cleaned)
    )


def _key_heading(cleaned: str) -> bool:
    """ "Answer key" / "Answers:" as a line, or a heading named only after answers ("## Solución")."""
    heading = _HEADING.match(cleaned)
    return bool(_KEY_LINE.match(cleaned) or (heading and _KEY_TITLE.match(heading.group("text"))))


def _key_entry(line: str) -> bool:
    """A line that belongs in an answer key: a bare or worked answer, a list of them, a table row of them."""
    cleaned = _clean(line)
    if cleaned.startswith("|"):
        cells = [cell for cell in _cells(cleaned) if cell]
        return any(map(is_answer_column, cells)) or (len(cells) >= 2 and bool(_ANSWER_SHAPE.match(cells[-1])))
    return bool(
        _is_answer_line(line)
        or _ANSWER_LIST.match(cleaned)
        or _ANSWER_ROW.match(cleaned)
        or _SOLVED_IN_LINE.search(_answer_body(line))
    )


def _section_is_key(lines: Sequence[str], start: int) -> bool:
    """Whether the lines from `start` to the next heading are answers only (at least one)."""
    found = False
    for line in lines[start:]:
        cleaned = _clean(line)
        if _is_heading_line(cleaned):
            break
        if not cleaned or _TABLE_SEPARATOR.match(cleaned):
            continue
        if not _key_entry(line):
            return False
        found = True
    return found


def _opens_key_section(cleaned: str, lines: Sequence[str], i: int) -> bool:
    """A heading or a short title that starts with answers ("## Answers to part A", "Solutions for level 2") over a
    section of answers only. Prose that starts the same way ("Solutions must be simplified.") is not one."""
    heading = _HEADING.match(cleaned)
    title = heading.group("text") if heading else cleaned
    if not _KEY_TITLE_START.match(title) or (not heading and (len(title.split()) > 6 or title.endswith("."))):
        return False
    return _section_is_key(lines, i + 1)


def _under_label(state: str, line: str, cleaned: str, heading: bool) -> tuple[str, bool]:
    """(next state, whether the line goes) for a line under an "Answer:" / "Solución" label. States: "open" (just
    seen), "block" (dropping its worked solution), "solved" (its final value went); "" ends the label."""
    if not cleaned:
        return ("" if state == "solved" else state), False
    body = _answer_body(line)
    solved = _is_answer_line(line) or bool(_SOLVED_IN_LINE.search(body))
    math = solved or _looks_like_math(body)
    if heading:
        return "", False
    if (
        _NUMBERED.match(cleaned)
        and not solved
        and (state == "solved" or _mostly_math(body) or _PREFIXED.match(cleaned))
    ):
        return "", False  # the next problem; a numbered step with words ("1) Resta 3: 2x = 8") stays in the solution
    if not math and state in ("open", "solved"):
        return "", False  # "Answer: ____" left blank, or the text after the solution
    return ("solved" if solved or state == "solved" else "block"), True


def _strip_lines(lines: Sequence[str], *, in_key: bool = False) -> list[str | None]:
    """Each line as kept (an answer cell or tail cut off), or None when it is an answer. See strip_answers."""
    out: list[str | None] = []
    label = ""
    drop: list[int] | None = None  # the answer columns of the table being read; None outside a table
    numbers: set[int] = set()  # the problem numbers seen so far: an answer key repeats them
    for i, line in enumerate(lines):
        cleaned = _clean(line)
        heading = _is_heading_line(cleaned)
        if in_key:
            if not heading or _section_is_key(lines, i + 1):
                out.append(None)
                continue
            in_key = False  # a heading over something other than answers ends the key
        if label:
            label, gone = _under_label(label, line, cleaned, heading)
            if gone:
                out.append(None)
                continue
        if _key_heading(cleaned):
            in_key, label, drop = True, "", None
            out.append(None)
            continue
        if _ANSWER_ROW.match(cleaned):
            out.append(None)
            continue
        if _ANSWER_HEAD.match(cleaned):
            label = "open"
            out.append(None)
            continue
        if _opens_key_section(cleaned, lines, i):
            in_key, drop = True, None
            out.append(None)
            continue
        numbered = _NUMBERED.match(cleaned)
        if numbered:
            number, body = int(numbered.group("n")), _answer_body(line)
            bare = _is_answer_line(line) and (bool(_LETTER_VALUE.match(body) or _PLAIN_NUMBER.match(body)))
            # A number seen before on a problem: "1. 5/6" or "1. 15 - 9 = 6, so x = 6" there is problem 1's answer.
            again = number in numbers and (_is_answer_line(line) or bool(_SOLVED_IN_LINE.search(body)))
            if bare or again:
                out.append(None)
                continue
            if _looks_like_math(body):
                numbers.add(number)
        if cleaned.startswith("|"):
            cells = _cells(cleaned)
            if drop is None:
                drop = [j for j, cell in enumerate(cells) if is_answer_column(cell)]  # the header row
            if drop:
                line = _row([cell for j, cell in enumerate(cells) if j not in drop])
            elif not _TABLE_SEPARATOR.match(cleaned):
                filled = [j for j, cell in enumerate(cells) if cell]
                # "| x + 9 = 15 | x = 6 |" under a header that does not say "answer": the value goes.
                if (
                    len(filled) >= 2
                    and _ANSWER_SHAPE.match(cells[filled[-1]])
                    and any(_looks_like_math(cells[j]) and not _ANSWER_SHAPE.match(cells[j]) for j in filled[:-1])
                ):
                    cells[filled[-1]] = ""
                    line = _row(cells)
        else:
            drop = None
        if _CSV_ROW.match(line):
            line = _CSV_ANSWER.sub("", line)
        line = _ARROW_ANSWER.sub("", _ANSWER_TAIL.sub("", line)).rstrip()
        out.append(None if cleaned and not line.strip() else line)
    return out


def _opens_on_answers(text: str, *, keyed: bool) -> bool:
    """Whether a chunk starts inside an answer key: its first line is a bare answer ("7. x = 14"), or, for a source
    known to have an answer key, a numbered line that ends solved ("8. 10 - 3x = 4, so x = 2")."""
    first = next((line for line in text.splitlines() if _clean(line)), "")
    if not first or _HEADING.match(_clean(first)) or _KEY_LINE.match(_clean(first)) or _CSV_ROW.match(first):
        return False
    if _is_answer_line(first):
        return True
    return keyed and bool(_NUMBERED.match(_clean(first))) and bool(_SOLVED_IN_LINE.search(_answer_body(first)))


def _has_answer_key(text: str) -> bool:
    return any(_key_heading(_clean(line)) for line in text.splitlines())


def strip_chunk_answers(chunks: Sequence[tuple[str, str]]) -> list[str]:
    """Each retrieved (source, text) chunk's text without its answers, like strip_answers, and also when an answer
    key spans several chunks of one source: the chunks after the one with the "Answer key" heading carry no heading,
    only answers. Chunks arrive in relevance order, not document order, so each is judged on its own lines plus
    whether any retrieved chunk of its source has an answer key."""
    keyed = {source for source, text in chunks if _has_answer_key(text)}
    return [strip_answers(text, in_key=_opens_on_answers(text, keyed=source in keyed)) for source, text in chunks]


def chunks_with_answers(chunks: Sequence[tuple[str, str]]) -> list[bool]:
    """Per (source, text) chunk: whether it still holds an answer, i.e. strip_chunk_answers takes something out."""

    def lines(text: str) -> list[str]:
        return [line.rstrip() for line in text.splitlines()]

    return [lines(text) != lines(kept) for (_, text), kept in zip(chunks, strip_chunk_answers(chunks), strict=True)]


def strip_answers(text: str, *, in_key: bool = False) -> str:
    """The text without its answers: "Answer: x = 6" / "Respuesta: ..." / "Solución: ..." lines, "(answer: 6)",
    "→ x = 6", a worked solution under an "Answer:" label, an answer-key section ("Answer key", "Answers",
    "Respuestas", "## Solución") up to a heading over something else, numbered answers ("7. x = 14", or "1. 5/6"
    once problem 1 was seen), a table's answer column and a quiz-bank row's "answer is ...". `in_key`: the text
    starts inside an answer key (a later chunk of one)."""
    return "\n".join(line for line in _strip_lines(text.splitlines(), in_key=in_key) if line is not None)


def strip_document_answers(parts: Sequence[str]) -> list[str]:
    """A whole document's text, in order (its pages or slides, or the one text of a .md file), without its answers,
    for the ingestor to chunk: a key or a worked solution that runs over paragraphs, pages or slides goes all of it,
    and a numbered answer counts as one when its problem came anywhere before. One stripped text per part."""
    owners: list[int] = []
    lines: list[str] = []
    for n, part in enumerate(parts):
        for line in part.splitlines():
            owners.append(n)
            lines.append(line)
    out: list[list[str]] = [[] for _ in parts]
    for n, line in zip(owners, _strip_lines(lines), strict=True):
        if line is not None:
            out[n].append(line)
    return ["\n".join(kept) for kept in out]


# ---- solving and inventing problems ---------------------------------------------------------------------------

_MATH_TOKEN = re.compile(r"\s*(?:(?P<num>\d+(?:\.\d+)?|\.\d+)|(?P<var>[a-z])|(?P<op>[-+*/÷()=]))", re.I)
_INSTRUCTION = re.compile(
    r"^\s*(?:solve|simplify|evaluate|compute|calculate|work\s+out|find|what\s+is)(?:\s+for\s+[a-z])?\s*:?\s*", re.I
)
Linear = tuple[Fraction, Fraction]  # a*x + b


class _Unsolvable(ValueError):
    pass


class _LinearParser:
    """Recursive descent over + - * / ( ) with implicit products (2x, 3(x + 1)); values are linear in one letter."""

    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self.tokens = tokens
        self.pos = 0
        self.var: str | None = None

    def peek(self) -> tuple[str, str] | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self) -> tuple[str, str]:
        token = self.peek()
        if token is None:
            raise _Unsolvable("unexpected end")
        self.pos += 1
        return token

    def expr(self) -> Linear:
        value = self.term()
        while (token := self.peek()) is not None and token[1] in "+-":
            self.take()
            a, b = self.term()
            value = (value[0] + a, value[1] + b) if token[1] == "+" else (value[0] - a, value[1] - b)
        return value

    def term(self) -> Linear:
        value = self.fraction()
        while (token := self.peek()) is not None:
            if token[1] in ("*", "÷"):
                self.take()
                value = _times(value, self.fraction()) if token[1] == "*" else _over(value, self.fraction())
            elif token[0] in ("num", "var") or token[1] == "(":
                value = _times(value, self.fraction())
            else:
                break
        return value

    def fraction(self) -> Linear:
        # A written fraction bar binds tighter than × and ÷: "5/6 ÷ 1/3" is (5/6) ÷ (1/3).
        value = self.unary()
        while (token := self.peek()) is not None and token[1] == "/":
            self.take()
            value = _over(value, self.unary())
        return value

    def unary(self) -> Linear:
        token = self.peek()
        if token is not None and token[1] in "+-":
            self.take()
            a, b = self.unary()
            return (-a, -b) if token[1] == "-" else (a, b)
        return self.atom()

    def atom(self) -> Linear:
        kind, text = self.take()
        if kind == "num":
            return Fraction(0), Fraction(text)
        if kind == "var":
            if self.var not in (None, text):
                raise _Unsolvable("more than one letter")
            self.var = text
            return Fraction(1), Fraction(0)
        if text == "(":
            value = self.expr()
            if self.take()[1] != ")":
                raise _Unsolvable("unbalanced brackets")
            return value
        raise _Unsolvable(f"unexpected {text!r}")


def _times(p: Linear, q: Linear) -> Linear:
    if p[0] and q[0]:
        raise _Unsolvable("not linear")
    return p[0] * q[1] + q[0] * p[1], p[1] * q[1]


def _over(p: Linear, q: Linear) -> Linear:
    if q[0]:
        raise _Unsolvable("dividing by the letter")
    return p[0] / q[1], p[1] / q[1]


def _fmt(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _math_tokens(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos, text = 0, text.strip()
    while pos < len(text):
        found = _MATH_TOKEN.match(text, pos)
        if not found:
            raise _Unsolvable(f"not math at {pos}")
        kind = found.lastgroup or "op"
        tokens.append((kind, found.group(kind).lower()))
        pos = found.end()
    return tokens


def solve_problem(text: str | None) -> str | None:
    """The answer of a one-letter linear equation ("2(x + 5) = 3x + 4" -> "x = 6") or of plain arithmetic
    ("1/2 + 1/3" -> "5/6"), else None (quadratics, systems, word problems...)."""
    norm = str(text or "").replace("×", "*").replace("·", "*").replace("−", "-").replace("–", "-")
    norm = _INSTRUCTION.sub("", norm.strip()).rstrip(" ?.")
    norm = re.sub(r"\s*=\s*[?_\s]*$", "", norm)  # "12 ÷ 4 = ?" asks for the value
    if "=" not in norm:
        norm = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", norm)  # "6 x 7" is a product, not a letter
    try:
        parser = _LinearParser(_math_tokens(norm))
        left = parser.expr()
        right = None
        if (token := parser.peek()) is not None and token[1] == "=":
            parser.take()
            right = parser.expr()
        if parser.peek() is not None or not parser.tokens:
            return None
        if right is None:
            return None if parser.var else _fmt(left[1])
        if parser.var is None or left[0] == right[0]:
            return None
        return f"{parser.var} = {_fmt((right[1] - left[1]) / (left[0] - right[0]))}"
    except (_Unsolvable, ValueError, ZeroDivisionError):
        return None


def _x(coefficient: int) -> str:
    return "x" if coefficient == 1 else f"{coefficient}x"


def _add_or_subtract(r: random.Random) -> str:
    a, b = r.randint(2, 9), r.randint(2, 12)
    return f"x + {a} = {a + b}" if r.random() < 0.5 else f"x - {a} = {b}"


def _multiply_or_divide(r: random.Random) -> str:
    a, b = r.randint(2, 9), r.randint(2, 10)
    return f"{a}x = {a * b}" if r.random() < 0.5 else f"x/{a} = {b}"


def _two_step(r: random.Random) -> str:
    a, x = r.randint(2, 6), r.randint(2, 10)
    if r.random() < 0.5:
        b = r.randint(1, 9)
        return f"{a}x + {b} = {a * x + b}"
    b = r.randint(1, min(9, a * x - 1))
    return f"{a}x - {b} = {a * x - b}"


def _two_step_harder(r: random.Random) -> str:
    if r.random() < 0.5:
        a, q, b = r.randint(2, 5), r.randint(1, 6), r.randint(1, 9)
        return f"x/{a} + {b} = {q + b}"
    a, x, d = r.randint(2, 5), r.randint(1, 6), r.randint(1, 9)
    return f"{a * x + d} - {a}x = {d}"


def _distributive(r: random.Random) -> str:
    a, b = r.randint(2, 5), r.randint(1, 6)
    if r.random() < 0.5:
        return f"{a}(x + {b}) = {a * (r.randint(1, 9) + b)}"
    return f"{a}(x - {b}) = {a * r.randint(1, 9)}"


def _both_sides(r: random.Random) -> str:
    c, extra, x, b = r.randint(1, 4), r.randint(1, 4), r.randint(1, 9), r.randint(1, 9)
    return f"{_x(c + extra)} + {b} = {_x(c)} + {extra * x + b}"


_INVENTED: tuple[tuple[str, Callable[[random.Random], str]], ...] = (
    ("one-step equations", _add_or_subtract),
    ("one-step equations", _multiply_or_divide),
    ("two-step equations", _two_step),
    ("two-step equations", _two_step_harder),
    ("distributive property", _distributive),
    ("x on both sides", _both_sides),
)


_INT = re.compile(r"\d+")
_FRACTION_PAIR = re.compile(r"(\d+)\s*/\s*(\d+)")


def _new_fraction(match: re.Match[str], r: random.Random) -> str:
    den = max(2, int(match.group(2)) + r.choice((-1, 1, 2, 3)))
    return f"{r.randint(1, den - 1)}/{den}"


def _new_int(match: re.Match[str], r: random.Random) -> str:
    n = int(match.group())
    return str(max(2 if n >= 2 else 1, n + r.choice((-2, -1, 1, 2, 3))))


def _positive_whole(answer: str | None) -> bool:
    value = (answer or "").split("=")[-1].strip()
    return value.isdigit() and int(value) > 0


def similar_problem(problem: str, rng: random.Random | None = None) -> str | None:
    """A problem shaped like `problem` with new numbers ("3/4 - 1/2" -> "5/6 - 1/3", "3x + 5 = 20" -> "4x + 7 = 19"),
    for when the model reads the board's problem back instead of making a new one. None if code can't vary it."""
    r = rng or random.Random()
    if solve_problem(problem) is None:
        return None
    sides = problem.split("=")
    for _ in range(30):
        if "=" not in problem:
            candidate = _FRACTION_PAIR.sub(lambda m: _new_fraction(m, r), problem)
            if candidate == problem:
                candidate = _INT.sub(lambda m: _new_int(m, r), problem)
            value = solve_problem(candidate)
            fine = value is not None and not value.startswith("-") and value != "0"
        else:
            left = _INT.sub(lambda m: _new_int(m, r), sides[0])
            if len(sides) == 2 and _INT.fullmatch(sides[1].strip()):
                # One number on the right: pick the answer first, so it stays a whole number.
                value = solve_problem(left.replace("x", f"({r.randint(2, 9)})"))
                candidate = f"{left.rstrip()} = {value}" if value and value.isdigit() and value != "0" else ""
            else:
                candidate = "=".join([left, *(_INT.sub(lambda m: _new_int(m, r), s) for s in sides[1:])])
            fine = bool(candidate) and _positive_whole(solve_problem(candidate))
        if fine and candidate.replace(" ", "") != problem.replace(" ", ""):
            return candidate.strip()
    return None


def invent_problem(number: int, total: int, rng: random.Random | None = None) -> QuizProblem:
    """A made-up problem `number` of `total` for a quiz without worksheet problems, with its answer: one-step
    equations first, x on both sides last. A 3B model asked to invent one mostly echoes the student instead."""
    rng = rng or random.Random()
    index = min(len(_INVENTED) - 1, (max(1, number) - 1) * len(_INVENTED) // max(1, total))
    topic, make = _INVENTED[index]
    problem = make(rng)
    return QuizProblem(problem, solve_problem(problem), topic, index + 1, INVENTED_SOURCE)


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
