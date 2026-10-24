from __future__ import annotations

import dataclasses
import json
import os
import threading
from pathlib import Path
from typing import Any

import pytest

from biobench_sar.rag.quiz_bank import (
    DEFAULT_LEVEL,
    QuizBank,
    QuizProblem,
    is_worksheet,
    parse_quiz_file,
    parse_worksheet_text,
)

WORKSHEET = """# Equations worksheet
Show your work for every problem.

## Level 1 - One-step equations
1. x + 3 = 7
   Answer: x = 4
2. Solve 2x = 10 (Answer: x = 5)
3. Write your name at the top ______

## Level 2: Two-step equations
Problem 4: 3x + 5 = 20
Q5. 2.5 + 1.5 = ______

## Answer key
4. x = 5
5. 4
"""


def _p(problem: str, answer: str | None = None, topic: str = "", level: int = 1, source: str = "w.md") -> QuizProblem:
    return QuizProblem(problem, answer, topic, level, source)


def test_quiz_problem_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _p("1 + 1").problem = "2"  # type: ignore[misc]


def test_worksheet_text_numbered_problems_answers_levels_and_key() -> None:
    problems = parse_worksheet_text(WORKSHEET, "w.md")
    assert [(p.problem, p.answer, p.level) for p in problems] == [
        ("x + 3 = 7", "x = 4", 1),
        ("Solve 2x = 10", "x = 5", 1),
        ("3x + 5 = 20", "x = 5", 2),
        ("2.5 + 1.5 =", "4", 2),
    ]
    assert problems[0].topic == "One-step equations" and problems[2].topic == "Two-step equations"
    assert {p.source for p in problems} == {"w.md"}


def test_worksheet_numbered_instructions_without_math_are_skipped() -> None:
    text = "# Homework\n1. Read the chapter.\n2. Use 2 colours.\n3) 12 ÷ 4 = ?\n"
    assert [p.problem for p in parse_worksheet_text(text, "h.md")] == ["12 ÷ 4 = ?"]


def test_worksheet_decimal_is_not_a_problem_number() -> None:
    assert [p.problem for p in parse_worksheet_text("Practice\n0.5 + 0.25 = ?\n1. 0.5 + 0.25 = ?", "p.md")] == [
        "0.5 + 0.25 = ?"
    ]


def test_worksheet_problems_without_answers() -> None:
    problems = parse_worksheet_text("Quiz\n1. 7 - 3 = ?\n2. 4 × 5 = ?", "q.txt")
    assert [(p.problem, p.answer, p.level, p.topic) for p in problems] == [
        ("7 - 3 = ?", None, DEFAULT_LEVEL, ""),
        ("4 × 5 = ?", None, DEFAULT_LEVEL, ""),
    ]


def test_heading_after_answer_key_starts_new_problems() -> None:
    text = "Worksheet\n1. 1 + 1 = ?\nAnswers\n1. 2\n## Level 3\n2. 9 - 4 = ?\n"
    assert [(p.problem, p.answer, p.level) for p in parse_worksheet_text(text, "w.md")] == [
        ("1 + 1 = ?", "2", 1),
        ("9 - 4 = ?", None, 3),
    ]


def test_spanish_worksheet() -> None:
    text = "# Hoja de ejercicios\nProblema 1: 2x + 1 = 9\nRespuesta: x = 4\n"
    assert [(p.problem, p.answer) for p in parse_worksheet_text(text, "e.md")] == [("2x + 1 = 9", "x = 4")]


@pytest.mark.parametrize(
    ("source", "text", "expected"),
    [
        ("fractions_worksheet.pdf", "Name: ____", True),
        ("unit-test.md", "", True),
        ("notes.md", "Chapter 3 quiz\n1. 1 + 1", True),
        ("notes.md", "Intro\n## Practice problems\n1. 1 + 1", True),
        ("tutoring_guide.md", "# Math tutoring guide\n1. Ask what 3x = 12 means.", False),
        ("protocol.md", "Lab protocol\n1. Add 5 mL = one tube", False),
    ],
)
def test_is_worksheet(source: str, text: str, expected: bool) -> None:
    assert is_worksheet(source, text) is expected


def test_parse_markdown_worksheet_file(tmp_path: Path) -> None:
    path = tmp_path / "9f2c.md"  # an upload is saved under a random name; the source keeps the real one
    path.write_text(WORKSHEET)
    problems = parse_quiz_file(path, "equations.md")
    assert len(problems) == 4 and {p.source for p in problems} == {"equations.md"}


def test_parse_guide_file_has_no_problems(tmp_path: Path) -> None:
    path = tmp_path / "tutoring_guide.md"
    path.write_text("# Tutoring guide\n1. If the student writes 3x = 12, ask what undoes times 3.\n")
    assert parse_quiz_file(path, path.name) == []


def test_parse_csv_quiz_bank(tmp_path: Path) -> None:
    path = tmp_path / "bank.csv"
    path.write_text(
        "﻿Problem , Answer,Topic,Level\n"
        '3x + 5 = 20,x = 5,equations,2\n"1/2 + 1/4",3/4,fractions,1\n,,,\n7 - 2,,,\n9 × 3,27,,hard\n'
    )
    assert parse_quiz_file(path, "bank.csv") == [
        _p("3x + 5 = 20", "x = 5", "equations", 2, "bank.csv"),
        _p("1/2 + 1/4", "3/4", "fractions", 1, "bank.csv"),
        _p("7 - 2", None, "", 1, "bank.csv"),
        _p("9 × 3", "27", "", 1, "bank.csv"),
    ]


def test_parse_csv_question_and_difficulty_columns(tmp_path: Path) -> None:
    path = tmp_path / "q.csv"
    path.write_text("question,solution,difficulty\nWhat is 6 x 7?,42,3.0\n")
    assert parse_quiz_file(path, "q.csv") == [_p("What is 6 x 7?", "42", "", 3, "q.csv")]


def test_parse_csv_without_problem_column_has_none(tmp_path: Path) -> None:
    path = tmp_path / "sites.csv"
    path.write_text("site,turbidity\nA,12\n")
    assert parse_quiz_file(path, "sites.csv") == []


def test_parse_docx_worksheet(tmp_path: Path) -> None:
    import docx

    document = docx.Document()
    for line in ("Fractions worksheet", "1. 1/2 + 1/3 = ?", "Answer: 5/6"):
        document.add_paragraph(line)
    path = tmp_path / "w.docx"
    document.save(str(path))
    assert parse_quiz_file(path, "w.docx") == [_p("1/2 + 1/3 = ?", "5/6", "", 1, "w.docx")]


def test_parse_pdf_worksheet(tmp_path: Path) -> None:
    import pymupdf

    path = tmp_path / "w.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), "Practice quiz\n1. 4x = 12\nAnswer: x = 3")
        doc.save(str(path))
    assert [(p.problem, p.answer) for p in parse_quiz_file(path, "w.pdf")] == [("4x = 12", "x = 3")]


@pytest.mark.parametrize("name", ["slides.pptx", "photo.png", "notes"])
def test_parse_other_types_has_none(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.write_bytes(b"\x00\x01")
    assert parse_quiz_file(path, name) == []


def test_parse_never_raises(tmp_path: Path) -> None:
    broken = tmp_path / "broken.docx"
    broken.write_bytes(b"not a zip")
    assert parse_quiz_file(broken, "broken.docx") == []
    assert parse_quiz_file(tmp_path / "missing.md", "missing.md") == []


# ---- QuizBank -----------------------------------------------------------------


@pytest.fixture
def bank_path(tmp_path: Path) -> Path:
    return tmp_path / "state" / "quiz_bank.json"


def test_bank_starts_empty_without_writing(bank_path: Path) -> None:
    bank = QuizBank(bank_path)
    assert bank.problems() == [] and bank.sources() == []
    assert not bank_path.exists()


def test_bank_orders_by_level_then_file_order(bank_path: Path) -> None:
    bank = QuizBank(bank_path)
    bank.replace_source("a.md", [_p("a1", level=2), _p("a2", level=1), _p("a3", level=2)])
    bank.replace_source("b.csv", [_p("b1", level=1), _p("b2", level=3)])
    assert [p.problem for p in bank.problems()] == ["a2", "b1", "a1", "a3", "b2"]
    assert {p.source for p in bank.problems(["b.csv"])} == {"b.csv"}, "sources are stamped with the file name"


def test_bank_filters_by_sources(bank_path: Path) -> None:
    bank = QuizBank(bank_path)
    bank.replace_source("a.md", [_p("a1")])
    bank.replace_source("b.md", [_p("b1")])
    assert [p.problem for p in bank.problems(["b.md", "missing.md"])] == ["b1"]
    assert bank.problems([]) == []
    assert [p.problem for p in bank.problems(None)] == ["a1", "b1"]


def test_bank_replace_remove_and_persist(bank_path: Path) -> None:
    bank = QuizBank(bank_path)
    bank.replace_source("a.md", [_p("old")])
    bank.replace_source("b.md", [_p("b1", "2", "sums", 2)])
    bank.replace_source("a.md", [_p("new")])
    assert bank.sources() == ["b.md", "a.md"], "a re-upload counts as the newest file"
    reloaded = QuizBank(bank_path)
    assert reloaded.problems() == [_p("new", source="a.md"), _p("b1", "2", "sums", 2, "b.md")]
    bank.replace_source("a.md", [])
    bank.remove_source("b.md")
    bank.remove_source("missing.md")
    assert QuizBank(bank_path).sources() == []
    assert json.loads(bank_path.read_text()) == {"sources": {}}


def test_bank_noop_changes_do_not_write(bank_path: Path) -> None:
    bank = QuizBank(bank_path)
    bank.replace_source("a.md", [])
    bank.remove_source("a.md")
    assert not bank_path.exists()


@pytest.mark.parametrize(
    "content",
    ["{bad", "[]", "null", '{"sources": []}', '{"sources": {"a.md": "x"}}', '{"sources": {"a.md": [{"answer": 1}]}}'],
)
def test_bank_corrupt_file_starts_empty(bank_path: Path, content: str) -> None:
    bank_path.parent.mkdir(parents=True)
    bank_path.write_text(content)
    assert QuizBank(bank_path).problems() == []


def test_bank_loads_loose_values(bank_path: Path) -> None:
    bank_path.parent.mkdir(parents=True)
    bank_path.write_text(json.dumps({"sources": {"a.md": [{"problem": "1 + 1", "answer": 2, "level": "x"}, 5]}}))
    assert QuizBank(bank_path).problems() == [_p("1 + 1", "2", "", DEFAULT_LEVEL, "a.md")]


def test_bank_write_is_atomic(bank_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bank = QuizBank(bank_path)
    bank.replace_source("a.md", [_p("1 + 1")])
    before = bank_path.read_text()

    def fail(src: Any, dst: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", fail)
    bank.replace_source("b.md", [_p("2 + 2")])
    assert bank_path.read_text() == before
    assert [p.name for p in bank_path.parent.iterdir()] == [bank_path.name], "temp file cleaned up"
    assert len(bank.problems()) == 2, "still applies in memory"


def test_bank_concurrent_updates(bank_path: Path) -> None:
    bank = QuizBank(bank_path)

    def worker(n: int) -> None:
        for i in range(20):
            bank.replace_source(f"f{n}-{i}.md", [_p(f"{n}-{i}")])
            bank.problems()

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(QuizBank(bank_path).problems()) == 80
