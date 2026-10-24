from __future__ import annotations

import dataclasses
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

import pytest

from biobench_sar.rag.quiz_bank import (
    DEFAULT_LEVEL,
    INVENTED_SOURCE,
    POSITION_LEVELS,
    QuizBank,
    QuizProblem,
    invent_problem,
    is_worksheet,
    parse_quiz_file,
    parse_worksheet_text,
    pick_progressive,
    similar_problem,
    solve_problem,
    strip_answers,
)

SEED = Path(__file__).resolve().parent.parent / "biobench_sar" / "knowledge" / "math"

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
        ("4 × 5 = ?", None, DEFAULT_LEVEL + 1, ""),
    ]


def test_seed_worksheet_levels_follow_its_headings() -> None:
    # The seed worksheet labels its sections with the quiz bank's level scale, so mixed quizzes climb evenly.
    problems = parse_worksheet_text((SEED / "worksheet_linear_equations.md").read_text(), "w.md")
    assert [p.level for p in problems] == [2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
    assert problems[0].problem == "x + 9 = 15" and problems[-1].problem == "2(x + 5) = 3x + 4"


def test_unlevelled_worksheet_gets_levels_by_position() -> None:
    lines = ["# Practice"] + [f"{n}. {n}x = {n * 2}" for n in range(1, 13)]
    levels = [p.level for p in parse_worksheet_text("\n".join(lines), "w.md")]
    assert levels == [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5], "spread over the levels by position"
    assert levels[-1] == POSITION_LEVELS


def test_seeded_quiz_is_progressive_and_short(tmp_path: Path) -> None:
    bank = QuizBank(tmp_path / "bank.json")
    for name in ("quiz_bank.csv", "worksheet_linear_equations.md", "tutoring_guide.md"):
        bank.replace_source(name, parse_quiz_file(SEED / name, name))
    everything = bank.problems()
    assert len(everything) == 43
    quiz = pick_progressive(everything, 10)
    assert len(quiz) == 10 and len({p.problem for p in quiz}) == 10
    assert [p.level for p in quiz] == sorted(p.level for p in quiz)
    assert quiz[0].level == 1 and quiz[-1].level == 5, "the subset climbs every level"
    assert "2(x + 5) = 3x + 4" not in [p.problem for p in quiz[:5]], "the hardest worksheet problem is not early"


def test_pick_progressive_edges() -> None:
    problems = [_p(str(n)) for n in range(5)]
    assert pick_progressive(problems, 10) == problems and pick_progressive(problems, 0) == problems
    assert pick_progressive(problems, 1) == problems[:1]
    assert [p.problem for p in pick_progressive(problems, 3)] == ["0", "2", "4"]


def test_answer_line_of_a_skipped_instruction_is_not_attached() -> None:
    text = "Worksheet\n1. 2x = 8\n2. Show your work\nAnswer: neatly\n3. x + 1 = 3\nAnswer: x = 2\n"
    assert [(p.problem, p.answer) for p in parse_worksheet_text(text, "w.md")] == [
        ("2x = 8", None),
        ("x + 1 = 3", "x = 2"),
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
        # neutral names, laid out like a worksheet
        ("algebra.md", "Solve each equation.\nProblem 1: 3x + 5 = 20\nAnswer: x = 5\nProblem 2: 2x = 8", True),
        ("unit3.txt", "1. 3x + 5 = 20\n2. 4x - 2 = 10", True),
        ("unit3.txt", "Notes\n1. 3x + 5 = 20\nAnswer: x = 5", True),
        ("tips.md", "1. If the student writes 3x = 12, ask what undoes times 3.\n2. Then ask about 2x = 6 too.", False),
        ("notes.md", "1. 3x + 5 = 20\n2. Read chapter two\n3. Bring the book\n4. Ask the teacher", False),
    ],
)
def test_is_worksheet(source: str, text: str, expected: bool) -> None:
    assert is_worksheet(source, text) is expected


def test_neutrally_named_worksheet_file_has_problems(tmp_path: Path) -> None:
    path = tmp_path / "algebra.md"
    path.write_text("Solve each equation.\nProblem 1: 3x + 5 = 20\nAnswer: x = 5\nProblem 2: 4x - 2 = 10\n")
    assert [(p.problem, p.answer) for p in parse_quiz_file(path, "algebra.md")] == [
        ("3x + 5 = 20", "x = 5"),
        ("4x - 2 = 10", None),
    ]
    assert parse_quiz_file(SEED / "tutoring_guide.md", "tutoring_guide.md") == []


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


# ---- answers kept from the model ---------------------------------------------


def _seed_chunks() -> list[str]:
    from biobench_sar.rag.ingestor import ingest_file

    return [c.text for name in ("quiz_bank.csv", "worksheet_linear_equations.md") for c in ingest_file(SEED / name)]


def test_strip_answers_on_the_real_seed_chunks() -> None:
    chunks = _seed_chunks()
    assert any("Answer: x = 4" in c for c in chunks) and any("answer is x = 5" in c for c in chunks)
    for text in map(strip_answers, chunks):
        assert not re.search(r"\banswer\s*(?:is|:)", text, re.I), text
        assert "solution is" not in text.lower()
    worksheet = strip_answers(chunks[-1])
    assert "5. 2x + 3 = 11\n" in worksheet, "the problems stay"
    assert not re.search(r"^\s*x = \d+\s*$", worksheet, re.M), "the answers go"
    csv_row = strip_answers(chunks[1])
    assert "problem is 3x + 5 = 20, topic is two-step equations, level is 3." in csv_row
    assert "x = 5" not in csv_row


def _keyed_seed_worksheet(tmp_path: Path) -> tuple[Path, list[QuizProblem]]:
    """The real seed worksheet laid out the other common way: problems first, every answer in an answer key at the
    end, one worked answer per paragraph, so the real ingestor splits the key over several chunks."""
    seed = (SEED / "worksheet_linear_equations.md").read_text(encoding="utf-8")
    problems = parse_worksheet_text(seed, "seed")
    assert len(problems) == 12 and all(p.answer for p in problems)
    lines = [line for line in seed.splitlines() if not line.strip().lower().startswith("answer:")]
    key = [
        f"{n}. {p.answer} (check it: put the value back into {p.problem} and both sides come out the same, "
        "which is how the student can see the step was right before moving on)"
        for n, p in enumerate(problems, 1)
    ]
    path = tmp_path / "worksheet_with_key.md"
    path.write_text("\n".join(lines) + "\n\n## Answer key\n\n" + "\n\n".join(key) + "\n", encoding="utf-8")
    return path, problems


def test_answer_key_spanning_several_real_chunks_is_stripped(tmp_path: Path) -> None:
    from biobench_sar.rag.ingestor import ingest_file
    from biobench_sar.rag.quiz_bank import strip_chunk_answers

    path, problems = _keyed_seed_worksheet(tmp_path)
    chunks = [(c.source, c.text) for c in ingest_file(path)]
    heading = [i for i, (_, text) in enumerate(chunks) if "## Answer key" in text]
    assert len(heading) == 1 and heading[0] < len(chunks) - 1, "the key really spans several chunks"
    later = [text for _, text in chunks[heading[0] + 1 :]]
    assert any(re.search(r"^\d+\. x = \d+ \(check", t, re.M) for t in later)
    assert any(re.search(r"^\d+\. x = \d+ \(check", strip_answers(t), re.M) for t in later), "the gap: per chunk"

    stripped = strip_chunk_answers(chunks)
    assert not any(re.search(r"^\d+\. x = -?\d+", t, re.M) for t in stripped), stripped
    assert "5. 2x + 3 = 11" in "\n".join(stripped), "the problems stay"
    parsed = parse_quiz_file(path, path.name)
    assert [p.problem for p in parsed] == [p.problem for p in problems], "the quiz bank still reads the whole file"
    assert all((got.answer or "").startswith(str(want.answer)) for got, want in zip(parsed, problems, strict=True))


@pytest.mark.parametrize("order", ["relevance", "continuation_only"])
def test_answer_key_chunks_are_stripped_in_any_retrieval_order(tmp_path: Path, order: str) -> None:
    from biobench_sar.rag.ingestor import ingest_file
    from biobench_sar.rag.quiz_bank import strip_chunk_answers

    path, _problems = _keyed_seed_worksheet(tmp_path)
    chunks = [(c.source, c.text) for c in ingest_file(path)]
    start = next(i for i, (_, text) in enumerate(chunks) if "## Answer key" in text)
    # Retrieval returns the most similar chunks first, and may skip the one with the heading altogether.
    picked = list(reversed(chunks)) if order == "relevance" else chunks[start + 1 :]
    for text in strip_chunk_answers(picked):
        assert not re.search(r"^\d+\. x = -?\d+", text, re.M), text


def test_strip_chunk_answers_leaves_the_real_seed_chunks_as_strip_answers_does() -> None:
    from biobench_sar.rag.ingestor import ingest_file
    from biobench_sar.rag.quiz_bank import strip_chunk_answers

    chunks = [
        (c.source, c.text)
        for name in ("quiz_bank.csv", "worksheet_linear_equations.md", "tutoring_guide.md")
        for c in ingest_file(SEED / name)
    ]
    stripped = strip_chunk_answers(chunks)
    assert stripped == [strip_answers(text) for _, text in chunks], "no notes lost to the answer-key check"
    for (source, _), text in zip(chunks, stripped, strict=True):
        if source != "tutoring_guide.md":  # prose about answers ("the answer is correct"), not an answer
            assert not re.search(r"\banswer\s*(?:is|:)", text, re.I), text


def test_strip_chunk_answers_keyed_source_drops_worked_answers() -> None:
    from biobench_sar.rag.quiz_bank import strip_chunk_answers

    key = ("ws.md", "# Worksheet\n1. x + 9 = 15\n2. 10 - 3x = 4\n## Answer key\n1. 15 - 9 = 6, so x = 6")
    rest = ("ws.md", "2. 10 - 3x = 4, so x = 2\n\n## Notes\nShow every step.")
    other = ("guide.md", "2. 10 - 3x = 4, so x = 2 is a common slip\n## Tips")
    out = strip_chunk_answers([rest, key, other])
    assert out[0] == "## Notes\nShow every step.", "a worked answer of a keyed source"
    assert out[1] == "# Worksheet\n1. x + 9 = 15\n2. 10 - 3x = 4"
    assert out[2] == other[1], "another file without a key keeps its prose"


def test_strip_answers_answer_label_on_its_own_line() -> None:
    assert strip_answers("1. x + 9 = 15\nAnswer:\nx = 6\n2. 2x = 8") == "1. x + 9 = 15\n2. 2x = 8"
    assert strip_answers("Respuesta:\n\n  x = 6\nSigue.") == "\nSigue."
    assert strip_answers("Answer:\nShow each step.") == "Show each step.", "only an answer after the label goes"


@pytest.mark.parametrize(
    ("line", "answer"),
    [
        ("7. x = 14", True),
        ("x = 5 or x = -3", True),
        ("5/6", True),
        ("(2, 3)", True),
        ("12. x = 6 (subtract nine)", True),
        ("| 3 | x = 7 |", True),
        ("Problem 3: x = 2", True),
        ("1. x + 9 = 15", False),
        ("3x = 15", False),
        ("1. 3/4 - 1/2", False),
        ("9. 4(x - 1) = 20", False),
        ("Row 1: problem is 1/4 + 2/4", False),
        ("- Same denominators: add the tops.", False),
    ],
)
def test_answer_line_shape(line: str, answer: bool) -> None:
    from biobench_sar.rag.quiz_bank import _is_answer_line

    assert _is_answer_line(line) is answer


def test_strip_answers_formats() -> None:
    text = (
        "# Quiz\n1. x + 9 = 15 (answer: 6)\n2. 2x = 8\n   **Answer:** x = 4\nNever read an answer out loud.\n"
        "## Answer key\n1. 6\n2. x = 4\n## Level 2\n3. y + 1 = 2\n"
    )
    assert (
        strip_answers(text)
        == "# Quiz\n1. x + 9 = 15\n2. 2x = 8\nNever read an answer out loud.\n## Level 2\n3. y + 1 = 2"
    )
    row = "Row 1: question is 5x = 35, solution is x = 7, difficulty is 2. Row 2: question is y + 2.5 = 8, answer is y = 5.5."
    assert strip_answers(row) == "Row 1: question is 5x = 35, difficulty is 2. Row 2: question is y + 2.5 = 8."
    assert strip_answers("If both sides match, the answer is correct.") == "If both sides match, the answer is correct."


# ---- solving and inventing ---------------------------------------------------


@pytest.mark.parametrize(
    ("problem", "answer"),
    [
        ("3x + 5 = 20", "x = 5"),
        ("2(x + 5) = 3x + 4", "x = 6"),
        ("x/2 - 4 = 3", "x = 14"),
        ("10 - 3x = 4", "x = 2"),
        ("-4x = 24", "x = -6"),
        ("y + 2.5 = 8", "y = 11/2"),
        ("Solve for x: 4(x - 1) = 20", "x = 6"),
        ("Solve 2x = 10", "x = 5"),
        ("1/4 + 2/4", "3/4"),
        ("5/6 ÷ 1/3", "5/2"),
        ("2/3 × 3/5", "2/5"),
        ("12 ÷ 4 = ?", "3"),
        ("What is 6 x 7?", "42"),
        ("x^2 + 5x + 6 = 0", None),
        ("x + y = 10 and x - y = 4", None),
        ("x * x = 4", None),
        ("2x = 2x", None),
        ("x / (x + 1) = 2", None),
        ("1/0", None),
        ("Show your work", None),
        ("", None),
        (None, None),
    ],
)
def test_solve_problem(problem: str | None, answer: str | None) -> None:
    assert solve_problem(problem) == answer


def test_solver_agrees_with_every_seed_answer() -> None:
    import csv

    from biobench_sar.llm.vlm_client import answer_value

    with (SEED / "quiz_bank.csv").open() as fh:
        for row in csv.DictReader(fh):
            solved = solve_problem(row["problem"])
            assert solved is None or answer_value(solved) == answer_value(row["answer"]), row
    for p in parse_quiz_file(SEED / "worksheet_linear_equations.md", "w.md"):
        assert solve_problem(p.problem) == p.answer, p


@pytest.mark.parametrize(
    "problem",
    [
        "3/4 - 1/2",
        "1/2 + 1/3",
        "5/6 ÷ 1/3",
        "3x + 5 = 20",
        "x/4 + 3 = 8",
        "10 - 3x = 4",
        "2(x + 5) = 3x + 4",
        "3y + 5 = 20",
        "2n - 4 = 10",
        "a/4 + 3 = 8",
    ],
)
def test_similar_problem_keeps_the_shape_with_new_numbers(problem: str) -> None:
    import random

    rng = random.Random(1)
    for _ in range(20):
        new = similar_problem(problem, rng)
        assert new and new.replace(" ", "") != problem.replace(" ", "")
        assert re.sub(r"\d+", "N", new) == re.sub(r"\d+", "N", problem), "same operations, same layout"
        value = solve_problem(new)
        assert value and not value.split("=")[-1].strip().startswith("-"), new
        if "=" in problem:
            assert value.split("=")[1].strip().isdigit(), "an equation keeps a whole-number answer"


@pytest.mark.parametrize("problem", ["x^2 = 49", "hello", "x + y = 3"])
def test_similar_problem_needs_something_code_can_solve(problem: str) -> None:
    assert similar_problem(problem) is None


def test_invented_problems_are_progressive_and_solvable() -> None:
    import random

    for seed in range(100):
        rng = random.Random(seed)
        quiz = [invent_problem(n, 10, rng) for n in range(1, 11)]
        assert [p.level for p in quiz] == [1, 1, 2, 2, 3, 4, 4, 5, 5, 6]
        for p in quiz:
            assert p.source == INVENTED_SOURCE and p.answer == solve_problem(p.problem)
            value = p.answer.split("=")[1].strip()  # type: ignore[union-attr]
            assert value.isdigit() and int(value) > 0, p  # a whole positive number for a middle-school quiz
    assert invent_problem(1, 1).level == 1 and invent_problem(5, 1).level == 6
