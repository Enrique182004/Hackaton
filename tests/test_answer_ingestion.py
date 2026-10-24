"""Math answers never reach the vector store: stripped from the whole document at ingestion (upload, seed, migration).

Every layout goes through the real ingestor and a real VectorStore (chromadb) with a fake embedder, and the test
asserts that no answer can be retrieved, while the problems still can and the quiz bank still has the answers.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import pytest

from biobench_sar import app as app_module
from biobench_sar.config import PACKAGE_DIR, Settings
from biobench_sar.rag.ingestor import ingest_file
from biobench_sar.rag.quiz_bank import QuizBank, chunks_with_answers, parse_quiz_file, strip_document_answers
from biobench_sar.rag.vector_store import VectorStore

SEED = PACKAGE_DIR / "knowledge" / "math"
DIM = 256


def fake_embed(texts: list[str]) -> list[list[float]]:
    out = []
    for t in texts:
        v = [0.0] * DIM
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        out.append([x / norm for x in v])
    return out


def x_is(value: str) -> str:
    """ "x = 6" as written in any spacing; not "3x = 6" or "x = 60"."""
    return rf"(?<![\w/])x\s*=\s*{re.escape(value)}(?![\d/.])"


def frac(value: str) -> str:
    return rf"(?<![\d/]){re.escape(value)}(?![\d/])"


class Layout(NamedTuple):
    name: str  # the file name: its extension picks the parser
    write: Callable[[Path], None]
    answers: list[str]  # regexes: none may be retrievable
    problems: list[str]  # texts that must still be retrievable
    quiz: dict[str, str]  # problem -> answer the quiz bank must still read from the original file


def _text(content: str) -> Callable[[Path], None]:
    return lambda path: path.write_text(content, encoding="utf-8")


ONE_STEP = ["x + 9 = 15", "x - 6 = 2", "8x = 56", "x/5 = 4"]
ONE_STEP_X = ["6", "8", "7", "20"]
EXPLAIN = (
    "subtract or add the same number on both sides, divide when x is multiplied, and put the value back into the "
    "equation to check that both sides come out the same before moving on to the next problem on the sheet"
)


def _answers_section() -> str:
    problems = "\n\n".join(f"{n}. {p}" for n, p in enumerate(ONE_STEP, 1))
    # One worked answer per paragraph, long enough that the key is chunked over several chunks.
    key = "\n\n".join(f"{n}. x = {v} ({EXPLAIN})" for n, v in enumerate(ONE_STEP_X, 1))
    notes = "## Notes for the tutor\n\nPraise every correct step and point at the term to move next."
    return f"# Worksheet: one-step equations\n\n{problems}\n\n## Answers\n\n{key}\n\n{notes}\n"


def _pdf_key_over_pages(path: Path) -> None:
    import pymupdf

    doc = pymupdf.open()
    first = "Worksheet: one-step equations\n" + "\n".join(f"{n}. {p}" for n, p in enumerate(ONE_STEP, 1))
    doc.new_page().insert_text((72, 72), first + "\nAnswer key\n1. x = 6\n2. x = 8")
    doc.new_page().insert_text((72, 72), "3. x = 7\n4. x = 20")  # the key goes on over the page break
    doc.save(str(path))


LAYOUTS = [
    Layout(
        "inline_answers.md",
        _text(
            "# Worksheet: one-step equations\n\n1. x + 9 = 15\nAnswer: x = 6\n\n2. x - 6 = 2\n**Answer:** x = 8\n\n"
            "3. 8x = 56 (answer: x = 7)\n\n4. x/5 = 4 → x = 20\n\n5. 2x + 3 = 11\nAnswer:\nx = 4\n\n"
            "Answers: x = 6, x = 8, x = 7, x = 20, x = 4\n"
        ),
        [x_is(v) for v in [*ONE_STEP_X, "4"]],
        [*ONE_STEP, "2x + 3 = 11"],
        {"x + 9 = 15": "x = 6", "x - 6 = 2": "x = 8", "8x = 56": "x = 7"},
    ),
    Layout(
        "answers_at_the_end.md",
        _text(_answers_section()),
        [x_is(v) for v in ONE_STEP_X],
        [*ONE_STEP, "Praise every correct step"],
        {p: f"x = {v}" for p, v in zip(ONE_STEP, ONE_STEP_X, strict=True)},
    ),
    Layout(
        "key_as_plain_lines.txt",
        _text(
            "Practice: one-step equations\n\n"
            + "\n".join(f"{n}. {p}" for n, p in enumerate(ONE_STEP, 1))
            + "\n\n"
            + "\n".join(f"{n}. x = {v}" for n, v in enumerate(ONE_STEP_X, 1))
            + "\n"
        ),
        [x_is(v) for v in ONE_STEP_X],
        ONE_STEP,
        {},
    ),
    Layout(
        "fractions_practice.txt",
        _text(
            "Practice: fractions\n\n1. 3/4 - 1/2\n2. 1/8 + 1/8\n3. 2/5 + 3/10\n4. 5/6 ÷ 1/3\n\n"
            "1. 1/4\n2. 1/4\n3. 7/10\n4. 5/2\n"
        ),
        [frac("1/4"), frac("7/10"), frac("5/2")],
        ["3/4 - 1/2", "1/8 + 1/8", "2/5 + 3/10", "5/6 ÷ 1/3"],
        {},
    ),
    Layout(
        "quiz_table.md",
        _text(
            "# Quiz: equations\n\n| # | Problem | Answer |\n|---|---------|--------|\n"
            + "".join(
                f"| {n} | {p} | x = {v} |\n" for n, (p, v) in enumerate(zip(ONE_STEP, ONE_STEP_X, strict=True), 1)
            )
            + "\nMore practice:\n\n| Problem | Check |\n|---|---|\n| 2x + 3 = 11 | x = 4 |\n| 6x - 5 = 25 | x = 5 |\n"
        ),
        [x_is(v) for v in [*ONE_STEP_X, "4", "5"]],
        [*ONE_STEP, "2x + 3 = 11", "6x - 5 = 25", "| # | Problem |"],
        {},
    ),
    Layout(
        "my_quiz_bank.csv",
        _text(
            "problem,answer,solution steps,topic,level\n"
            + "".join(
                f"{p},x = {v},undo it on both sides so x = {v},one-step equations,1\n"
                for p, v in zip(ONE_STEP, ONE_STEP_X, strict=True)
            )
        ),
        [x_is(v) for v in ONE_STEP_X],
        ONE_STEP,
        {p: f"x = {v}" for p, v in zip(ONE_STEP, ONE_STEP_X, strict=True)},
    ),
    Layout(
        "banco_de_problemas.csv",
        _text(
            "problema,respuesta,tema\n"
            + "".join(f"{p},x = {v},ecuaciones\n" for p, v in zip(ONE_STEP, ONE_STEP_X, strict=True))
        ),
        [x_is(v) for v in ONE_STEP_X],
        ONE_STEP,
        {},
    ),
    Layout(
        "ejercicios.md",
        _text(
            "# Ejercicios: ecuaciones\n\n1. x + 9 = 15\n\n2. x - 6 = 2\n\n3. 2x + 3 = 11\nSolución:\n"
            "Resta 3 en ambos lados: 2x = 8\nDivide entre 2: x = 4\n\n4. 8x = 56\nRespuesta: x = 7\n\n"
            "## Respuestas\n\n1. x = 6\n\n2. x = 8\n\n3. x = 4\n"
        ),
        [x_is(v) for v in ("6", "8", "4", "7")] + [r"2x\s*=\s*8"],
        ["x + 9 = 15", "x - 6 = 2", "2x + 3 = 11", "8x = 56"],
        {"x + 9 = 15": "x = 6", "x - 6 = 2": "x = 8", "8x = 56": "x = 7"},
    ),
    Layout(
        "solutions_per_problem.md",
        _text(
            "# Homework\n\n### Problem 1\n\nx + 9 = 15\n\n### Solution\n\nSubtract nine from both sides.\n\nx = 6\n\n"
            "### Problem 2\n\nx - 6 = 2\n\n### Solution\n\nx = 8\n"
        ),
        [x_is("6"), x_is("8"), r"Subtract nine"],
        ["x + 9 = 15", "x - 6 = 2"],
        {},
    ),
    Layout(
        "key_with_sections.md",
        _text(
            "# Worksheet\n\n## Level 2\n\n1. x + 9 = 15\n2. x - 6 = 2\n\n## Level 3\n\n3. 2x + 3 = 11\n\n"
            "## Answer key\n\n### Level 2\n\n| # | Answer |\n|---|---|\n| 1 | x = 6 |\n| 2 | 8 |\n\n"
            "### Level 3\n\n3. 11 - 3 = 8, so x = 4\n\n## Notes\n\nKeep every step on the board.\n"
        ),
        [x_is("6"), x_is("4"), r"\|\s*8\s*\|"],
        ["x + 9 = 15", "x - 6 = 2", "2x + 3 = 11", "Keep every step on the board."],
        {},
    ),
    Layout(
        "key_over_two_pages.pdf",
        _pdf_key_over_pages,
        [x_is(v) for v in ONE_STEP_X],
        ONE_STEP,
        {p: f"x = {v}" for p, v in zip(ONE_STEP, ONE_STEP_X, strict=True)},
    ),
]


def _all_chunks(store: VectorStore, query: str) -> list[str]:
    return [chunk.text for chunk in store.query(query, k=max(1, store.count()))]


def _leaks(texts: list[str], answers: list[str]) -> list[tuple[str, str]]:
    return [(pattern, text) for text in texts for pattern in answers if re.search(pattern, text, re.I)]


@pytest.mark.parametrize("layout", LAYOUTS, ids=[layout.name for layout in LAYOUTS])
def test_no_answer_is_retrievable_after_ingestion(tmp_path: Path, layout: Layout) -> None:
    path = tmp_path / layout.name
    layout.write(path)
    assert _leaks([c.text for c in ingest_file(path)], layout.answers), "the layout really carries its answers"

    store = VectorStore(tmp_path / "chroma", embedder=fake_embed)
    store.replace_source(layout.name, ingest_file(path, strip_answers=True))
    assert store.count() > 0
    # k covers every stored chunk, so whatever the query, all of them come back: none may hold an answer.
    for query in ("answer key", "answers respuestas solución", "x = 6", "what is x", *layout.problems):
        retrieved = _all_chunks(store, query)
        assert not _leaks(retrieved, layout.answers), (query, retrieved)
    stored = "\n".join(_all_chunks(store, "problem"))
    for problem in layout.problems:
        assert problem in stored, f"{problem!r} is no answer and stays"
    assert not any(chunks_with_answers([(layout.name, text) for text in _all_chunks(store, "x")])), (
        "nothing left for the prompt-time safety net or the startup migration to find"
    )
    found = {p.problem: p.answer for p in parse_quiz_file(path, layout.name)}
    for problem, answer in layout.quiz.items():
        assert (found.get(problem) or "").startswith(answer), "the quiz bank still reads the original file"


def test_the_real_seed_files_lose_their_answers_and_keep_their_notes() -> None:
    worksheet = ingest_file(SEED / "worksheet_linear_equations.md", strip_answers=True)
    text = "\n".join(c.text for c in worksheet)
    assert "Answer:" not in text and "12. 2(x + 5) = 3x + 4" in text and "## Level 4" in text
    rows = "\n".join(c.text for c in ingest_file(SEED / "quiz_bank.csv", strip_answers=True))
    assert "answer is" not in rows and "problem is 3x + 5 = 20, topic is two-step equations, level is 3." in rows
    guide = SEED / "tutoring_guide.md"
    assert [c.text for c in ingest_file(guide, strip_answers=True)] == [c.text for c in ingest_file(guide)], (
        "a guide without answers is stored as it is: its tables and tips are not answers"
    )


def test_strip_document_answers_keeps_one_text_per_page() -> None:
    pages = ["1. x + 9 = 15\nAnswer key\n1. x = 6", "2. x = 8\n## Next steps\nPraise the student.", ""]
    assert strip_document_answers(pages) == ["1. x + 9 = 15", "## Next steps\nPraise the student.", ""]


# ---- upload and seed ------------------------------------------------------------------------------------------


@pytest.fixture
def real_store(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "vectors", embedder=fake_embed)


def _app(settings: Settings, fakes: Any, store: VectorStore, templates_dir: Path) -> Any:
    fakes.store = store
    flask_app, _ = app_module.create_app(settings, fakes, template_folder=str(templates_dir))
    flask_app.testing = True
    return flask_app


def _upload(client: Any, name: str, content: str) -> dict[str, Any]:
    data = {"files[]": [(io.BytesIO(content.encode()), name)]}
    response = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    return response.get_json()["files"][0]


def test_math_upload_stores_no_answers_and_the_quiz_keeps_them(
    settings: Settings, fakes: Any, real_store: VectorStore, templates_dir: Path
) -> None:
    flask_app = _app(settings, fakes, real_store, templates_dir)
    client = flask_app.test_client()
    assert _upload(client, "answers_at_the_end.md", _answers_section())["chunks"] > 0
    for query in ("x = 6", "answers", "x + 9 = 15"):
        results = client.get("/api/search", query_string={"q": query, "k": 10}).get_json()["results"]
        assert results and not _leaks([r["text"] for r in results], [x_is(v) for v in ONE_STEP_X]), results
    bank = flask_app.config["SAR_ASSISTANT"].quiz_bank
    assert {p.problem: p.answer for p in bank.problems()}["x + 9 = 15"].startswith("x = 6")


def test_biology_upload_keeps_its_text(
    settings: Settings, fakes: Any, real_store: VectorStore, templates_dir: Path
) -> None:
    flask_app = _app(dataclasses.replace(settings, profile="biology"), fakes, real_store, templates_dir)
    _upload(flask_app.test_client(), "lab_quiz.md", "# Lab quiz\n\n1. Which agar?\nAnswer: non-nutrient agar\n")
    assert any("Answer: non-nutrient agar" in text for text in _all_chunks(real_store, "agar"))


def test_a_file_of_only_answers_stores_nothing_and_says_why(
    settings: Settings, fakes: Any, real_store: VectorStore, templates_dir: Path
) -> None:
    client = _app(settings, fakes, real_store, templates_dir).test_client()
    result = _upload(client, "key.md", "## Answer key\n\n1. x = 6\n2. x = 8\n")
    assert result["chunks"] == 0 and result["error"] == app_module.ONLY_ANSWERS_ERROR
    assert real_store.count() == 0
    assert _upload(client, "blank.md", "  \n")["error"] == app_module.NO_TEXT_ERROR
    assert _upload(client, "key.csv", "answer,solution\nx = 6,x = 6\n")["error"] == app_module.ONLY_ANSWERS_ERROR


def test_seeding_math_stores_no_answers(tmp_path: Path, real_store: VectorStore) -> None:
    bank = QuizBank(tmp_path / "bank.json")
    assert app_module.seed_knowledge(real_store, SEED, quiz_bank=bank, strip_answers=True) > 0
    texts = _all_chunks(real_store, "answer")
    assert not any(chunks_with_answers([("seed", text) for text in texts]))
    assert not any(re.search(r"Answer: x|answer is x", text) for text in texts)
    answers = {p.problem: p.answer for p in bank.problems()}
    assert answers["2x + 3 = 11"] == "x = 4" and answers["3x + 5 = 20"] == "x = 5", "the quiz bank keeps them"


def test_seed_app_knowledge_strips_for_math_only(
    settings: Settings, fakes: Any, real_store: VectorStore, templates_dir: Path, tmp_path: Path
) -> None:
    flask_app = _app(settings, fakes, real_store, templates_dir)
    app_module.seed_app_knowledge(flask_app, settings)
    assert not any("Answer:" in text for text in _all_chunks(real_store, "answer"))
    bio_store = VectorStore(tmp_path / "bio", embedder=fake_embed)
    folder = tmp_path / "biology"
    folder.mkdir()
    (folder / "quiz.md").write_text("# Quiz\n\n1. Which agar?\nAnswer: non-nutrient agar\n")
    biology = dataclasses.replace(settings, profile="biology", chroma_dir=tmp_path / "bio_chroma")
    bio_app = _app(biology, fakes, bio_store, templates_dir)
    app_module.seed_app_knowledge(bio_app, biology, folder)
    assert any("Answer: non-nutrient agar" in text for text in _all_chunks(bio_store, "agar"))


# ---- startup migration of stores filled before stripping -----------------------------------------------------


def _seed_copy(tmp_path: Path) -> Path:
    folder = tmp_path / "seed_math"
    shutil.copytree(SEED, folder)
    return folder


def _store_old_way(store: VectorStore, path: Path, name: str | None = None) -> None:
    """What the app stored before answers were stripped at ingestion."""
    name = name or path.name
    store.replace_source(name, [dataclasses.replace(c, source=name) for c in ingest_file(path)])


def _user_worksheet(tmp_path: Path) -> Path:
    path = tmp_path / "upload" / "my_worksheet.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text("# My worksheet\n\n1. x + 2 = 9\nAnswer: x = 7\n", encoding="utf-8")
    return path


def test_migration_reingests_seed_files_and_flags_uploads(tmp_path: Path, real_store: VectorStore) -> None:
    seed = _seed_copy(tmp_path)
    for name in ("worksheet_linear_equations.md", "quiz_bank.csv", "tutoring_guide.md"):
        _store_old_way(real_store, seed / name)
    _store_old_way(real_store, _user_worksheet(tmp_path))
    guide_before = sorted(_texts_of(real_store, "tutoring_guide.md"))
    bank = QuizBank(tmp_path / "bank.json")

    reingested, flagged = app_module.migrate_answer_keys(real_store, seed, quiz_bank=bank)
    assert sorted(reingested) == ["quiz_bank.csv", "worksheet_linear_equations.md"]
    assert flagged == ["my_worksheet.md"], "an upload's original is gone: it can only be flagged"
    texts = {d["source"]: _texts_of(real_store, d["source"]) for d in real_store.list_sources()}
    for name in reingested:
        assert not any(chunks_with_answers([(name, text) for text in texts[name]])), name
    assert sorted(_texts_of(real_store, "tutoring_guide.md")) == guide_before, "no answers in it: left alone"
    assert any("Answer: x = 7" in text for text in texts["my_worksheet.md"]), "flagged, not rewritten"
    assert {p.problem: p.answer for p in bank.problems()}["10 - 3x = 4"] == "x = 2", "quiz read from the seed"
    assert app_module.migrate_answer_keys(real_store, seed) == ([], ["my_worksheet.md"]), "done once"


def _texts_of(store: VectorStore, source: str) -> list[str]:
    count = next(d["chunks"] for d in store.list_sources() if d["source"] == source)
    return [c.text for c in store.query("x", k=count, sources=[source])]


def test_migration_reingests_an_older_version_of_a_seed_file(tmp_path: Path, real_store: VectorStore) -> None:
    seed = _seed_copy(tmp_path)
    older = tmp_path / "older" / "worksheet_linear_equations.md"
    older.parent.mkdir()
    text = (seed / older.name).read_text(encoding="utf-8")
    older.write_text(text.replace("12. 2(x + 5) = 3x + 4\nAnswer: x = 6", "12. 3(x + 1) = 2x + 9\nAnswer: x = 6"))
    _store_old_way(real_store, older)
    assert app_module.migrate_answer_keys(real_store, seed) == (["worksheet_linear_equations.md"], [])
    assert "12. 2(x + 5) = 3x + 4" in "\n".join(_texts_of(real_store, older.name)), "the bundled version now"


def test_migration_never_overwrites_the_students_file_under_a_seed_name(
    tmp_path: Path, real_store: VectorStore
) -> None:
    seed = _seed_copy(tmp_path)
    own = tmp_path / "own" / "quiz_bank.csv"
    own.parent.mkdir()
    own.write_text("problem,answer\n7x = 49,x = 7\n2 + 2,4\n", encoding="utf-8")
    _store_old_way(real_store, own)
    before = _texts_of(real_store, "quiz_bank.csv")
    assert app_module.migrate_answer_keys(real_store, seed) == ([], ["quiz_bank.csv"])
    assert _texts_of(real_store, "quiz_bank.csv") == before


def test_migrated_app_flags_documents_until_they_are_uploaded_again(
    settings: Settings, fakes: Any, real_store: VectorStore, templates_dir: Path, tmp_path: Path
) -> None:
    _store_old_way(real_store, SEED / "worksheet_linear_equations.md")
    worksheet = _user_worksheet(tmp_path)
    _store_old_way(real_store, worksheet)
    flask_app = _app(settings, fakes, real_store, templates_dir)
    client = flask_app.test_client()
    reingested, flagged = app_module.migrate_app_knowledge(flask_app, settings)
    assert (reingested, flagged) == (["worksheet_linear_equations.md"], ["my_worksheet.md"])
    documents = {d["source"]: d for d in client.get("/api/documents").get_json()["documents"]}
    assert documents["my_worksheet.md"]["has_answers"] is True
    assert "has_answers" not in documents["worksheet_linear_equations.md"]
    _upload(client, "my_worksheet.md", worksheet.read_text(encoding="utf-8"))
    documents = {d["source"]: d for d in client.get("/api/documents").get_json()["documents"]}
    assert "has_answers" not in documents["my_worksheet.md"], "re-uploaded: stored without its answers now"
    assert not any("x = 7" in text for text in _texts_of(real_store, "my_worksheet.md"))


def test_migration_is_math_only(settings: Settings, fakes: Any, real_store: VectorStore, templates_dir: Path) -> None:
    _store_old_way(real_store, SEED / "worksheet_linear_equations.md")
    biology = dataclasses.replace(settings, profile="biology")
    flask_app = _app(biology, fakes, real_store, templates_dir)
    assert app_module.migrate_app_knowledge(flask_app, biology) == ([], [])
    assert any("Answer:" in text for text in _texts_of(real_store, "worksheet_linear_equations.md"))


def test_prepare_knowledge_migrates_then_seeds(
    settings: Settings, fakes: Any, real_store: VectorStore, templates_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(app_module, "migrate_app_knowledge", lambda *a: calls.append("migrate"))
    monkeypatch.setattr(app_module, "seed_app_knowledge", lambda *a: calls.append("seed"))
    flask_app = _app(settings, fakes, real_store, templates_dir)
    app_module.prepare_knowledge(flask_app, settings)
    assert calls == ["migrate", "seed"]
    calls.clear()
    app_module.prepare_knowledge(flask_app, dataclasses.replace(settings, seed_knowledge=False))
    assert calls == ["migrate"], "the migration runs even with seeding switched off"

    def broken(*_args: Any) -> None:
        raise RuntimeError("store down")

    monkeypatch.setattr(app_module, "migrate_app_knowledge", broken)
    calls.clear()
    app_module.prepare_knowledge(flask_app, settings)
    assert calls == ["seed"], "a failed check does not stop seeding"
