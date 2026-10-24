"""Benchmark VLMClient.evaluate against a real Ollama backend.

Usage: python scripts/bench_vlm.py [-n 3] [--model qwen2.5vl:3b] [--url http://localhost:11434]
       python scripts/bench_vlm.py --scenarios [--runs 5] [--profile biology|math] [--only H,P] [--raw]

Math scenarios: H/HA help, P/PF practice (equation / fraction board), QD dictate, QH stuck mid-quiz (hint),
QC/QW/QN check (right / wrong / only the problem written), QDI/QCI/QWI quiz without worksheet problems,
LO/LE/HF lesson offer, starter and follow-up. Quiz cases get the real seeded chunks, answers included.
Spanish (replies in Spanish): ELO/ELE lesson offer and "invéntame uno", EH/EHA help on 3x + 5 = 20,
EQD/EQC/EQW quiz dictate and check (right / wrong).
Every file switched off (no notes at all): EQCN/EQWN (Spanish) and QCN/QWN quiz check, right / wrong; a right answer
fails if the model copied the next problem as the student's answer. HS/EHS: help on 3x + 5 = 20 with the seeded notes
as the math profile stores them now (answers stripped at ingestion), no leak.
Crop: a small "3.5 - x^2 = 0.25" on a big board sent as the high-res ink crop: CR what the model copies off the
board (board_answer of a quiz check: "3.5", then the minus sign) and CRH a help reply whose first step fits the signs
and decimals as written; CRW / CRHW the same on the whole board, as before the crop.
Biology: A missing step, B complete, AB complete after an earlier "not yet".
Graded in code (grading.py): QQC/QQW/QQP a quadratic x^2 - 2x - 15 = 0 checked right / wrong / half right, QSC/QSW a
system right / wrong, EQQW the wrong quadratic in Spanish. A wrong answer must be "error" with one hint and no answer.

--flow runs the whole pipeline (Assistant + this model, a synthetic board, a temporary quiz bank): a quiz with a wrong
then right answer on 3x + 5 = 20, a wrong then right quadratic, then "stop the quiz" -> the confirmation -> "yes" ->
stopped with the score. --flow es runs it in Spanish ("Termina el quiz" -> "Sí").

Example (the final-wiring live check): --scenarios --profile math --runs 2 --only ELO,ELE,EH,EHA,EQD,EQC,EQW,CR
Code grading: --scenarios --profile math --runs 2 --only QC,QW,QQC,QQW,QQP,QSC,QSW,H,HA,EQC,EQW,EQQW
Pipeline: --flow en --runs 2 and --flow es --runs 2
Answer-leak check: --scenarios --profile math --runs 3 --only EQCN,EQWN and --runs 2 --only H,EH,HS,EHS
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biobench_sar import grading
from biobench_sar.lang import detect_language
from biobench_sar.llm.vlm_client import OFFER_QUESTION_ES, VLMClient, praises, reads_problem
from biobench_sar.schemas import AvatarAction, InkRegion, RetrievedChunk

SIZE = (1920, 1080)
MATH_SEED_DIR = Path(__file__).resolve().parent.parent / "biobench_sar" / "knowledge" / "math"
QUERIES = ["Is my flowchart correct?", "Check my procedure, please.", "Does this look right to you?"]
SCENARIO_QUERY = "Check my flowchart. Is the procedure correct and complete?"
STEPS = ["Lake water sample", "Filter 0.45um", "NNA + E. coli", "Incubate 30C", "Microscopy"]
STERILE_STEPS = ["Autoclave media + tools", "Flame-sterilize loop", *STEPS]

CONTEXT = [
    RetrievedChunk(
        "Acanthamoeba is isolated by plating filtered water on non-nutrient agar (NNA) "
        "overlaid with a heat-killed E. coli lawn and incubating at 30C.",
        "protocol.pdf",
        0.9,
        {},
    ),
    RetrievedChunk(
        "Sterile technique: autoclave media and tools at 121C and flame-sterilize loops.",
        "sterile_technique.pdf",
        0.8,
        {},
    ),
]
MATH_CONTEXT = [
    RetrievedChunk(
        "Two-step equations: undo addition or subtraction first, then multiplication or division. "
        "Ask the student to write each step before moving on.",
        "math_tutoring_guide.md",
        0.8,
        {},
    )
]

EQUATION = "3x + 5 = 20"
SPOKEN_EQUATION = "three x plus five equals twenty"
# The solved value said as a result ("x equals five", "fifteen divided by three is five"), not the constant in
# "plus five" / "subtract five" or a question like "what is five less than twenty?".
_LEAK_RE = re.compile(
    r"(?:\bx|\banswer|\bsolution|\b(?:three|3))\s*(?:=|equals|is(?:\s+equal\s+to)?|would\s+be|will\s+be|"
    r"gives(?:\s+you)?)\s*(?:5|five)\b",
    re.IGNORECASE,
)
_NOTATION_RE = re.compile(r"[\^/*=+<>]|\d")
_FRACTION_RE = re.compile(
    r"\d\s*/\s*\d|\b(?:half|halves|thirds?|fourths?|quarters?|fifths?|sixths?|sevenths?|eighths?|ninths?|tenths?|"
    r"over)\b",
    re.IGNORECASE,
)
QUIZ_TASK = {
    "kind": "quiz",
    "stage": "check",
    "number": 1,
    "total": 5,
    "problem": EQUATION,
    "answer": "x = 5",
    "next_problem": "2x - 4 = 10",
}
LEARN_TOPIC = "solving fractions"
LEARN_QUERY = "I want to learn how to solve fractions."
OFFER_TURN = {
    "q": LEARN_QUERY,
    "a": "A fraction is a part of a whole. Do you have an exercise in mind, or should I make one up for you?",
}
EXAMPLE_TURN = {
    "q": "Make one up for me.",
    "a": "Let's start easy. One half plus one fourth. Write it on the board and try the first step.",
}
FRACTION_SUM = "1/2 + 1/4"
_FRACTION_LEAK_RE = re.compile(r"\b(?:three\s+(?:fourths|quarters)|3\s*/\s*4)\b", re.IGNORECASE)
_FRACTION_WORD = (
    r"(?:halves|half|thirds?|fourths?|quarters?|fifths?|sixths?|sevenths?|eighths?|ninths?|tenths?|"
    r"\d\s*/\s*\d)"
)
# "One half plus one fourth equals three fourths": the starter problem solved, not just dictated.
_SOLVED_FRACTION_RE = re.compile(
    rf"\b(?:equals|is|makes|gives)\s+(?:\w+\s+){{0,2}}?{_FRACTION_WORD}(?:\s|[.!,]|$)", re.IGNORECASE
)
FRACTION_BOARD = "3/4 - 1/2"
SPOKEN_FRACTION_BOARD = ("three fourths minus one half", "three quarters minus one half")
QUIZ_PROBLEM = "2x + 3 = 11"
INVENTED_PROBLEM = "4x - 3 = 13"  # an invented quiz problem: the task has no answer, code solves it (x = 4)


def seed_context(*needles: str, stripped: bool = False) -> list[RetrievedChunk]:
    """The real seeded math chunks that mention a problem: with their "Answer:" lines and "answer is" columns (as an
    older store holds them), or `stripped` of them at ingestion, as the math profile stores them now."""
    from biobench_sar.rag.ingestor import SUPPORTED_EXTENSIONS, ingest_file

    chunks: list[RetrievedChunk] = []
    for path in sorted(MATH_SEED_DIR.glob("*")):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        for c in ingest_file(path, strip_answers=stripped):
            if any(n in c.text for n in needles):
                chunks.append(RetrievedChunk(c.text, c.source, 0.7, {}))
    return chunks


def leaks_answer(spoken: str) -> bool:
    """The value of x in 3x + 5 = 20 given away: the regex, or the reply stating the solution sympy finds."""
    return bool(_LEAK_RE.search(spoken)) or grading.states_solution(spoken, grading.solve_text(EQUATION))


def has_notation(spoken: str) -> bool:
    """Symbols or digits the TTS voice would read badly; the prompt asks for words only."""
    return bool(_NOTATION_RE.search(spoken))


def dictates_new_equation(spoken: str) -> bool:
    """At least one equation in the reply that is not just the board's equation read back."""
    text = " ".join(spoken.lower().split())
    equations = len(re.findall(r"\bequals\b|=", text))
    repeats = text.count(SPOKEN_EQUATION) + len(re.findall(r"3\s*x\s*\+\s*5\s*=\s*20", text))
    return equations > repeats


def reads_next_problem(spoken: str) -> bool:
    return bool(re.search(r"\b(?:two|2)\s*(?:times\s+)?x\s*(?:minus|-)\s*(?:four|4)\b", spoken, re.IGNORECASE))


def asks_offer(spoken: str) -> bool:
    text = spoken.lower()
    return "exercise" in text and "make one up" in text and "?" in text


def dictates_fraction(spoken: str) -> bool:
    return bool(_FRACTION_RE.search(spoken)) and not _SOLVED_FRACTION_RE.search(spoken)


def echoes(spoken: str, query: str) -> bool:
    norm = [" ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()) for s in (spoken, query)]
    return norm[0] == norm[1]


class Scenario(NamedTuple):
    name: str
    query: str
    # (jpeg, ink regions) or (jpeg, ink regions, InkCrop) when the image is a crop of the board.
    board: Callable[[], tuple[Any, ...]]
    expected: str
    mode: str = "help"
    task: dict[str, Any] | None = None
    # A list, or a function that loads it when the scenario runs (the seeded math chunks).
    context: list[RetrievedChunk] | Callable[[], list[RetrievedChunk]] = CONTEXT
    check: Callable[[str], str | None] = lambda _spoken: None  # returns a failure reason or None
    turns: list[dict[str, str]] | None = None
    language: str = "en"
    # Check what the model copied off the board (board_problem / board_answer) instead of the spoken reply.
    reads: bool = False

    def chunks(self) -> list[RetrievedChunk]:
        return self.context() if callable(self.context) else list(self.context)


def make_flowchart(steps: list[str]) -> tuple[bytes, list[InkRegion]]:
    img = np.full((SIZE[1], SIZE[0], 3), 255, np.uint8)
    n = len(steps)
    bw, bh = min(300, (SIZE[0] - 120) // n - 40), 120
    gap = (SIZE[0] - 2 * 60 - n * bw) // max(n - 1, 1)
    y = SIZE[1] // 2 - bh // 2
    regions: list[InkRegion] = []
    for i, label in enumerate(steps):
        x = 60 + i * (bw + gap)
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 0, 0), 4)
        words = label.split(" ", 1) if len(label) > 14 else [label]
        for j, w in enumerate(words):
            cv2.putText(img, w, (x + 15, y + 50 + j * 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        if i < n - 1:
            cv2.arrowedLine(
                img, (x + bw + 5, y + bh // 2), (x + bw + gap - 5, y + bh // 2), (0, 0, 0), 4, tipLength=0.3
            )
        regions.append({"x": x, "y": y, "w": bw, "h": bh, "cx": x + bw // 2, "cy": y + bh // 2})
    return _jpeg(img), regions


def draw_board(lines: list[str], size: tuple[int, int] = SIZE) -> tuple[np.ndarray, list[InkRegion]]:
    """Marker-style lines of math on a white board of `size`, one ink region per line, in projector coordinates."""
    img = np.full((size[1], size[0], 3), 255, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick, step = 3.0 * size[0] / SIZE[0], max(3, round(8 * size[0] / SIZE[0])), round(260 * size[1] / SIZE[1])
    regions: list[InkRegion] = []
    for i, line in enumerate(lines):
        (w, h), base = cv2.getTextSize(line, font, scale, thick)
        x, y = round(300 * size[0] / SIZE[0]), round(300 * size[1] / SIZE[1]) + i * step
        cv2.putText(img, line, (x, y), font, scale, (20, 20, 20), thick, cv2.LINE_AA)
        top, height = y - h, h + base
        regions.append({"x": x, "y": top, "w": w, "h": height, "cx": x + w // 2, "cy": top + height // 2})
    return img, regions


def make_board(lines: list[str]) -> tuple[bytes, list[InkRegion]]:
    """Marker-style lines of math, one ink region per line, in projector coordinates."""
    img, regions = draw_board(lines)
    return _jpeg(img), regions


def _jpeg(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return buf.tobytes()


def _no_leak(spoken: str) -> str | None:
    return "gave the final answer" if leaks_answer(spoken) else None


def _new_equation(spoken: str) -> str | None:
    return None if dictates_new_equation(spoken) else "no new equation dictated"


def _next_problem(spoken: str) -> str | None:
    return None if reads_next_problem(spoken) else "next problem not read"


def _no_fraction_leak(spoken: str) -> str | None:
    return "gave the final answer" if _FRACTION_LEAK_RE.search(spoken) else None


def _offer(spoken: str) -> str | None:
    return None if asks_offer(spoken) else "did not ask: exercise in mind or make one up"


def _fraction(spoken: str) -> str | None:
    if _SOLVED_FRACTION_RE.search(spoken):
        return "solved the starter problem"
    return None if dictates_fraction(spoken) else "no fraction problem dictated"


def _new_fraction_problem(spoken: str) -> str | None:
    text = " ".join(spoken.lower().split())
    if any(board in text for board in SPOKEN_FRACTION_BOARD) or re.search(r"3\s*/\s*4\s*-\s*1\s*/\s*2", text):
        return "dictated the board's problem back"
    return None if dictates_fraction(spoken) else "no fraction problem dictated"


def _reads(problem: str) -> Callable[[str], str | None]:
    def check(spoken: str) -> str | None:
        return None if reads_problem(spoken, problem) else f"did not read {problem}"

    return check


def _reads_some_equation(spoken: str) -> str | None:
    return None if re.search(r"\bequals\b", spoken, re.IGNORECASE) else "no problem dictated"


def _no_invented_leak(spoken: str) -> str | None:
    found = re.search(r"\bx\s+(?:=|equals|is|should\s+be)\s+(?:4|four)\b", spoken, re.IGNORECASE)
    return "gave the final answer" if found else None


def _invented_next(spoken: str) -> str | None:
    return None if reads_problem(spoken, "5x + 2 = 27") else "next problem not read"


MISSING_STEP_TURN = {"q": SCENARIO_QUERY, "a": "Not yet. There is no autoclave or flame-sterilize step before plating."}
BIO_SCENARIOS = {
    "A": Scenario("missing sterile step", SCENARIO_QUERY, lambda: make_flowchart(STEPS), "error"),
    "B": Scenario("complete procedure", SCENARIO_QUERY, lambda: make_flowchart(STERILE_STEPS), "celebrating"),
    "AB": Scenario(
        "complete procedure after an earlier 'not yet'",
        SCENARIO_QUERY,
        lambda: make_flowchart(STERILE_STEPS),
        "celebrating",
        turns=[MISSING_STEP_TURN],
    ),
    "BA": Scenario(
        "missing sterile step after an earlier 'complete'",
        SCENARIO_QUERY,
        lambda: make_flowchart(STEPS),
        "error",
        turns=[{"q": SCENARIO_QUERY, "a": "Every sterile step is present and in order. Great work!"}],
    ),
}


def quiz_context() -> list[RetrievedChunk]:
    """What RAG hands a quiz turn on the seeded files: the CSV rows and the worksheet, answers included."""
    return seed_context(EQUATION, QUIZ_PROBLEM)


QUIZ_DICTATE = {
    "kind": "quiz",
    "stage": "dictate",
    "number": 2,
    "total": 10,
    "problem": QUIZ_PROBLEM,
    "answer": "x = 4",
    "next_problem": None,
}
QUIZ_HINT = {**QUIZ_TASK, "stage": "hint", "next_problem": None}
INVENTED_DICTATE = {
    "kind": "quiz",
    "stage": "dictate",
    "number": 1,
    "total": 10,
    "problem": None,
    "answer": None,
    "next_problem": None,
}
INVENTED_CHECK = {
    "kind": "quiz",
    "stage": "check",
    "number": 4,
    "total": 10,
    "problem": INVENTED_PROBLEM,
    "answer": None,
    "next_problem": "5x + 2 = 27",
}
MATH_SCENARIOS = {
    "H": Scenario(
        f"help on {EQUATION}",
        "I'm stuck. What do I do next?",
        lambda: make_board([EQUATION]),
        "pointing",
        context=MATH_CONTEXT,
        check=_no_leak,
    ),
    "HA": Scenario(
        f"asks for the answer to {EQUATION}",
        "Just tell me what x is.",
        lambda: make_board([EQUATION]),
        "pointing",
        context=MATH_CONTEXT,
        check=_no_leak,
    ),
    "P": Scenario(
        f"similar problem to {EQUATION}",
        "Give me a similar problem.",
        lambda: make_board([EQUATION]),
        "idle",
        mode="practice",
        task={"kind": "practice"},
        context=MATH_CONTEXT,
        check=_new_equation,
    ),
    "PF": Scenario(
        f"similar problem to {FRACTION_BOARD}",
        "Another one like this.",
        lambda: make_board([FRACTION_BOARD]),
        "idle",
        mode="practice",
        task={"kind": "practice"},
        context=MATH_CONTEXT,
        check=_new_fraction_problem,
    ),
    "QD": Scenario(
        f"quiz dictate worksheet problem {QUIZ_PROBLEM}",
        "Quiz me.",
        lambda: make_board([]),
        "idle",
        mode="quiz",
        task=QUIZ_DICTATE,
        context=quiz_context,
        check=_reads(QUIZ_PROBLEM),
    ),
    "QH": Scenario(
        f"stuck mid-quiz on {EQUATION}: a hint",
        "I'm stuck.",
        lambda: make_board([EQUATION, "3x = 15"]),
        "pointing",
        mode="quiz",
        task=QUIZ_HINT,
        context=quiz_context,
        check=_no_leak,
        turns=[{"q": "Quiz me.", "a": "Problem one of five. Three x plus five equals twenty. Write your answer."}],
    ),
    "QC": Scenario(
        "quiz check, correct answer x = 5",
        "Is my answer right?",
        lambda: make_board([EQUATION, "x = 5"]),
        "celebrating",
        mode="quiz",
        task=QUIZ_TASK,
        context=quiz_context,
        check=_next_problem,
    ),
    "QW": Scenario(
        "quiz check, wrong answer x = 7",
        "Is my answer right?",
        lambda: make_board([EQUATION, "x = 7"]),
        "error",
        mode="quiz",
        task=QUIZ_TASK,
        context=quiz_context,
        check=_no_leak,
    ),
    "QN": Scenario(
        "quiz check, only the problem written",
        "Is my answer right?",
        lambda: make_board([EQUATION]),
        "idle",
        mode="quiz",
        task=QUIZ_TASK,
        context=quiz_context,
        check=_no_leak,
    ),
    "QDI": Scenario(
        "invented quiz: dictate problem 1, empty board",
        "Start a quiz.",
        lambda: make_board([]),
        "idle",
        mode="quiz",
        task=INVENTED_DICTATE,
        context=[],
        check=_reads_some_equation,
    ),
    "QCI": Scenario(
        f"invented quiz: check {INVENTED_PROBLEM}, correct x = 4",
        "Check my answer.",
        lambda: make_board([INVENTED_PROBLEM, "x = 4"]),
        "celebrating",
        mode="quiz",
        task=INVENTED_CHECK,
        context=[],
        check=_invented_next,
    ),
    "QWI": Scenario(
        f"invented quiz: check {INVENTED_PROBLEM}, wrong x = 3",
        "Check my answer.",
        lambda: make_board([INVENTED_PROBLEM, "x = 3"]),
        "error",
        mode="quiz",
        task=INVENTED_CHECK,
        context=[],
        check=_no_invented_leak,
    ),
    "LO": Scenario(
        "learn offer, empty board",
        LEARN_QUERY,
        lambda: make_board([]),
        "idle",
        task={"kind": "learn", "topic": LEARN_TOPIC, "stage": "offer"},
        context=[],
        check=_offer,
    ),
    "LE": Scenario(
        "learn example after 'make one up'",
        "Make one up for me.",
        lambda: make_board([]),
        "idle",
        task={"kind": "learn", "topic": LEARN_TOPIC, "stage": "example"},
        context=[],
        check=_fraction,
        turns=[OFFER_TURN],
    ),
    "HF": Scenario(
        f"follow-up help on the dictated {FRACTION_SUM}",
        "I wrote it. What do I do first?",
        lambda: make_board([FRACTION_SUM]),
        "pointing",
        context=[],
        check=_no_fraction_leak,
        turns=[OFFER_TURN, EXAMPLE_TURN],
    ),
}

# ---- Spanish -------------------------------------------------------------------------------------------------
# The solved value said as a result in Spanish ("x es igual a cinco", "la respuesta es cinco", "x vale cinco",
# "quince entre tres es cinco"), not "más cinco" / "resta cinco" or a question.
_LEAK_RE_ES = re.compile(
    r"(?:\bx|\brespuesta|\bsoluci[oó]n|\bvalor\s+de\s+x|\btres)\s*(?:=|es\s+igual\s+a|es|vale|ser[ií]a|da|queda\s+en)"
    r"\s*(?:5|cinco)\b",
    re.IGNORECASE,
)
_FRACTION_ES = re.compile(
    r"\b(?:medios?|tercios?|cuartos?|quintos?|sextos?|s[eé]ptimos?|octavos?|novenos?|d[eé]cimos?)\b", re.I
)
_SOLVED_FRACTION_ES = re.compile(
    r"\b(?:es\s+igual\s+a|da|son|resulta)\s+(?:\w+\s+){0,2}?(?:medios?|tercios?|cuartos?|quintos?|sextos?|octavos?)\b",
    re.I,
)
LEARN_TOPIC_ES = "resolver fracciones"
LEARN_QUERY_ES = "Quiero aprender a resolver fracciones."
OFFER_TURN_ES = {"q": LEARN_QUERY_ES, "a": f"Una fracción es una parte de un todo. {OFFER_QUESTION_ES}"}


def spanish(spoken: str) -> str | None:
    """A failure reason unless the reply is Spanish words only."""
    if detect_language(spoken, default="es") != "es":
        return "not Spanish"
    return "symbols/digits" if has_notation(spoken) else None


def _all(*checks: Callable[[str], str | None]) -> Callable[[str], str | None]:
    def check(spoken: str) -> str | None:
        reasons = [r for r in (c(spoken) for c in checks) if r]
        return ", ".join(reasons) or None

    return check


def _no_leak_es(spoken: str) -> str | None:
    return "gave the final answer" if _LEAK_RE_ES.search(spoken) else None


def _offer_es(spoken: str) -> str | None:
    text = spoken.lower()
    asked = "ejercicio" in text and "invente" in text and "?" in spoken
    return None if asked else "did not ask: ¿tienes un ejercicio en mente o invento uno?"


def _fraction_es(spoken: str) -> str | None:
    if _SOLVED_FRACTION_ES.search(spoken):
        return "solved the starter problem"
    return None if _FRACTION_ES.search(spoken) else "no fraction problem dictated"


def _reads_es(problem: str) -> Callable[[str], str | None]:
    def check(spoken: str) -> str | None:
        return None if reads_problem(spoken, problem, "es") else f"did not read {problem} in Spanish"

    return check


SPANISH_SCENARIOS = {
    "ELO": Scenario(
        "ES learn offer, empty board",
        LEARN_QUERY_ES,
        lambda: make_board([]),
        "idle",
        task={"kind": "learn", "topic": LEARN_TOPIC_ES, "stage": "offer"},
        context=[],
        check=_all(_offer_es, spanish),
        language="es",
    ),
    "ELE": Scenario(
        "ES learn example after 'invéntame uno'",
        "Invéntame uno.",
        lambda: make_board([]),
        "idle",
        task={"kind": "learn", "topic": LEARN_TOPIC_ES, "stage": "example"},
        context=[],
        check=_all(_fraction_es, spanish),
        turns=[OFFER_TURN_ES],
        language="es",
    ),
    "EH": Scenario(
        f"ES help on {EQUATION}",
        "Estoy atorado, ¿qué hago ahora?",
        lambda: make_board([EQUATION]),
        "pointing",
        context=MATH_CONTEXT,
        check=_all(_no_leak_es, spanish),
        language="es",
    ),
    "EHA": Scenario(
        f"ES asks for the answer to {EQUATION}",
        "Nada más dime cuánto vale x.",
        lambda: make_board([EQUATION]),
        "pointing",
        context=MATH_CONTEXT,
        check=_all(_no_leak_es, spanish),
        language="es",
    ),
    "EQD": Scenario(
        f"ES quiz dictate {QUIZ_PROBLEM}",
        "Hazme un quiz.",
        lambda: make_board([]),
        "idle",
        mode="quiz",
        task=QUIZ_DICTATE,
        context=quiz_context,
        check=_all(_reads_es(QUIZ_PROBLEM), spanish),
        language="es",
    ),
    "EQC": Scenario(
        "ES quiz check, correct answer x = 5",
        "¿Está bien mi respuesta?",
        lambda: make_board([EQUATION, "x = 5"]),
        "celebrating",
        mode="quiz",
        task=QUIZ_TASK,
        context=quiz_context,
        check=_all(_reads_es(str(QUIZ_TASK["next_problem"])), spanish),
        language="es",
    ),
    "EQW": Scenario(
        "ES quiz check, wrong answer x = 7",
        "¿Está bien mi respuesta?",
        lambda: make_board([EQUATION, "x = 7"]),
        "error",
        mode="quiz",
        task=QUIZ_TASK,
        context=quiz_context,
        check=_all(_no_leak_es, spanish),
        language="es",
    ),
}


# ---- the high-res ink crop ------------------------------------------------------------------------------------
CROP_EQUATION = "3.5 - x^2 = 0.25"
CAMERA = (1920, 1200)  # a camera frame bigger than the WXGA board: the crop keeps its detail
CROP_TEXT_SCALE = 0.65  # digits ~14 camera px tall: about 7 px once the whole board is shrunk to 1024 px
BOARD = (1280, 800)
# Copied off the board with the decimal point and the minus sign: "3.5 - x^2 = 0.25" / "3.5 - x squared equals 0.25".
_COPIED_CROP_RE = re.compile(r"^\s*3\.5\s*[-−]\s*x\b")
# A first step that only fits the board as written: "add x squared" (the minus sign was seen) or a step on 3.5 / 0.25
# (the decimal points were seen). "subtract x squared" or "add three point five" means a sign was misread.
_RIGHT_STEP_RE = re.compile(
    r"\badd\s+x\s+squared\b|\bthree\s+point\s+five\b|\bzero\s+point\s+two\s+five\b", re.IGNORECASE
)
_WRONG_SIGN_RE = re.compile(r"\bsubtract\s+x\s+squared\b|\badd\s+three\s+point\s+five\b", re.IGNORECASE)


def _small_equation_frame() -> np.ndarray:
    """A camera frame of a big board with one small line of math on it, marker-thin, like a student's."""
    img = np.full((CAMERA[1], CAMERA[0], 3), 235, np.uint8)
    x, y = 860, 640
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, CROP_TEXT_SCALE, 2
    for text in ("3.5", "-", "x"):
        cv2.putText(img, text, (x, y), font, scale, (25, 25, 25), thick, cv2.LINE_AA)
        x += cv2.getTextSize(text, font, scale, thick)[0][0] + 8
    cv2.putText(img, "2", (x - 3, y - 10), font, scale * 0.6, (25, 25, 25), 1, cv2.LINE_AA)
    x += 12
    for text in ("=", "0.25"):
        cv2.putText(img, text, (x, y), font, scale, (25, 25, 25), thick, cv2.LINE_AA)
        x += cv2.getTextSize(text, font, scale, thick)[0][0] + 8
    return img


def crop_board() -> tuple[bytes, list[InkRegion], Any]:
    """The pipeline's own path: ink found on the warped board, the crop rendered from the raw camera frame."""
    from biobench_sar.config import Settings
    from biobench_sar.core.homography import Calibration
    from biobench_sar.core.ink_crop import crop_ink_view
    from biobench_sar.core.vision_loop import find_ink_regions

    raw = _small_equation_frame()
    calibration = Calibration.identity(CAMERA, BOARD)
    ink = find_ink_regions(calibration.warp_frame(raw), **Settings(profile="math").ink_kwargs)
    crop = crop_ink_view(raw, calibration, ink)
    assert crop is not None, "no ink found on the synthetic board"
    return crop.jpeg, ink, crop


def whole_board() -> tuple[bytes, list[InkRegion]]:
    from biobench_sar.config import Settings
    from biobench_sar.core.homography import Calibration
    from biobench_sar.core.vision_loop import find_ink_regions

    calibration = Calibration.identity(CAMERA, BOARD)
    warped = calibration.warp_frame(_small_equation_frame())
    return _jpeg(warped), find_ink_regions(warped, **Settings(profile="math").ink_kwargs)


def _copied_crop(read: str) -> str | None:
    return None if _COPIED_CROP_RE.match(read or "") else f"copied {read!r}: not 3.5 and the minus sign"


def _step_fits_the_board(spoken: str) -> str | None:
    if _WRONG_SIGN_RE.search(spoken):
        return "a step for a misread sign"
    return None if _RIGHT_STEP_RE.search(spoken) else "no step on x squared, 3.5 or 0.25"


CROP_QUERY = "Check my answer."
# The problem is not given: the model has only the photo to read it from.
CROP_READ_TASK = {"kind": "quiz", "stage": "check", "number": 1, "total": 5, "problem": "the problem on the board"}
CROP_HELP_QUERY = "Read my equation out loud, then tell me the first step."
CROP_SCENARIOS = {
    "CR": Scenario(
        f"small {CROP_EQUATION} on a big board, high-res crop: what it copies",
        CROP_QUERY,
        crop_board,
        "",
        mode="quiz",
        task=CROP_READ_TASK,
        context=[],
        check=_copied_crop,
        reads=True,
    ),
    "CRW": Scenario(
        f"small {CROP_EQUATION} on a big board, whole board (before the crop): what it copies",
        CROP_QUERY,
        whole_board,
        "",
        mode="quiz",
        task=CROP_READ_TASK,
        context=[],
        check=_copied_crop,
        reads=True,
    ),
    "CRH": Scenario(
        f"small {CROP_EQUATION} on a big board, high-res crop: help",
        CROP_HELP_QUERY,
        crop_board,
        "pointing",
        context=[],
        check=_step_fits_the_board,
    ),
    "CRHW": Scenario(
        f"small {CROP_EQUATION} on a big board, whole board (before the crop): help",
        CROP_HELP_QUERY,
        whole_board,
        "pointing",
        context=[],
        check=_step_fits_the_board,
    ),
}


def stored_seed_context() -> list[RetrievedChunk]:
    """The seeded notes about 3x + 5 = 20 as the math profile stores them now: without their answers."""
    return seed_context(EQUATION, stripped=True)


def _graded(spoken: str) -> str | None:
    """A right answer that came back as "I can't see your answer" / "ask me again": the read was not the board."""
    unread = re.search(r"no veo|no pude leer|can't see|couldn't read", spoken, re.IGNORECASE)
    return "not graded (board_answer was not the student's answer)" if unread else None


NO_FILES_SCENARIOS = {
    "EQCN": Scenario(
        "ES quiz check, every file switched off, correct answer x = 5",
        "¿Está bien mi respuesta?",
        lambda: make_board([EQUATION, "x = 5"]),
        "celebrating",
        mode="quiz",
        task=QUIZ_TASK,
        context=[],
        check=_all(_graded, _reads_es(str(QUIZ_TASK["next_problem"])), spanish),
        language="es",
    ),
    "EQWN": Scenario(
        "ES quiz check, every file switched off, wrong answer x = 7",
        "¿Está bien mi respuesta?",
        lambda: make_board([EQUATION, "x = 7"]),
        "error",
        mode="quiz",
        task=QUIZ_TASK,
        context=[],
        check=_all(_graded, _no_leak_es, spanish),
        language="es",
    ),
    "QCN": Scenario(
        "quiz check, every file switched off, correct answer x = 5",
        "Is my answer right?",
        lambda: make_board([EQUATION, "x = 5"]),
        "celebrating",
        mode="quiz",
        task=QUIZ_TASK,
        context=[],
        check=_all(_graded, _next_problem),
    ),
    "QWN": Scenario(
        "quiz check, every file switched off, wrong answer x = 7",
        "Is my answer right?",
        lambda: make_board([EQUATION, "x = 7"]),
        "error",
        mode="quiz",
        task=QUIZ_TASK,
        context=[],
        check=_all(_graded, _no_leak),
    ),
    "HS": Scenario(
        f"help on {EQUATION} with the seeded notes as stored now",
        "I'm stuck. What do I do next?",
        lambda: make_board([EQUATION]),
        "pointing",
        context=stored_seed_context,
        check=_no_leak,
    ),
    "EHS": Scenario(
        f"ES help on {EQUATION} with the seeded notes as stored now",
        "Estoy atorado, ¿qué hago ahora?",
        lambda: make_board([EQUATION]),
        "pointing",
        context=stored_seed_context,
        check=_all(_no_leak_es, spanish),
        language="es",
    ),
}
# ---- graded in code: several values and systems ----------------------------------------------------------------
QUADRATIC = "x^2 - 2x - 15 = 0"
QUADRATIC_TASK = {
    "kind": "quiz",
    "stage": "check",
    "number": 10,
    "total": 10,
    "problem": QUADRATIC,
    "answer": "x = 5 or x = -3",
    "next_problem": None,
}
SYSTEM = "x + y = 10 and x - y = 4"
SYSTEM_TASK = {**QUADRATIC_TASK, "number": 9, "problem": SYSTEM, "answer": "x = 7 and y = 3"}


def _never_states(problem: str, language: str = "en") -> Callable[[str], str | None]:
    """A wrong answer's reply: "Not quite" with a hint, never praise, never the solution."""
    solutions = grading.solve_text(problem)

    def check(spoken: str) -> str | None:
        if grading.states_solution(spoken, solutions, language):
            return "gave the solution"
        if not re.search(r"^\s*(?:casi|not\s+quite)\b", spoken, re.I):
            return "no 'Not quite' / 'Casi'"
        return "praised a wrong answer" if praises(spoken, language) else None

    return check


GRADED_SCENARIOS = {
    "QQC": Scenario(
        f"quiz check {QUADRATIC}, right: x = 5 or x = -3",
        "Is my answer right?",
        lambda: make_board([QUADRATIC, "x = 5 or x = -3"]),
        "celebrating",
        mode="quiz",
        task=QUADRATIC_TASK,
        context=[],
        check=_graded,
    ),
    "QQW": Scenario(
        f"quiz check {QUADRATIC}, wrong: x = 7 or x = -3",
        "Is my answer right?",
        lambda: make_board([QUADRATIC, "x = 7 or x = -3"]),
        "error",
        mode="quiz",
        task=QUADRATIC_TASK,
        context=[],
        check=_all(_graded, _never_states(QUADRATIC)),
    ),
    "QQP": Scenario(
        f"quiz check {QUADRATIC}, half right: x = 5",
        "Is my answer right?",
        lambda: make_board([QUADRATIC, "x = 5"]),
        "error",
        mode="quiz",
        task=QUADRATIC_TASK,
        context=[],
        check=_all(_graded, _never_states(QUADRATIC)),
    ),
    "QSC": Scenario(
        f"quiz check the system {SYSTEM}, right: x = 7, y = 3",
        "Check my answer.",
        lambda: make_board(["x + y = 10", "x - y = 4", "x = 7, y = 3"]),
        "celebrating",
        mode="quiz",
        task=SYSTEM_TASK,
        context=[],
        check=_graded,
    ),
    "QSW": Scenario(
        f"quiz check the system {SYSTEM}, wrong: x = 8, y = 2",
        "Check my answer.",
        lambda: make_board(["x + y = 10", "x - y = 4", "x = 8, y = 2"]),
        "error",
        mode="quiz",
        task=SYSTEM_TASK,
        context=[],
        check=_all(_graded, _never_states(SYSTEM)),
    ),
    "EQQW": Scenario(
        f"ES quiz check {QUADRATIC}, wrong: x = 7 or x = -3",
        "¿Está bien mi respuesta?",
        lambda: make_board([QUADRATIC, "x = 7 o x = -3"]),
        "error",
        mode="quiz",
        task=QUADRATIC_TASK,
        context=[],
        check=_all(_graded, _never_states(QUADRATIC, "es"), spanish),
        language="es",
    ),
}
MATH_SCENARIOS.update(SPANISH_SCENARIOS)
MATH_SCENARIOS.update(NO_FILES_SCENARIOS)
MATH_SCENARIOS.update(CROP_SCENARIOS)
MATH_SCENARIOS.update(GRADED_SCENARIOS)
SCENARIOS = {"biology": BIO_SCENARIOS, "math": MATH_SCENARIOS}


def judge(scenario: Scenario, action: AvatarAction, read: str | None = None) -> list[str]:
    state_ok = not scenario.expected or action.animation_state == scenario.expected
    problems = [] if state_ok else [f"state {action.animation_state}"]
    if echoes(action.spoken_text, scenario.query):
        problems.append("repeated the student")
    reason = scenario.check((read or "") if scenario.reads else action.spoken_text)
    return problems + ([reason] if reason else [])


def run_scenarios(
    client: VLMClient, runs: int, profile: str, only: list[str] | None = None, *, raw: bool = False
) -> None:
    for key, scenario in SCENARIOS[profile].items():
        if only and key not in only:
            continue
        jpeg, regions, *rest = scenario.board()
        crop = rest[0] if rest else None
        size = BOARD if key.startswith("CR") else SIZE
        context = scenario.chunks()
        counts: Counter[str] = Counter()
        passed, notation, times = 0, 0, []
        for i in range(runs):
            t = time.perf_counter()
            args = (scenario.query, jpeg, context, regions, size)
            kwargs: dict[str, Any] = {"mode": scenario.mode, "task": scenario.task, "recent_turns": scenario.turns}
            kwargs.update(language=scenario.language, crop=crop)
            # raw skips evaluate's math post-processing (quiz grading, offer question, mathspeak): the model alone.
            if raw or scenario.reads:
                reply = client._evaluate(*args, **kwargs)
                a, read = reply.action, reply.board_answer or reply.board_problem
            else:
                a, read = client.evaluate(*args, **kwargs), None
            times.append(time.perf_counter() - t)
            counts[a.animation_state] += 1
            fails = judge(scenario, a, read)
            passed += not fails
            notation += has_notation(a.spoken_text)
            verdict = "ok" if not fails else "FAIL: " + ", ".join(fails)
            shown = getattr(a, "display_text", None)
            print(
                f"  [{key}] run {i + 1}: {times[-1]:.1f}s {a.animation_state} ({a.target_x},{a.target_y}) "
                f"{verdict} {a.spoken_text!r}"
                + (f" display={shown!r}" if shown and shown != a.spoken_text else "")
                + (f" board_answer={read!r}" if read is not None else "")
            )
        sym = f", symbols/digits in {notation}/{runs}" if profile == "math" else ""
        print(
            f"case {key} ({scenario.name}): expect {scenario.expected or 'any state'}: {dict(counts)} -> "
            f"{passed}/{runs} pass"
            f"{sym}, median {statistics.median(times):.1f}s"
        )


# ---- the whole pipeline --------------------------------------------------------------------------------------------
FLOW_PROBLEMS = [("3x + 5 = 20", "x = 5"), (QUADRATIC, "x = 5 or x = -3"), (SYSTEM, "x = 7 and y = 3")]


class FlowStep(NamedTuple):
    query: str
    board: list[str]
    expected: str  # the animation state, "" for any
    check: Callable[[dict[str, Any]], str | None]  # the turn's {"avatar", "said", "tutor"} -> a failure reason


def _said(*needles: str) -> Callable[[dict[str, Any]], str | None]:
    def check(turn: dict[str, Any]) -> str | None:
        text = turn["said"].lower()
        return None if all(n.lower() in text for n in needles) else f"did not say {needles}"

    return check


def _graded_step(verdict: str, problem: str, language: str = "en") -> Callable[[dict[str, Any]], str | None]:
    never = _never_states(problem, language)

    def check(turn: dict[str, Any]) -> str | None:
        if turn["avatar"]["highlight"] is None:
            return "no highlight on the answer"
        return never(turn["said"]) if verdict == "wrong" else None

    return check


def _confirming(turn: dict[str, Any]) -> str | None:
    return None if turn["tutor"].get("confirm_stop") is True and turn["tutor"]["quiz"] else "no confirm_stop"


def _stopped(*needles: str) -> Callable[[dict[str, Any]], str | None]:
    said = _said(*needles)

    def check(turn: dict[str, Any]) -> str | None:
        if turn["tutor"]["quiz"] is not None or turn["tutor"].get("confirm_stop"):
            return "the quiz is still running"
        return said(turn)

    return check


Q1, Q2 = FLOW_PROBLEMS[0][0], FLOW_PROBLEMS[1][0]
FLOWS: dict[str, list[FlowStep]] = {
    "en": [
        FlowStep("Quiz me.", [], "idle", _said("problem one")),
        FlowStep("Check my answer.", [Q1, "x = 7"], "error", _graded_step("wrong", Q1)),
        FlowStep("Check my answer.", [Q1, "x = 5"], "celebrating", _graded_step("right", Q1)),
        FlowStep("Check my answer.", [Q2, "x = 7 or x = -3"], "error", _graded_step("wrong", Q2)),
        FlowStep("Check my answer.", [Q2, "x = 5 or x = -3"], "celebrating", _graded_step("right", Q2)),
        FlowStep("Stop the quiz.", [], "idle", _confirming),
        # Both problems were missed once first: none counts as solved on the first try.
        FlowStep("Yes.", [], "idle", _stopped("the quiz is over", "zero of three")),
    ],
    "es": [
        FlowStep("Hazme un quiz.", [], "idle", _said("problema uno")),
        FlowStep("Revisa mi respuesta.", [Q1, "x = 7"], "error", _graded_step("wrong", Q1, "es")),
        FlowStep("Revisa mi respuesta.", [Q1, "x = 5"], "celebrating", _graded_step("right", Q1, "es")),
        FlowStep("Revisa mi respuesta.", [Q2, "x = 7 o x = -3"], "error", _graded_step("wrong", Q2, "es")),
        FlowStep("Termina el quiz.", [], "idle", _confirming),
        FlowStep("Sí.", [], "idle", _stopped("terminamos el quiz", "cero de tres")),
    ],
}


class _FlowServices:
    """The pipeline's services around a real model: the board a step shows, no notes, no voice."""

    def __init__(self, client: VLMClient) -> None:
        self.vlm = client
        self.vision = self
        self.store = self
        self.tts = self
        self.frame: Any = None

    def latest_warped(self) -> Any:
        return self.frame

    def query(self, *_: Any, **__: Any) -> list[RetrievedChunk]:
        return []

    def speak(self, *_: Any, **__: Any) -> None:
        return None


def run_flow(client: VLMClient, language: str, runs: int) -> None:
    """The Assistant pipeline with this model: real ink detection on a synthetic board, a temporary quiz bank."""
    import tempfile

    from biobench_sar.assistant import Assistant
    from biobench_sar.config import Settings
    from biobench_sar.rag.quiz_bank import QuizProblem

    for run in range(runs):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                profile="math",
                selection_file=Path(tmp) / "selection.json",
                quiz_bank_file=Path(tmp) / "quiz_bank.json",
                quiz_length=len(FLOW_PROBLEMS),
                language=language,
            )
            events: list[tuple[str, dict[str, Any]]] = []
            assistant = Assistant(settings, lambda name, data, seen=events: seen.append((name, data)), _inline)
            problems = [QuizProblem(p, a, "flow", 1, "flow.csv") for p, a in FLOW_PROBLEMS]
            assistant.quiz_bank.replace_source("flow.csv", problems)
            services = _FlowServices(client)
            assistant.services = services
            passed = 0
            for step in FLOWS[language]:
                services.frame = draw_board(step.board, settings.projector_size)[0] if step.board else None
                events.clear()
                t = time.perf_counter()
                assistant.handle_query(step.query)
                took = time.perf_counter() - t
                avatar = next(d for n, d in reversed(events) if n == "avatar_update")
                said = next((d["text"] for n, d in reversed(events) if n == "speak"), avatar["spoken_text"])
                turn = {"avatar": avatar, "said": said, "tutor": assistant.tutor.to_dict()}
                state_ok = not step.expected or avatar["animation_state"] == step.expected
                fails = [] if state_ok else [f"state {avatar['animation_state']}"]
                fails += [r] if (r := step.check(turn)) else []
                passed += not fails
                quiz = turn["tutor"]["quiz"] or {}
                print(
                    f"  [flow {language} run {run + 1}] {step.query!r} board={step.board} -> {took:.1f}s "
                    f"{avatar['animation_state']} ({avatar['target_x']},{avatar['target_y']}) hl={avatar['highlight']} "
                    f"{'ok' if not fails else 'FAIL: ' + ', '.join(fails)} {said!r} "
                    f"score={quiz.get('score')} n={quiz.get('number')} confirm={turn['tutor'].get('confirm_stop')}"
                )
            print(f"flow {language} run {run + 1}: {passed}/{len(FLOWS[language])} steps pass")


def _inline(fn: Callable[..., Any], *args: Any) -> None:
    fn(*args)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--model", default="qwen2.5vl:3b")
    ap.add_argument("--url", default="http://localhost:11434")
    ap.add_argument("--cold", action="store_true", help="unload the model first (measures load + first answer)")
    ap.add_argument("--profile", choices=sorted(SCENARIOS), default="biology")
    ap.add_argument("--scenarios", action="store_true", help="run the profile's pass/fail scenarios")
    ap.add_argument("--only", help="comma-separated scenario keys, e.g. H,QC")
    ap.add_argument("--runs", type=int, default=5, help="runs per scenario")
    ap.add_argument("--raw", action="store_true", help="grade the model without evaluate's math post-processing")
    ap.add_argument(
        "--flow", choices=sorted(FLOWS), help="run the whole pipeline: a quiz checked in code, then stopped"
    )
    args = ap.parse_args()
    if args.cold:
        import requests

        requests.post(f"{args.url}/api/generate", json={"model": args.model, "keep_alive": 0}, timeout=60)
        time.sleep(2)

    client = VLMClient("ollama", args.model, args.url, timeout=180, profile=args.profile)
    t0 = time.perf_counter()
    client.warmup()
    print(f"warmup: {time.perf_counter() - t0:.1f}s")
    if args.flow:
        client.profile = "math"
        run_flow(client, args.flow, args.runs)
        return
    if args.scenarios:
        run_scenarios(client, args.runs, args.profile, args.only.split(",") if args.only else None, raw=args.raw)
        return
    if args.profile == "math":
        scenario = MATH_SCENARIOS["H"]
        jpeg, regions = scenario.board()
        queries, context = [scenario.query], scenario.context
    else:
        jpeg, regions = make_flowchart(STEPS)
        queries, context = QUERIES, CONTEXT
    times: list[float] = []
    for i in range(args.n):
        t = time.perf_counter()
        a = client.evaluate(queries[i % len(queries)], jpeg, context, regions, SIZE)
        dt = time.perf_counter() - t
        times.append(dt)
        print(f"run {i + 1}: {dt:.1f}s  {a.animation_state} ({a.target_x},{a.target_y}) {a.spoken_text!r}")
    print(f"median: {statistics.median(times):.1f}s  min {min(times):.1f}s  max {max(times):.1f}s")


if __name__ == "__main__":
    main()
