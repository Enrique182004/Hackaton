from __future__ import annotations

import base64
import difflib
import json
import logging
import random
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from fractions import Fraction
from typing import Any, NamedTuple

import cv2
import numpy as np
import requests

from biobench_sar.rag.quiz_bank import invent_problem, similar_problem, solve_problem, strip_answers
from biobench_sar.schemas import (
    AvatarAction,
    InkRegion,
    RetrievedChunk,
    _extract_json_object,
    error_action,
    parse_vlm_response,
)

logger = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 800
KEEP_ALIVE = "30m"
NUM_PREDICT = 200
TEMPERATURE = 0.2
# A repeated "another one" should not get the same problem back.
MODE_TEMPERATURE = {"practice": 0.6}
RETRY_TEMPERATURE = 0.7
PROFILES = ("biology", "math")
MODES = ("help", "practice", "quiz")
# "learn" is a task kind, not a session mode: it can interrupt any mode, so it gets its own prompt.
PROMPT_KINDS = (*MODES, "learn")
LEARN_STAGES = ("offer", "example", "own")
# "hint": the student is stuck on the current quiz problem (help for it, never its answer).
QUIZ_STAGES = ("dictate", "hint", "check")
OFFER_QUESTION = "Do you have an exercise in mind, or should I make one up for you?"
DEFAULT_QUIZ_TOTAL = 10
MAX_TURNS = 3
MAX_TURN_CHARS = 240
# How close a reply may get to the student's words or an earlier reply before it counts as a copy.
ECHO_RATIO = 0.85
REPEAT_RATIO = 0.9

NO_ANSWER_TEXT = "I can't see your answer yet. Write it on the board, then ask me to check it."
WRONG_TEXT = "Not quite. Look at your steps again and try once more."
STEP_TEXT = "Look at what is being done to x, and undo it on both sides. Write that step on the board."

SYSTEM_PROMPT = """You are an expert environmental-biology research assistant, shown as an animated avatar \
projected on a whiteboard. You support a thesis on identifying Acanthamoeba in Laguna de Bustillos and on the \
hydrology and soil runoff of Leptosol and Feozem soils in Cuauhtemoc, Chihuahua, Mexico.

You verify flowcharts and sterile lab procedures drawn on the whiteboard. A COMPLETE Acanthamoeba isolation \
flowchart must contain ALL of: "Autoclave media + tools", "Flame-sterilize loop", water sampling, filtration, \
NNA + E. coli plating, incubation, microscopy. Use the reference context when relevant and name the source.

Be concise: spoken_text is spoken aloud, at most 3 short sentences, no markdown.

Respond with ONLY a JSON object:
{"spoken_text": str, "animation_state": "idle|walking|pointing|celebrating|error", "target_x": int, "target_y": int}

animation_state rules (apply the first that fits):
1. "error": the user asks you to check/verify a flowchart or procedure AND any required step is missing, \
wrong, or out of order (e.g. no autoclave or no flame-sterilize box). Name the missing step. target = the gap \
between the two neighbouring boxes (midpoint of their centers), or the wrong box's center.
2. "celebrating": the user asks you to check a procedure AND every required step is present and in order. \
target = center of the last box.
3. "pointing": the question is about one specific drawing; target = that ink region's center.
4. "idle" or "walking": the question is NOT about the drawing (general science or small talk); target = \
screen center.
Check each required step by name. If the "Autoclave" and "Flame-sterilize" boxes ARE both drawn before the \
sample and the rest of the order is right, the answer is "celebrating", never "error". If either is absent, \
the answer is "error", never "celebrating".

target_x / target_y are projector pixels within the given width and height.

Examples (JSON only):
Chart: Sample -> Filter -> NNA -> Incubate -> Microscopy. Query: "Is this right?"
{"spoken_text": "Not yet. There is no autoclave or flame-sterilize step before plating.", "animation_state": "error", "target_x": 640, "target_y": 540}
Chart: Autoclave -> Flame-sterilize -> Sample -> Filter -> NNA -> Incubate -> Microscopy. Query: "Check it."
{"spoken_text": "Every sterile step is present and in order. Great work!", "animation_state": "celebrating", "target_x": 1700, "target_y": 540}
Query: "Why is the lake water turbid?"
{"spoken_text": "Runoff carries fine sediment from Leptosol soils into the lake.", "animation_state": "idle", "target_x": 960, "target_y": 540}"""


MATH_BASE = """You are a friendly math tutor for middle and high school students (fractions, equations, algebra), \
shown as an animated avatar projected on a whiteboard. The student writes with a marker; you see a photo of the board.

spoken_text is read aloud by a text-to-speech voice: at most 3 short sentences, simple words, no markdown.
CRITICAL: write ALL math in words, numbers too. NEVER use symbols like ^ / * = + - < > or digits in spoken_text.
Say "two thirds" not 2/3, "x squared" not x^2, "three x plus five equals twelve" not 3x+5=12, "negative four" not -4, \
"two times y" not 2*y, "the square root of nine" not sqrt(9).
Use "Conversation so far" only to understand short follow-ups like "I did it" or "what now?". Never copy an \
earlier reply or the student's words: always answer the student's latest message.

Respond with ONLY a JSON object:
{schema}
target_x / target_y are projector pixels within the given width and height."""
MATH_SCHEMA = '{"spoken_text": str, "animation_state": "idle|walking|pointing|celebrating|error", "target_x": int, \
"target_y": int}'
# Reading the answer into its own key first makes the model look at the board, and lets code check the value.
QUIZ_SCHEMA = MATH_SCHEMA.replace('{"spoken_text"', '{"board_answer": str, "spoken_text"')
# Same for practice: code can then tell when the "new" problem is the board's one read back.
PRACTICE_SCHEMA = MATH_SCHEMA.replace('{"spoken_text"', '{"board_problem": str, "spoken_text"')

MATH_HELP = """MODE: step-by-step help.
- Read the equation on the board and the student's latest step.
- Explain ONLY the next single operation in simple words, e.g. "subtract five from both sides".
- NEVER give the final answer or the value of the variable, even if the student asks for it.
- If the latest step has a mistake, say gently what to fix.
- animation_state "pointing"; target = center of the ink region with the part of the equation to work on.
- End by asking the student to write the next step.
- If the board is empty, answer a general question briefly in plain words, or ask the student to write the \
problem: "idle", target = screen center.

Examples (JSON only):
Board (ink center 700,400): 2x + 3 = 11. Query: "Help me."
{"spoken_text": "First get the x term alone. Subtract three from both sides, then write that step.", "animation_state": "pointing", "target_x": 700, "target_y": 400}
Board (ink center 900,500): 1/2 + 1/3. Query: "I'm stuck."
{"spoken_text": "To add one half and one third, find a common denominator. What number do two and three both go into?", "animation_state": "pointing", "target_x": 900, "target_y": 500}
Board (ink center 650,620): x^2 = 49. Query: "Just tell me x."
{"spoken_text": "You can do it! Undo x squared by taking the square root of both sides. Write that step.", "animation_state": "pointing", "target_x": 650, "target_y": 620}"""

MATH_PRACTICE = """MODE: similar problem.
- FIRST copy the problem written on the board into board_problem, exactly ("" if the board is empty).
- Make ONE new problem of the same type and difficulty with NEW numbers (and maybe a new letter). \
It must be different from the board's problem: never read the board's problem back.
- Dictate only the new problem in words so the student can write it. Do not solve it or say its answer.
- animation_state "idle"; target = screen center.
- If the board is empty, make up a simple one-step equation.

Examples (JSON only, 1920x1080 screen):
Board: 2x + 3 = 11. Query: "Give me a similar problem."
{"board_problem": "2x + 3 = 11", "spoken_text": "Try this one. Four x plus seven equals nineteen. Write it down and solve it.", "animation_state": "idle", "target_x": 960, "target_y": 540}
Board: 2/3 + 1/6. Query: "Another one like this."
{"board_problem": "2/3 + 1/6", "spoken_text": "Here is a new one. Three fourths plus one eighth. Give it a try.", "animation_state": "idle", "target_x": 960, "target_y": 540}"""

MATH_QUIZ = """MODE: quiz. The message gives the quiz stage and the problem.
- Stage dictate: say the problem number, read the given problem in words and ask for the answer on the board. \
Do not solve it. board_answer "", "idle", target = screen center.
- Stage hint: the student is stuck. Explain ONLY the next step of the given problem and ask them to write it. \
NEVER say the answer. board_answer "", "pointing" at the ink to work on.
- Stage check: FIRST look at the photo and copy the bottom line of the student's writing into board_answer, \
exactly as written (the problem itself if no answer is written yet). Never solve the problem yourself.
  - Correct: "celebrating"; praise briefly. If a next problem is given, also read it in words.
  - Wrong: "error"; say "Not quite" and give ONE short hint. NEVER say the correct answer \
and do not read the next problem.
  target = center of the answer's ink region.

Examples (JSON only, 1920x1080 screen):
Quiz stage: dictate. Problem 1 of 5: x + 4 = 10
{"board_answer": "", "spoken_text": "Problem one of five. x plus four equals ten. Write your answer on the board.", "animation_state": "idle", "target_x": 960, "target_y": 540}
Quiz stage: hint. Problem 2 of 5: 2x + 3 = 11. Board: 2x + 3 = 11
{"board_answer": "", "spoken_text": "First get two x alone. What can you subtract from both sides?", "animation_state": "pointing", "target_x": 800, "target_y": 400}
Quiz stage: check. Problem 1 of 5: x + 4 = 10. Next problem: 2x = 14. Board: x + 4 = 10, then x = 9
{"board_answer": "x = 9", "spoken_text": "Not quite. What can you subtract from both sides to get x alone?", "animation_state": "error", "target_x": 800, "target_y": 600}
Quiz stage: check. Problem 1 of 5: x + 4 = 10. Next problem: 2x = 14. Board: x + 4 = 10, then x = 6
{"board_answer": "x = 6", "spoken_text": "Correct, great job! Problem two. Two x equals fourteen.", "animation_state": "celebrating", "target_x": 800, "target_y": 600}"""

MATH_LEARN = f"""TASK: the student wants to learn a topic. The board may be empty. The message gives the topic \
and the stage. Use plain words for a middle or high school student.
- Stage offer: one or two plain sentences that introduce the idea of the topic, then ask exactly: \
"{OFFER_QUESTION}" Do not solve anything yet.
- Stage example: dictate ONE easy starter problem on the topic in words, then invite the student to write it on the \
board and try the first step. Do not solve it.
- Stage own: ask the student to write their exercise on the board and then ask you for help.
- Always "idle", target = screen center.

Examples (JSON only, 1920x1080 screen):
Learn stage: offer. Topic: equations. Query: "Teach me equations."
{{"spoken_text": "An equation is like a balance: both sides are equal. We solve it by doing the same thing to both \
sides until the letter is alone. {OFFER_QUESTION}", "animation_state": "idle", "target_x": 960, "target_y": 540}}
Learn stage: example. Topic: adding fractions. Query: "Make one up."
{{"spoken_text": "Let's start easy. One fourth plus two fourths. Write it on the board and try the first step.", \
"animation_state": "idle", "target_x": 960, "target_y": 540}}
Learn stage: own. Topic: fractions. Query: "I have one."
{{"spoken_text": "Great! Write your exercise on the board, then ask me for help.", "animation_state": "idle", \
"target_x": 960, "target_y": 540}}"""

_SCHEMAS = {"quiz": QUIZ_SCHEMA, "practice": PRACTICE_SCHEMA}
MATH_PROMPTS = {
    kind: MATH_BASE.replace("{schema}", _SCHEMAS.get(kind, MATH_SCHEMA)) + "\n\n" + rules
    for kind, rules in (("help", MATH_HELP), ("practice", MATH_PRACTICE), ("quiz", MATH_QUIZ), ("learn", MATH_LEARN))
}
# Repeated in the message: a 3B model follows the user turn more closely than the system prompt.
MODE_REMINDERS = {
    "help": "Explain only the next step. Never say the final answer or the value of the variable.",
    "practice": "Make ONE new problem like the one on the board (same type and difficulty, NEW numbers) and "
    "dictate it in words. Never read the board's problem back; do not solve or explain either problem.",
}

MODE_LABELS = {
    "help": "help (step-by-step)",
    "practice": "practice (similar problem)",
    "quiz": "quiz",
    "learn": "learn a topic",
}


@dataclass(frozen=True)
class TutorAction(AvatarAction):
    """An AvatarAction plus what the tutor needs to know about it. to_dict() still has only the four schema keys.

    failed: the call itself failed (timeout, model unreachable, bad JSON), so it says nothing about the student.
    verdict: a graded quiz check, "correct" or "wrong"; None when there was no answer to grade.
    dictated: the quiz problem this reply read out (dictate: the current one; a correct check: the next one).
    """

    failed: bool = False
    verdict: str | None = None
    dictated: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(AvatarAction)}


def _tutor(action: AvatarAction, **changes: Any) -> TutorAction:
    """`action` as a TutorAction (keeping any tutor fields it has), with `changes` applied."""
    names = [f.name for f in fields(TutorAction)]
    return TutorAction(**{**{n: getattr(action, n) for n in names if hasattr(action, n)}, **changes})


def failed_action(message: str, projector_size: tuple[int, int]) -> TutorAction:
    return _tutor(error_action(message, projector_size), failed=True)


def normalize_profile(profile: str | None) -> str:
    return "biology" if str(profile or "").strip().lower() == "biology" else "math"


def normalize_mode(mode: str | None) -> str:
    mode = str(mode or "").strip().lower()
    return mode if mode in MODES else "help"


def _task_kind(task: Mapping[str, Any] | None) -> str | None:
    kind = task.get("kind") if isinstance(task, Mapping) else None
    return kind if kind in PROMPT_KINDS else None


def resolve_mode(mode: str | None, task: Mapping[str, Any] | None) -> str:
    """The task's kind is more specific than the session mode, so it wins when they disagree."""
    return _task_kind(task) or normalize_mode(mode)


def quiz_stage(task: Mapping[str, Any] | None) -> str:
    stage = (task or {}).get("stage")
    return stage if stage in QUIZ_STAGES else "dictate"


def system_prompt(profile: str, mode: str) -> str:
    if normalize_profile(profile) == "biology":
        return SYSTEM_PROMPT
    kind = str(mode or "").strip().lower()
    return MATH_PROMPTS[kind if kind in PROMPT_KINDS else "help"]


def _ink_text(ink_regions: list[InkRegion], *, math: bool = False) -> str:
    if not ink_regions:
        return "Ink regions detected on the whiteboard: none."
    # In math, "x=300" reads as the student's answer: the model copied box coordinates as "x = 300".
    box = "left={x} top={y} width={w} height={h}" if math else "box x={x} y={y} w={w} h={h}"
    lines = [f"{i}. {box.format(**r)}, center=({r['cx']}, {r['cy']})" for i, r in enumerate(ink_regions, 1)]
    return "Ink regions detected on the whiteboard:\n" + "\n".join(lines)


def _context_text(context: list[RetrievedChunk], *, math: bool = False) -> str:
    # Math notes are worksheets and quiz banks: their answers would be read out or copied as the student's answer.
    texts = [(c.source, strip_answers(c.text) if math else c.text) for c in context]
    texts = [(source, text) for source, text in texts if text.strip()]
    if not texts:
        return "Reference context: none."
    blocks = [f"[{i}] (source: {source})\n{text[:MAX_CHUNK_CHARS]}" for i, (source, text) in enumerate(texts, 1)]
    return "Reference context:\n" + "\n\n".join(blocks)


def _as_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _clean(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


_MIXED_RE = re.compile(r"(-?)(\d+)\s+(\d+)\s*/\s*(\d+)")


def answer_value(text: Any) -> Fraction | None:
    """The number in an answer like "x = 5", "-4", "2/3", "1 1/2" or "0.5"; None unless it is one plain number."""
    sides = str(text if text is not None else "").strip().lower().replace("−", "-").rstrip(".").split("=")
    if len(sides) > 2 or (len(sides) == 2 and not re.fullmatch(r"\s*[a-z]?\s*", sides[0])):
        return None
    value = sides[-1].strip()
    mixed = _MIXED_RE.fullmatch(value)
    try:
        if mixed:
            sign, whole, num, den = mixed.groups()
            return (-1 if sign else 1) * (int(whole) + Fraction(int(num), int(den)))
        return Fraction(value.replace(" ", "")) if value else None
    except (ValueError, ZeroDivisionError):
        return None


def _seed_problem(number: int, total: int) -> str:
    # Deterministic per number, so the prompt and evaluate's post-processing pick the same problem.
    return invent_problem(number, total, random.Random(number * 7919 + total)).problem


def expected_answer(task: Mapping[str, Any] | None) -> str | None:
    """The task's answer, or the one code works out from the problem (e.g. an invented problem)."""
    t = task or {}
    return _clean(t.get("answer")) or solve_problem(_clean(t.get("problem")))


def quiz_task_text(task: Mapping[str, Any] | None) -> str:
    t = task or {}
    stage = quiz_stage(t)
    number = _as_int(t.get("number"), 1)
    total = _as_int(t.get("total"), DEFAULT_QUIZ_TOTAL)
    problem, upcoming = _clean(t.get("problem")), _clean(t.get("next_problem"))
    head = f"Quiz stage: {stage}. Problem {number} of {total}: "
    if stage == "dictate":
        # With no worksheet problem a 3B model echoes the student instead of making one up: code picks it.
        problem = problem or _seed_problem(number, total)
        return f"{head}{problem}\nRead this problem out in words. Do not solve it."
    if stage == "hint":
        return (
            f"{head}{problem or 'the problem on the board'}\nThe student is stuck. Explain ONLY the next single step "
            "for this problem and ask them to write it. NEVER say the answer or the value of the letter."
        )
    lines = [head + (problem or "the problem on the board (see the conversation).")]
    answer = expected_answer(t)
    # A numeric answer is checked in code (check_quiz_answer) and left out here: shown it, a small model copies it
    # into board_answer instead of reading the board, and may say it aloud.
    if answer and answer_value(answer) is None:
        lines.append(f"Expected answer: {answer}. The student's answer is correct ONLY if its value is the same.")
    elif not answer:
        lines.append("Expected answer: not given, solve the problem yourself.")
    if upcoming:
        lines.append(f"Next problem: {upcoming}\nOnly if the answer is correct, also read the next problem in words.")
    elif number >= total:
        lines.append("This is the last problem: if the answer is correct, congratulate the student on finishing.")
    lines.append(
        "First copy the bottom line of the student's writing in the photo into board_answer, exactly as written "
        '(not a problem from this message); do not solve anything yourself. If the answer is wrong: "error", one '
        "hint, never the answer."
    )
    return "\n".join(lines)


def learn_stage(task: Mapping[str, Any] | None) -> str:
    stage = (task or {}).get("stage")
    return stage if stage in LEARN_STAGES else "offer"


def learn_task_text(task: Mapping[str, Any] | None) -> str:
    stage = learn_stage(task)
    topic = _clean((task or {}).get("topic")) or "this math topic"
    todo = {
        "offer": f'Introduce the idea of {topic} in one or two plain sentences, then ask: "{OFFER_QUESTION}"',
        "example": f"The student wants you to make one up: dictate ONE easy starter problem on {topic} in words "
        "and invite them to write it and try the first step. Do not solve it.",
        "own": "The student has their own exercise: ask them to write it on the board and then ask you for help.",
    }[stage]
    return f"Learn stage: {stage}. Topic: {topic}\n{todo}"


def _short(value: Any) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= MAX_TURN_CHARS else text[: MAX_TURN_CHARS - 3].rstrip() + "..."


def turns_text(recent_turns: Sequence[Mapping[str, Any]] | None, speaker: str = "Student") -> str | None:
    """The last few exchanges, oldest first, so follow-ups like "make one up" or "I did it" make sense."""
    lines: list[str] = []
    for turn in list(recent_turns or [])[-MAX_TURNS:]:
        if not isinstance(turn, Mapping):
            continue
        question, answer = _short(turn.get("q")), _short(turn.get("a"))
        lines += [f"{speaker}: {question}"] if question else []
        lines += [f"You: {answer}"] if answer else []
    return "Conversation so far (oldest first):\n" + "\n".join(lines) if lines else None


def build_prompt(
    query: str,
    context: list[RetrievedChunk],
    ink_regions: list[InkRegion],
    projector_size: tuple[int, int],
    *,
    profile: str = "biology",
    mode: str = "help",
    task: Mapping[str, Any] | None = None,
    recent_turns: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    width, height = projector_size
    size = f"Projector size: {width}x{height} pixels (origin top-left)."
    if normalize_profile(profile) == "biology":
        parts = [f"User query: {query}", size, _ink_text(ink_regions), _context_text(context)]
        history = turns_text(recent_turns, speaker="User")
        if history:
            # After the board, not before it: an earlier "Not yet, no autoclave" must not decide a re-check.
            parts.append(f"{history}\nEarlier turns are only for follow-up questions: judge the CURRENT board.")
        return "\n\n".join([*parts, "Answer with the JSON object only."])
    mode = resolve_mode(mode, task)
    parts = [f"Mode: {MODE_LABELS[mode]}." + (f" {MODE_REMINDERS[mode]}" if mode in MODE_REMINDERS else "")]
    if mode == "quiz":
        parts.append(quiz_task_text(task))
    elif mode == "learn":
        parts.append(learn_task_text(task))
    history = turns_text(recent_turns)
    parts += [
        *([history] if history else []),
        f"Student said: {query}",
        f"{size} Screen center: ({width // 2}, {height // 2}).",
        _ink_text(ink_regions, math=True),
        _context_text(context, math=True),
        "Answer with the JSON object only. Write all math in words, numbers too (five, not 5).",
    ]
    return "\n\n".join(parts)


def _reply_value(text: str | None, key: str) -> str | None:
    """A helper key of the model's reply (board_answer, board_problem): a string or a plain JSON number. The regex
    only rescues truncated JSON."""
    data = _extract_json_object(text or "")
    if data is not None:
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            return None
        return _clean(value)
    match = re.search(rf'"{key}"\s*:\s*(?:"((?:[^"\\]|\\.)*)"|(-?\d+(?:\.\d+)?))', text or "")
    if not match:
        return None
    if match.group(2) is not None:
        return match.group(2)
    try:
        return _clean(json.loads(f'"{match.group(1)}"'))
    except ValueError:
        return None


def board_answer(text: str | None) -> str | None:
    return _reply_value(text, "board_answer")


# ---- what a reply says, checked in code ------------------------------------------------------------------------

_NUMBER_WORDS = frozenset(
    re.findall(
        r"\w+",
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
        "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand "
        "half third fourth fifth sixth seventh eighth ninth tenth plus minus times over equals negative",
    )
)
_WORD_FORMS = {"halves": "half", "quarter": "fourth", "quarters": "fourth", "divided": "over"}
_NUMBER = (
    r"(?:\d|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
    r"seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)"
)
# "So x equals five.", "The answer is 5", "x should be seven": the value of the letter said as a result. Not a step
# such as "three x equals fifteen" (the letter has a number in front) and not a question.
_REVEAL_RE = re.compile(
    r"(?:^|[,;:]|\b(?:so|then|and|means|now|great|yes|correct|because)\b)\s*"
    r"(?:x|y|n|the\s+(?:answer|solution|value\s+of\s+[a-z])|your\s+answer)\s+"
    r"(?:=|equals|is(?:\s+equal\s+to)?|would\s+be|will\s+be|should\s+be|must\s+be)\s+"
    rf"(?:negative\s+|minus\s+)?{_NUMBER}"
)
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]*")


def _spoken(text: str) -> str:
    """The text as the voice will say it, so "x = 5" and "x equals five" compare the same."""
    to_speech = _math_to_speech()
    if to_speech is None:
        return text
    try:
        return str(to_speech(text))
    except Exception:
        return text


def _norm_words(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", _spoken(text).lower()).split())


def _math_seq(text: str) -> list[str]:
    """The numbers and operations of a text as words, in order ("thirds" -> "third", "divided" -> "over")."""
    out: list[str] = []
    for word in _norm_words(text).split():
        word = _WORD_FORMS.get(word, word)
        if word.endswith("s") and word[:-1] in _NUMBER_WORDS:
            word = word[:-1]
        if word in _NUMBER_WORDS or word.isdigit():
            out.append(word)
    return out


def _math_words(text: str) -> Counter[str]:
    return Counter(_math_seq(text))


def dictates_problem(spoken: str, like: str | None = None) -> bool:
    """Whether a reply dictates a whole problem: an operation, and "equals" when it should be an equation."""
    said = set(_math_seq(spoken))
    if not said & {"plus", "minus", "times", "over"}:
        return False
    return "equals" in said or "=" not in (like or "")


def reads_back(spoken: str, problem: str | None) -> bool:
    """Whether a reply says this exact problem (its numbers and operations in the same order)."""
    wanted, said = _math_seq(problem or ""), _math_seq(spoken)
    n = len(wanted)
    return bool(wanted) and any(said[i : i + n] == wanted for i in range(len(said) - n + 1))


def reads_problem(spoken: str, problem: str | None) -> bool:
    """Whether a reply reads the problem out: all of its numbers and operations are there (in words or not)."""
    wanted = _math_words(problem or "")
    return bool(wanted) and not (wanted - _math_words(spoken))


def reveals_value(spoken: str) -> bool:
    """A sentence (not a question) that says what the letter equals: a final answer."""
    sentences = _SENTENCE_RE.findall(_spoken(spoken).lower())
    return any(_REVEAL_RE.search(s.strip()) for s in sentences if not s.strip().endswith("?"))


def _similar(a: str, b: str) -> float:
    a, b = _norm_words(a), _norm_words(b)
    return difflib.SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def _is_echo(spoken: str, query: str | None) -> bool:
    return bool(query) and _similar(spoken, str(query)) >= ECHO_RATIO


def _repeated(spoken: str, recent_turns: Sequence[Mapping[str, Any]] | None) -> str | None:
    """The earlier reply this one copies word for word (all of it, or nearly), if any."""
    words = _norm_words(spoken)
    for turn in reversed(list(recent_turns or [])[-MAX_TURNS:]):
        earlier = str(turn.get("a") or "") if isinstance(turn, Mapping) else ""
        before = _norm_words(earlier)
        if before and (_similar(spoken, earlier) >= REPEAT_RATIO or (len(before.split()) >= 4 and before in words)):
            return earlier
    return None


# ---- deterministic task rules ---------------------------------------------------------------------------------


def _dictation(number: int, total: int, problem: str) -> str:
    return f"Problem {number} of {total}. {problem}. Write your answer on the board."


def check_quiz_answer(
    action: AvatarAction, task: Mapping[str, Any] | None, answer_on_board: str | None, *, query: str | None = None
) -> AvatarAction:
    """Grade the quiz answer in code: a 3B model reads "x = 7" fine but often calls it correct anyway."""
    t = task or {}
    number, total = _as_int(t.get("number"), 1), _as_int(t.get("total"), 0)
    upcoming = _clean(t.get("next_problem"))
    expected, given = answer_value(expected_answer(t)), answer_value(answer_on_board)
    spoken, state = action.spoken_text, action.animation_state
    usable = spoken.strip() not in ("", "...") and not _is_echo(spoken, query)
    if expected is None:  # not a plain number (two solutions, a word answer): the model's verdict stands
        verdict = {"celebrating": "correct", "error": "wrong"}.get(state)
        if verdict != "correct":
            return _tutor(action, verdict=verdict)
        praise = spoken if usable else "Correct, great job!"
    elif given is None:
        # Nothing read off the board: never approve (the model praises anything) and never count it as a miss.
        return _tutor(action, spoken_text=NO_ANSWER_TEXT, animation_state="idle", verdict=None)
    elif given != expected:
        keep = state == "error" and usable and not reveals_value(spoken) and not reads_problem(spoken, upcoming)
        return _tutor(action, spoken_text=spoken if keep else WRONG_TEXT, animation_state="error", verdict="wrong")
    else:
        praise = spoken if state == "celebrating" and usable else "Correct, great job!"
    if upcoming and not reads_problem(praise, upcoming):
        praise += f" Problem {number + 1}. {upcoming}"  # the next turn checks it, so it must be read out now
    elif not upcoming and total and number >= total and praise == "Correct, great job!":
        praise += " That was the last problem. You finished the quiz!"
    return _tutor(action, spoken_text=praise, animation_state="celebrating", verdict="correct", dictated=upcoming)


def _quiz_dictation(action: AvatarAction, task: Mapping[str, Any], query: str | None) -> AvatarAction:
    number, total = _as_int(task.get("number"), 1), _as_int(task.get("total"), DEFAULT_QUIZ_TOTAL)
    problem = _clean(task.get("problem")) or _seed_problem(number, total)
    spoken = action.spoken_text
    # The model often echoes "Quiz me." or reads an earlier turn: the student must hear exactly this problem.
    if _is_echo(spoken, query) or not reads_problem(spoken, problem) or reveals_value(spoken):
        spoken = _dictation(number, total, problem)
    return _tutor(action, spoken_text=spoken, animation_state="idle", dictated=problem)


# "One half plus one fourth equals three fourths.": an arithmetic problem dictated together with its result.
# Equations are left alone: in "x plus three equals eight" the "equals eight" is part of the problem.
_SOLVED_RE = re.compile(
    r"^(?P<problem>.*\b(?:plus|minus|times|over|divided\s+by)\b.*?)\s*,?\s+"
    r"(?:equals|is\s+equal\s+to|is|makes|gives(?:\s+you)?)\s+[^.!?]+(?P<end>[.!?]*)$",
    re.I,
)
_LETTER_RE = re.compile(r"\b[a-z]\b(?<!\ba)(?<!\bi)", re.I)


def unsolved(spoken: str) -> str:
    """A dictated problem without the result the model added to it (practice problems and lesson starters)."""
    out: list[str] = []
    for sentence in _SENTENCE_RE.findall(_spoken(spoken)):
        stripped = sentence.strip()
        found = _SOLVED_RE.match(stripped)
        if found and not _LETTER_RE.search(found.group("problem")):
            stripped = found.group("problem").rstrip(" ,") + (found.group("end") or ".")
        if stripped:
            out.append(stripped)
    return " ".join(out) or spoken


_OFFER_ASKED_RE = re.compile(r"\b(?:make\s+one\s+up|exercise|one\s+for\s+you|problem\s+in\s+mind)\b", re.I)


def _with_offer_question(spoken: str) -> str:
    spoken = spoken.strip()
    if spoken in ("", "..."):
        return OFFER_QUESTION
    sentences = [s.strip() for s in _SENTENCE_RE.findall(spoken) if s.strip()]
    if sentences and sentences[-1].endswith("?") and _OFFER_ASKED_RE.search(sentences[-1]):
        return spoken  # asked in its own words
    if sentences and sentences[-1].endswith("?"):
        sentences = sentences[:-1]  # one question at the end, not two in a row
    return " ".join([*sentences, OFFER_QUESTION])


def apply_task_rules(
    action: AvatarAction,
    mode: str,
    task: Mapping[str, Any] | None,
    answer_on_board: str | None = None,
    *,
    query: str | None = None,
) -> AvatarAction:
    """Make what the task fixes deterministic where a 3B model wavers: quiz grading and dictation, idle states,
    the offer question. A failed call is left as it is; any state the model picks is not a failure."""
    if getattr(action, "failed", False):
        return action
    kind = resolve_mode(mode, task)
    t = task or {}
    if kind == "quiz":
        stage = quiz_stage(t)
        if stage == "check":
            return check_quiz_answer(action, t, answer_on_board, query=query)
        if stage == "hint":
            return replace(action, animation_state="pointing")
        return _quiz_dictation(action, t, query)
    if kind == "learn" and learn_stage(t) == "offer":
        action = replace(action, spoken_text=_with_offer_question(action.spoken_text))
    if kind == "practice" or (kind == "learn" and learn_stage(t) == "example"):
        action = replace(action, spoken_text=unsolved(action.spoken_text))  # the student solves it, not the voice
    # Dictating or offering is not about one drawing; at temperature 0.6 the model still picks "pointing"/"walking".
    return replace(action, animation_state="idle") if kind in ("practice", "learn") else action


def _starter(topic: str | None) -> str:
    if "fraction" in (topic or "").lower():
        return "1/5 + 2/5"
    return "x + 3 = 8"


def fallback_text(kind: str, task: Mapping[str, Any] | None, has_ink: bool) -> str:
    """What to say when the model's reply is unusable even after a retry."""
    t = task or {}
    if kind == "learn":
        topic = _clean(t.get("topic")) or "this topic"
        return {
            "offer": f"Let's learn about {topic}. {OFFER_QUESTION}",
            "example": f"Let's start easy. {_starter(topic)}. Write it on the board and try the first step.",
            "own": "Great! Write your exercise on the board, then ask me for help.",
        }[learn_stage(t)]
    if kind == "practice":
        return f"Try this one. {_seed_problem(3, 6)}. Write it down and solve it."
    if kind == "quiz" or has_ink:
        return STEP_TEXT
    return "Write the problem on the board, then ask me for help with the first step."


def _retry_note(
    action: AvatarAction,
    kind: str,
    task: Mapping[str, Any] | None,
    query: str,
    recent_turns: Any,
    board: str | None = None,
) -> str | None:
    """Why a finished math reply must be asked for again, or None. Quiz dictation and grading are fixed in code."""
    if kind == "quiz" and quiz_stage(task) != "hint":
        return None
    spoken = action.spoken_text
    if _is_echo(spoken, query):
        return "Your last reply only repeated the student's words. Answer what they asked instead."
    if kind == "practice" and board and reads_back(spoken, board):
        return f"Your last reply read the board's problem ({board}) back. Dictate a NEW problem with new numbers."
    if kind == "practice" and board and solve_problem(board) and not dictates_problem(spoken, board):
        return f"Your last reply did not dictate a whole new problem like {board}. Say all of it in words."
    earlier = _repeated(spoken, recent_turns)
    if earlier:
        return f'Your last reply copied an earlier one ("{_short(earlier)}"). Answer the student\'s latest message.'
    if kind in ("help", "quiz") and reveals_value(spoken):
        return "Your last reply gave away the value of the letter. Explain only the next step, never the answer."
    return None


def _salvage(
    action: AvatarAction,
    kind: str,
    task: Mapping[str, Any] | None,
    query: str,
    recent_turns: Any,
    has_ink: bool,
    board: str | None = None,
) -> AvatarAction:
    """Keep the sentences that are fine (not an echo, a copy or the answer); otherwise use the task's fallback."""
    spoken = action.spoken_text
    if kind == "practice" and board and (reads_back(spoken, board) or not dictates_problem(spoken, board)):
        variant = similar_problem(board)  # same shape, new numbers, made in code
        if variant:
            return replace(action, spoken_text=f"Try this one. {variant}. Write it down and solve it.")
    earlier = [str(t.get("a") or "") for t in list(recent_turns or []) if isinstance(t, Mapping)]

    def copied(sentence: str) -> bool:
        words = _norm_words(sentence)
        return any(
            _similar(sentence, e) >= REPEAT_RATIO or (len(words.split()) >= 4 and words in _norm_words(e))
            for e in earlier
            if e
        )

    kept = [
        s.strip()
        for s in _SENTENCE_RE.findall(action.spoken_text)
        if s.strip() and not _is_echo(s, query) and not reveals_value(s) and not copied(s)
    ]
    spoken = " ".join(kept) if kept else fallback_text(kind, task, has_ink)
    if kind == "learn" and learn_stage(task) == "offer":
        spoken = _with_offer_question(spoken)
    return replace(action, spoken_text=spoken)


def _math_to_speech() -> Callable[[str], str] | None:
    try:
        from biobench_sar.llm.mathspeak import to_speech
    except ImportError:
        return None
    except Exception as exc:  # a broken module must not break evaluate, which never raises
        logger.warning("mathspeak import failed, speaking the raw text: %s", exc)
        return None
    return to_speech


def speakable(action: AvatarAction) -> AvatarAction:
    """Spell out any notation the model left in (2/3, x^2, 3x+5=12) so the TTS voice reads it naturally."""
    to_speech = _math_to_speech()
    if to_speech is None:
        return action
    try:
        spoken = str(to_speech(action.spoken_text)).strip()
    except Exception as exc:
        logger.warning("mathspeak failed, speaking the raw text: %s", exc)
        return action
    return replace(action, spoken_text=spoken) if spoken else action


class RawReply(NamedTuple):
    """The model's answer before evaluate's math post-processing."""

    action: AvatarAction
    board_answer: str | None
    failed: bool = False
    board_problem: str | None = None


class VLMClient:
    def __init__(
        self,
        backend: str,
        model: str,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 90,
        *,
        max_image_side: int = 1024,
        profile: str = "biology",
    ) -> None:
        self.backend = backend.lower()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_image_side = max_image_side
        self.profile = normalize_profile(profile)
        self.session = requests.Session()

    def warmup(self) -> None:
        """Load the model into memory ahead of the first question. Never raises."""
        if self.backend != "ollama":
            return
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "ok"}],
                "stream": False,
                "keep_alive": KEEP_ALIVE,
                "options": {"num_predict": 1},
            }
            self.session.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout).raise_for_status()
        except Exception as exc:
            logger.warning("VLM warmup failed: %s", exc)

    def _prepare_image(self, image_jpeg: bytes | None) -> bytes | None:
        if not image_jpeg:
            return image_jpeg
        try:
            img = cv2.imdecode(np.frombuffer(image_jpeg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return image_jpeg
            h, w = img.shape[:2]
            scale = self.max_image_side / max(h, w)
            if scale >= 1:
                return image_jpeg
            img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return buf.tobytes() if ok else image_jpeg
        except Exception as exc:
            logger.warning("Image downscale failed, sending original: %s", exc)
            return image_jpeg

    def evaluate(
        self,
        query: str,
        image_jpeg: bytes | None,
        context: list[RetrievedChunk],
        ink_regions: list[InkRegion],
        projector_size: tuple[int, int],
        *,
        mode: str = "help",
        task: Mapping[str, Any] | None = None,
        recent_turns: Sequence[Mapping[str, Any]] | None = None,
    ) -> AvatarAction:
        """Never raises. A failed call comes back as a TutorAction with failed=True."""
        task = task if isinstance(task, Mapping) else None
        args = (query, image_jpeg, context, ink_regions, projector_size, mode, task, recent_turns)
        reply = self._evaluate(*args)
        if reply.failed or self.profile != "math":
            return reply.action
        try:
            kind = resolve_mode(mode, task)
            t = task or {}
            if kind == "quiz" and quiz_stage(t) == "dictate" and not _clean(t.get("problem")):
                number, total = _as_int(t.get("number"), 1), _as_int(t.get("total"), DEFAULT_QUIZ_TOTAL)
                task = {**t, "problem": _seed_problem(number, total)}
                args = (query, image_jpeg, context, ink_regions, projector_size, mode, task, recent_turns)
            action, board = self._finish(reply, mode, task, query), reply.board_problem
            note = None if self.backend == "mock" else _retry_note(action, kind, task, query, recent_turns, board)
            if note is None:
                return action
            logger.info("asking the model again: %s", note)
            retry = self._evaluate(*args, note=note)
            if not retry.failed:
                again, board = self._finish(retry, mode, task, query), retry.board_problem or board
                if _retry_note(again, kind, task, query, recent_turns, board) is None:
                    return again
                action = again
            return speakable(_salvage(action, kind, task, query, recent_turns, bool(ink_regions), board))
        except Exception as exc:  # evaluate must never raise
            logger.exception("Math post-processing failed, using the raw answer: %s", exc)
            return reply.action

    def _finish(self, reply: RawReply, mode: str, task: Mapping[str, Any] | None, query: str) -> AvatarAction:
        return speakable(apply_task_rules(reply.action, mode, task, reply.board_answer, query=query))

    def _evaluate(
        self,
        query: str,
        image_jpeg: bytes | None,
        context: list[RetrievedChunk],
        ink_regions: list[InkRegion],
        projector_size: tuple[int, int],
        mode: str = "help",
        task: Mapping[str, Any] | None = None,
        recent_turns: Sequence[Mapping[str, Any]] | None = None,
        note: str | None = None,
    ) -> RawReply:
        """The raw model answer and the board_answer it read, before evaluate's math post-processing."""
        try:
            mode = resolve_mode(mode, task)
            if self.backend == "mock":
                text = self._mock(query, context, ink_regions, projector_size, mode, task)
                return RawReply(parse_vlm_response(text, projector_size), board_answer(text))
            prompt = build_prompt(
                query,
                context,
                ink_regions,
                projector_size,
                profile=self.profile,
                mode=mode,
                task=task,
                recent_turns=recent_turns,
            )
            if note:
                prompt += f"\n\n{note}"
            system = system_prompt(self.profile, mode)
            temperature = MODE_TEMPERATURE.get(mode, TEMPERATURE) if self.profile == "math" else TEMPERATURE
            temperature = RETRY_TEMPERATURE if note else temperature
            image_jpeg = self._prepare_image(image_jpeg)
            if self.backend == "ollama":
                text = self._ollama(prompt, image_jpeg, system, temperature)
            elif self.backend == "openai":
                text = self._openai(prompt, image_jpeg, system, temperature)
            else:
                return RawReply(failed_action(f"Unknown VLM backend '{self.backend}'.", projector_size), None, True)
            if _extract_json_object(text or "") is None:
                logger.error("VLM reply is not JSON: %.200s", text)
                return RawReply(_tutor(parse_vlm_response(text, projector_size), failed=True), None, True)
            action = parse_vlm_response(text, projector_size)
            return RawReply(action, board_answer(text), False, _reply_value(text, "board_problem"))
        except requests.Timeout:
            logger.error("VLM request timed out (backend=%s)", self.backend)
            message = "The AI model took too long to answer. Please try again."
        except requests.RequestException as exc:
            logger.error("VLM request failed: %s", exc)
            message = "I couldn't reach the AI model. Please check that it is running."
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            logger.error("Bad VLM payload: %s", exc)
            message = "The AI model sent an unexpected reply. Please try again."
        except Exception as exc:  # evaluate must never raise
            logger.exception("Unexpected VLM failure: %s", exc)
            message = "Something went wrong while thinking. Please try again."
        return RawReply(failed_action(message, projector_size), None, True)

    def _ollama(
        self, prompt: str, image_jpeg: bytes | None, system: str = SYSTEM_PROMPT, temperature: float = TEMPERATURE
    ) -> str:
        user: dict[str, object] = {"role": "user", "content": prompt}
        if image_jpeg:
            user["images"] = [base64.b64encode(image_jpeg).decode("ascii")]
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, user],
            "stream": False,
            "format": "json",
            "keep_alive": KEEP_ALIVE,
            "options": {"temperature": temperature, "num_predict": NUM_PREDICT},
        }
        resp = self.session.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return str(resp.json()["message"]["content"])

    def _openai(
        self, prompt: str, image_jpeg: bytes | None, system: str = SYSTEM_PROMPT, temperature: float = TEMPERATURE
    ) -> str:
        content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
        if image_jpeg:
            b64 = base64.b64encode(image_jpeg).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        path = "/chat/completions" if self.base_url.endswith("/v1") else "/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        resp = self.session.post(f"{self.base_url}{path}", json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        return str(resp.json()["choices"][0]["message"]["content"])

    def _mock(
        self,
        query: str,
        context: list[RetrievedChunk],
        ink_regions: list[InkRegion],
        projector_size: tuple[int, int],
        mode: str = "help",
        task: Mapping[str, Any] | None = None,
    ) -> str:
        """An offline stand-in. For math tasks it plays a model that follows the task (and grades every quiz answer
        as correct), so the whole tutor flow, quiz included, can be clicked through without a model."""
        width, height = projector_size
        center = {"target_x": width // 2, "target_y": height // 2}
        largest = max(ink_regions, key=lambda i: i["w"] * i["h"]) if ink_regions else None
        at_ink = {"target_x": largest["cx"], "target_y": largest["cy"]} if largest else center
        t = task or {}
        reply: dict[str, Any] | None = None
        if self.profile == "math":
            if mode == "quiz":
                stage = quiz_stage(t)
                number, total = _as_int(t.get("number"), 1), _as_int(t.get("total"), DEFAULT_QUIZ_TOTAL)
                problem = _clean(t.get("problem")) or _seed_problem(number, total)
                if stage == "dictate":
                    reply = {"spoken_text": _dictation(number, total, problem), "animation_state": "idle", **center}
                elif stage == "hint":
                    reply = {"spoken_text": STEP_TEXT, "animation_state": "pointing", **at_ink}
                else:
                    answer = expected_answer(t) or ""
                    reply = {"board_answer": answer, "spoken_text": "Correct, great job!", **at_ink}
                    reply["animation_state"] = "celebrating"
            elif mode in ("learn", "practice"):
                reply = {"spoken_text": fallback_text(mode, t, bool(ink_regions)), "animation_state": "idle", **center}
        if reply is None:
            heard = f'I heard: "{query}".'
            cite = f" According to {context[0].source}." if context else ""
            if largest:
                reply = {
                    "spoken_text": f"{heard}{cite} This drawing looks like the main element.",
                    "animation_state": "pointing",
                    **at_ink,
                }
            else:
                reply = {
                    "spoken_text": f"{heard}{cite} I don't see any drawing yet.",
                    "animation_state": "idle",
                    **center,
                }
        return json.dumps(reply)
