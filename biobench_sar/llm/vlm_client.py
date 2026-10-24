from __future__ import annotations

import base64
import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from fractions import Fraction
from typing import Any

import cv2
import numpy as np
import requests

from biobench_sar.schemas import (
    AvatarAction,
    InkRegion,
    RetrievedChunk,
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
PROFILES = ("biology", "math")
MODES = ("help", "practice", "quiz")
# "learn" is a task kind, not a session mode: it can interrupt any mode, so it gets its own prompt.
PROMPT_KINDS = (*MODES, "learn")
LEARN_STAGES = ("offer", "example", "own")
OFFER_QUESTION = "Do you have an exercise in mind, or should I make one up for you?"
DEFAULT_QUIZ_TOTAL = 10
MAX_TURNS = 3
MAX_TURN_CHARS = 240

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
Use "Conversation so far" to understand short follow-ups like "I did it" or "what now?".

Respond with ONLY a JSON object:
{schema}
target_x / target_y are projector pixels within the given width and height."""
MATH_SCHEMA = '{"spoken_text": str, "animation_state": "idle|walking|pointing|celebrating|error", "target_x": int, \
"target_y": int}'
# Reading the answer into its own key first makes the model look at the board, and lets code check the value.
QUIZ_SCHEMA = MATH_SCHEMA.replace('{"spoken_text"', '{"board_answer": str, "spoken_text"')

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
- Read the equation on the board.
- Make ONE new problem of the same type and difficulty: change the numbers (and maybe the letter). \
It must be different from the board's problem.
- Dictate it in words so the student can write it. Do not solve it.
- animation_state "idle"; target = screen center.
- If the board is empty, make up a simple one-step equation.

Examples (JSON only, 1920x1080 screen):
Board: 2x + 3 = 11. Query: "Give me a similar problem."
{"spoken_text": "Try this one. Four x plus seven equals nineteen. Write it down and solve it.", "animation_state": "idle", "target_x": 960, "target_y": 540}
Board: 3/4 - 1/2. Query: "Another one like this."
{"spoken_text": "Here is a new one. Five sixths minus one third. Give it a try.", "animation_state": "idle", "target_x": 960, "target_y": 540}"""

MATH_QUIZ = """MODE: quiz. The message gives the quiz stage and the problem.
- Stage dictate: say the problem number and read the problem in words, then ask the student to write the answer \
on the board. Do not solve it. board_answer "", "idle", target = screen center.
- Stage check: FIRST copy the student's answer exactly as written on the board (usually the last line) into \
board_answer, or "" if no answer is written. Then decide if it is correct.
  - Correct: "celebrating"; praise briefly. If a next problem is given, also read it in words.
  - Wrong or nothing written: "error"; say "Not quite" and give ONE short hint. NEVER say the correct answer \
and do not read the next problem.
  target = center of the answer's ink region.

Examples (JSON only, 1920x1080 screen):
Quiz stage: dictate. Problem 1 of 5: x + 4 = 10
{"board_answer": "", "spoken_text": "Problem one of five. x plus four equals ten. Write your answer on the board.", "animation_state": "idle", "target_x": 960, "target_y": 540}
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

MATH_PROMPTS = {
    kind: MATH_BASE.replace("{schema}", QUIZ_SCHEMA if kind == "quiz" else MATH_SCHEMA) + "\n\n" + rules
    for kind, rules in (("help", MATH_HELP), ("practice", MATH_PRACTICE), ("quiz", MATH_QUIZ), ("learn", MATH_LEARN))
}
# Repeated in the message: a 3B model follows the user turn more closely than the system prompt.
MODE_REMINDERS = {
    "help": "Explain only the next step. Never say the final answer or the value of the variable.",
    "practice": "Make ONE new problem like the one on the board (same type and difficulty, different numbers) and "
    "dictate it in words. Do not solve or explain the board's problem.",
}

MODE_LABELS = {
    "help": "help (step-by-step)",
    "practice": "practice (similar problem)",
    "quiz": "quiz",
    "learn": "learn a topic",
}


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


def _context_text(context: list[RetrievedChunk]) -> str:
    if not context:
        return "Reference context: none."
    blocks = [f"[{i}] (source: {c.source})\n{c.text[:MAX_CHUNK_CHARS]}" for i, c in enumerate(context, 1)]
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
    sides = str(text or "").strip().lower().replace("\u2212", "-").rstrip(".").split("=")
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


def quiz_task_text(task: Mapping[str, Any] | None) -> str:
    t = task or {}
    stage = "check" if t.get("stage") == "check" else "dictate"
    number = _as_int(t.get("number"), 1)
    total = _as_int(t.get("total"), DEFAULT_QUIZ_TOTAL)
    problem, answer, upcoming = _clean(t.get("problem")), _clean(t.get("answer")), _clean(t.get("next_problem"))
    head = f"Quiz stage: {stage}. Problem {number} of {total}: "
    if stage == "dictate":
        if problem:
            return f"{head}{problem}\nRead this problem out in words. Do not solve it."
        return (
            f"{head}no worksheet problem is available, so make up one yourself (problem one is the easiest, each "
            "next problem a little harder). Read it out in words. Do not solve it."
        )
    lines = [head + (problem or "the problem you made up earlier (see the conversation and the board).")]
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
    elif not problem:
        lines.append(f"If the answer is correct, also make up problem {number + 1}, a little harder, and read it out.")
    lines.append(
        "First copy the student's answer from the board into board_answer. "
        'If the answer is wrong: "error", one hint, never the answer.'
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
        history = turns_text(recent_turns, speaker="User")
        parts = [f"User query: {query}", size, _ink_text(ink_regions), _context_text(context)]
        return "\n\n".join([*([history] if history else []), *parts, "Answer with the JSON object only."])
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
        _context_text(context),
        "Answer with the JSON object only. Write all math in words.",
    ]
    return "\n\n".join(parts)


_BOARD_ANSWER_RE = re.compile(r'"board_answer"\s*:\s*"((?:[^"\\]|\\.)*)"')


def board_answer(text: str | None) -> str | None:
    match = _BOARD_ANSWER_RE.search(text or "")
    if not match:
        return None
    try:
        return _clean(json.loads(f'"{match.group(1)}"'))
    except ValueError:
        return None


def check_quiz_answer(
    action: AvatarAction, task: Mapping[str, Any] | None, answer_on_board: str | None
) -> AvatarAction:
    """Grade a numeric quiz answer in code: a 3B model reads "x = 7" fine but often calls it correct anyway."""
    t = task or {}
    expected, given = answer_value(t.get("answer")), answer_value(answer_on_board)
    if expected is None or given is None:
        return action  # nothing checkable: keep the model's own verdict
    state = "celebrating" if given == expected else "error"
    if action.animation_state == state:
        return action
    if state == "error":
        spoken = "Not quite. Look at your steps again and try once more."
    else:
        number, total, upcoming = _as_int(t.get("number"), 1), _as_int(t.get("total"), 0), _clean(t.get("next_problem"))
        spoken = "Correct, great job!"
        if upcoming:
            spoken += f" Problem {number + 1}. {upcoming}"
        elif total and number >= total:
            spoken += " That was the last problem. You finished the quiz!"
    return replace(action, spoken_text=spoken, animation_state=state)


def apply_task_rules(
    action: AvatarAction, mode: str, task: Mapping[str, Any] | None, answer_on_board: str | None = None
) -> AvatarAction:
    """Make what the task fixes deterministic where a 3B model wavers: quiz grading, idle states, the offer question."""
    if action.animation_state == "error" and answer_on_board is None:
        return action  # a failed call (timeout, bad JSON): its message must stay as it is
    kind = resolve_mode(mode, task)
    if kind == "quiz" and (task or {}).get("stage") == "check":
        return check_quiz_answer(action, task, answer_on_board)
    if kind == "learn" and learn_stage(task) == "offer" and "make one up" not in action.spoken_text.lower():
        spoken = action.spoken_text.strip()
        action = replace(action, spoken_text=OFFER_QUESTION if spoken in ("", "...") else f"{spoken} {OFFER_QUESTION}")
    # Dictating or offering is not about one drawing; at temperature 0.6 the model still picks "pointing"/"walking".
    return replace(action, animation_state="idle") if kind in ("practice", "quiz", "learn") else action


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
        action, answer_on_board = self._evaluate(
            query, image_jpeg, context, ink_regions, projector_size, mode, task, recent_turns
        )
        if self.profile != "math":
            return action
        return speakable(apply_task_rules(action, mode, task, answer_on_board))

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
    ) -> tuple[AvatarAction, str | None]:
        """The raw model answer and the board_answer it read, before evaluate's math post-processing."""
        try:
            if self.backend == "mock":
                return self._mock(query, context, ink_regions, projector_size), None
            mode = resolve_mode(mode, task)
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
            system = system_prompt(self.profile, mode)
            temperature = MODE_TEMPERATURE.get(mode, TEMPERATURE) if self.profile == "math" else TEMPERATURE
            image_jpeg = self._prepare_image(image_jpeg)
            if self.backend == "ollama":
                text = self._ollama(prompt, image_jpeg, system, temperature)
            elif self.backend == "openai":
                text = self._openai(prompt, image_jpeg, system, temperature)
            else:
                return error_action(f"Unknown VLM backend '{self.backend}'.", projector_size), None
            return parse_vlm_response(text, projector_size), board_answer(text)
        except requests.Timeout:
            logger.error("VLM request timed out (backend=%s)", self.backend)
            return error_action("The AI model took too long to answer. Please try again.", projector_size), None
        except requests.RequestException as exc:
            logger.error("VLM request failed: %s", exc)
            return error_action("I couldn't reach the AI model. Please check that it is running.", projector_size), None
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            logger.error("Bad VLM payload: %s", exc)
            return error_action("The AI model sent an unexpected reply. Please try again.", projector_size), None
        except Exception as exc:  # evaluate must never raise
            logger.exception("Unexpected VLM failure: %s", exc)
            return error_action("Something went wrong while thinking. Please try again.", projector_size), None

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
    ) -> AvatarAction:
        heard = f'I heard: "{query}".'
        cite = f" According to {context[0].source}." if context else ""
        if ink_regions:
            r = max(ink_regions, key=lambda i: i["w"] * i["h"])
            reply = {
                "spoken_text": f"{heard}{cite} This drawing looks like the main element.",
                "animation_state": "pointing",
                "target_x": r["cx"],
                "target_y": r["cy"],
            }
        else:
            reply = {
                "spoken_text": f"{heard}{cite} I don't see any drawing yet.",
                "animation_state": "idle",
                "target_x": projector_size[0] // 2,
                "target_y": projector_size[1] // 2,
            }
        return parse_vlm_response(json.dumps(reply), projector_size)
