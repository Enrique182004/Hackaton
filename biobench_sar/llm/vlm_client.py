from __future__ import annotations

import base64
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
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
{"spoken_text": str, "animation_state": "idle|walking|pointing|celebrating|error", "target_x": int, "target_y": int}
target_x / target_y are projector pixels within the given width and height."""

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
on the board. Do not solve it. "idle", target = screen center.
- Stage check: read the student's answer on the board and compare its value with the expected answer.
  - Same value: "celebrating", target = center of the answer's ink region; praise briefly. If a next problem is \
given, also read it in words in the same spoken_text.
  - Different value or nothing written: "error", target = center of the answer's ink region; give ONE short hint. \
NEVER say the correct answer and do not read the next problem.

Examples (JSON only, 1920x1080 screen):
Quiz stage: dictate. Problem 1 of 5: x + 4 = 10
{"spoken_text": "Problem one of five. x plus four equals ten. Write your answer on the board.", "animation_state": "idle", "target_x": 960, "target_y": 540}
Quiz stage: check. Problem 1 of 5: x + 4 = 10. Expected answer: x = 6. Next problem: 2x = 14. Board (ink center 800,600): x = 9
{"spoken_text": "Not quite, nine is not right. What can you subtract from both sides to get x alone?", "animation_state": "error", "target_x": 800, "target_y": 600}
Quiz stage: check. Problem 1 of 5: x + 4 = 10. Expected answer: x = 6. Next problem: 2x = 14. Board (ink center 800,600): x = 6
{"spoken_text": "Correct, great job! Problem two. Two x equals fourteen.", "animation_state": "celebrating", "target_x": 800, "target_y": 600}"""

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
{{"spoken_text": "Let's start easy. One fourth plus two fourths. Write it on the board and try adding the top \
numbers.", "animation_state": "idle", "target_x": 960, "target_y": 540}}
Learn stage: own. Topic: fractions. Query: "I have one."
{{"spoken_text": "Great! Write your exercise on the board, then ask me for help.", "animation_state": "idle", \
"target_x": 960, "target_y": 540}}"""

MATH_PROMPTS = {
    kind: f"{MATH_BASE}\n\n{rules}"
    for kind, rules in (("help", MATH_HELP), ("practice", MATH_PRACTICE), ("quiz", MATH_QUIZ), ("learn", MATH_LEARN))
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


def _ink_text(ink_regions: list[InkRegion]) -> str:
    if not ink_regions:
        return "Ink regions detected on the whiteboard: none."
    lines = [
        f"{i}. box x={r['x']} y={r['y']} w={r['w']} h={r['h']}, center=({r['cx']}, {r['cy']})"
        for i, r in enumerate(ink_regions, 1)
    ]
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
    if answer:
        lines.append(f"Expected answer: {answer}. The student's answer is correct ONLY if its value is the same.")
    else:
        lines.append("Expected answer: not given, solve the problem yourself.")
    if upcoming:
        lines.append(f"Next problem: {upcoming}\nOnly if the answer is correct, also read the next problem in words.")
    elif number >= total:
        lines.append("This is the last problem: if the answer is correct, congratulate the student on finishing.")
    elif not problem:
        lines.append(f"If the answer is correct, also make up problem {number + 1}, a little harder, and read it out.")
    lines.append('If the answer is wrong: "error", one hint, never the answer.')
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
    parts = [f"Mode: {MODE_LABELS[mode]}."]
    if mode == "quiz":
        parts.append(quiz_task_text(task))
    elif mode == "learn":
        parts.append(learn_task_text(task))
    history = turns_text(recent_turns)
    parts += [
        *([history] if history else []),
        f"Student said: {query}",
        f"{size} Screen center: ({width // 2}, {height // 2}).",
        _ink_text(ink_regions),
        _context_text(context),
        "Answer with the JSON object only. Write all math in words.",
    ]
    return "\n\n".join(parts)


def apply_task_rules(action: AvatarAction, task: Mapping[str, Any] | None) -> AvatarAction:
    """A 3B model sometimes drops the offer question or picks a drawing state; learn turns always need both right."""
    if _task_kind(task) != "learn" or action.animation_state == "error":
        return action  # "error" here is a failed call, whose message must stay as it is
    spoken = action.spoken_text
    if learn_stage(task) == "offer" and "make one up" not in spoken.lower():
        spoken = OFFER_QUESTION if spoken.strip() in ("", "...") else f"{spoken.rstrip()} {OFFER_QUESTION}"
    return replace(action, spoken_text=spoken, animation_state="idle")


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
        action = self._evaluate(query, image_jpeg, context, ink_regions, projector_size, mode, task, recent_turns)
        if self.profile != "math":
            return action
        return speakable(apply_task_rules(action, task))

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
    ) -> AvatarAction:
        """The raw model answer, before the math post-processing in evaluate."""
        try:
            if self.backend == "mock":
                return self._mock(query, context, ink_regions, projector_size)
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
                return error_action(f"Unknown VLM backend '{self.backend}'.", projector_size)
            return parse_vlm_response(text, projector_size)
        except requests.Timeout:
            logger.error("VLM request timed out (backend=%s)", self.backend)
            return error_action("The AI model took too long to answer. Please try again.", projector_size)
        except requests.RequestException as exc:
            logger.error("VLM request failed: %s", exc)
            return error_action("I couldn't reach the AI model. Please check that it is running.", projector_size)
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            logger.error("Bad VLM payload: %s", exc)
            return error_action("The AI model sent an unexpected reply. Please try again.", projector_size)
        except Exception as exc:  # evaluate must never raise
            logger.exception("Unexpected VLM failure: %s", exc)
            return error_action("Something went wrong while thinking. Please try again.", projector_size)

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
