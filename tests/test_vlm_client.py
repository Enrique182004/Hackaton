from __future__ import annotations

import base64
import importlib.util
import json
import re
import subprocess
import sys
import types
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import requests

from biobench_sar import grading
from biobench_sar.llm import vlm_client as vc
from biobench_sar.llm.vlm_client import OFFER_QUESTION, SYSTEM_PROMPT, VLMClient, build_prompt, system_prompt
from biobench_sar.schemas import ANIMATION_STATES, AvatarAction, InkRegion, RetrievedChunk

SIZE = (1920, 1080)
JPEG = b"\xff\xd8fakejpeg"
B64 = base64.b64encode(JPEG).decode()
GOOD = {"spoken_text": "Add autoclaving.", "animation_state": "error", "target_x": 100, "target_y": 200}

REGIONS: list[InkRegion] = [
    {"x": 0, "y": 0, "w": 10, "h": 10, "cx": 5, "cy": 5},
    {"x": 500, "y": 400, "w": 200, "h": 100, "cx": 600, "cy": 450},
]
CHUNKS = [RetrievedChunk("Autoclave media at 121C. " + "x" * 2000, "protocol.pdf", 0.9, {})]


def _resp(payload: object) -> MagicMock:
    r = MagicMock()
    r.json.return_value = payload
    r.raise_for_status.return_value = None
    return r


def _ollama_resp(text: str) -> MagicMock:
    return _resp({"message": {"content": text}})


def _client(backend: str, base_url: str = "http://x:11434") -> VLMClient:
    return VLMClient(backend, "m", base_url, api_key="k")


def _run(c: VLMClient, response: MagicMock | Exception, image: bytes | None = JPEG):
    with patch.object(
        c.session,
        "post",
        side_effect=[response] if isinstance(response, Exception) else None,
        return_value=None if isinstance(response, Exception) else response,
    ) as post:
        action = c.evaluate("check my flowchart", image, CHUNKS, REGIONS, SIZE)
    return action, post


def test_ollama_payload_and_parse() -> None:
    action, post = _run(_client("ollama"), _ollama_resp(json.dumps(GOOD)))
    assert post.call_args.args[0] == "http://x:11434/api/chat"
    p = post.call_args.kwargs["json"]
    assert p["stream"] is False and p["format"] == "json" and p["options"] == {"temperature": 0.2, "num_predict": 200}
    assert p["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert p["messages"][1]["images"] == [B64]
    assert p["keep_alive"] == "30m"
    assert "check my flowchart" in p["messages"][1]["content"]
    assert action.animation_state == "error" and (action.target_x, action.target_y) == (100, 200)


def test_ollama_without_image() -> None:
    _, post = _run(_client("ollama"), _ollama_resp(json.dumps(GOOD)), image=None)
    assert "images" not in post.call_args.kwargs["json"]["messages"][1]


def test_openai_payload_and_url() -> None:
    body = {"choices": [{"message": {"content": json.dumps(GOOD)}}]}
    action, post = _run(_client("openai", "https://api.example.com"), _resp(body))
    assert post.call_args.args[0] == "https://api.example.com/v1/chat/completions"
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer k"
    p = post.call_args.kwargs["json"]
    assert p["response_format"] == {"type": "json_object"}
    parts = p["messages"][1]["content"]
    assert parts[0]["type"] == "text"
    assert parts[1]["image_url"]["url"] == f"data:image/jpeg;base64,{B64}"
    assert action.spoken_text == "Add autoclaving."


def test_openai_no_duplicate_v1() -> None:
    body = {"choices": [{"message": {"content": json.dumps(GOOD)}}]}
    _, post = _run(_client("openai", "https://api.example.com/v1/"), _resp(body))
    assert post.call_args.args[0] == "https://api.example.com/v1/chat/completions"


def test_markdown_wrapped_json() -> None:
    wrapped = "Sure!\n```json\n" + json.dumps(GOOD) + "\n```"
    action, _ = _run(_client("ollama"), _ollama_resp(wrapped))
    assert action.animation_state == "error" and action.target_x == 100


def test_garbage_gives_error_action() -> None:
    action, _ = _run(_client("ollama"), _ollama_resp("no json here"))
    assert action.animation_state == "error"


def test_bad_payload_gives_error_action() -> None:
    action, _ = _run(_client("ollama"), _resp({"unexpected": 1}))
    assert action.animation_state == "error"


def test_timeout_gives_error_action() -> None:
    action, _ = _run(_client("ollama"), requests.Timeout("slow"))
    assert action.animation_state == "error" and "too long" in action.spoken_text
    assert (action.target_x, action.target_y) == (960, 540)


def test_connection_error_gives_error_action() -> None:
    action, _ = _run(_client("openai"), requests.ConnectionError("down"))
    assert action.animation_state == "error"


def test_mock_points_at_largest_region() -> None:
    c = _client("mock")
    with patch.object(c.session, "post") as post:
        action = c.evaluate("hello", None, CHUNKS, REGIONS, SIZE)
    post.assert_not_called()
    assert action.animation_state == "pointing" and (action.target_x, action.target_y) == (600, 450)
    assert "hello" in action.spoken_text and "protocol.pdf" in action.spoken_text


def test_mock_no_regions_idle_center() -> None:
    action = _client("mock").evaluate("hi", None, [], [], SIZE)
    assert action.animation_state == "idle" and (action.target_x, action.target_y) == (960, 540)


def test_unknown_backend_gives_error_action() -> None:
    assert _client("bogus").evaluate("q", None, [], [], SIZE).animation_state == "error"


def test_build_prompt_contents() -> None:
    prompt = build_prompt("why turbid?", CHUNKS, REGIONS, SIZE)
    assert "why turbid?" in prompt and "1920x1080" in prompt
    assert "center=(600, 450)" in prompt and "2. " in prompt
    assert "protocol.pdf" in prompt and "Autoclave media" in prompt
    assert "x" * 900 not in prompt


def _big_jpeg(w: int = 1920, h: int = 1080) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), 200, np.uint8))
    assert ok
    return buf.tobytes()


def _sent_image_shape(post: MagicMock) -> tuple[int, ...]:
    b64 = post.call_args.kwargs["json"]["messages"][1]["images"][0]
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR).shape


def test_large_image_is_downscaled() -> None:
    _, post = _run(_client("ollama"), _ollama_resp(json.dumps(GOOD)), image=_big_jpeg())
    h, w = _sent_image_shape(post)[:2]
    assert (w, h) == (1024, 576)


def test_small_image_untouched() -> None:
    small = _big_jpeg(640, 480)
    _, post = _run(_client("ollama"), _ollama_resp(json.dumps(GOOD)), image=small)
    assert post.call_args.kwargs["json"]["messages"][1]["images"] == [base64.b64encode(small).decode()]


def test_max_image_side_configurable() -> None:
    c = VLMClient("ollama", "m", "http://x:11434", max_image_side=512)
    _, post = _run(c, _ollama_resp(json.dumps(GOOD)), image=_big_jpeg())
    assert _sent_image_shape(post)[1] == 512


def test_warmup_ollama_sends_keep_alive() -> None:
    c = _client("ollama")
    with patch.object(c.session, "post", return_value=_resp({})) as post:
        c.warmup()
    assert post.call_args.kwargs["json"]["keep_alive"] == "30m"


def test_warmup_never_raises() -> None:
    c = _client("ollama")
    with patch.object(c.session, "post", side_effect=requests.ConnectionError("down")):
        c.warmup()


def test_warmup_noop_for_other_backends() -> None:
    for backend in ("mock", "openai"):
        c = _client(backend)
        with patch.object(c.session, "post") as post:
            c.warmup()
        post.assert_not_called()


def test_system_prompt_schema_rules_and_few_shot() -> None:
    for key in ("spoken_text", "animation_state", "target_x", "target_y"):
        assert key in SYSTEM_PROMPT
    for state in ("idle", "walking", "pointing", "celebrating", "error"):
        assert f'"{state}"' in SYSTEM_PROMPT or state in SYSTEM_PROMPT
    assert "gap" in SYSTEM_PROMPT and "Flame-sterilize" in SYSTEM_PROMPT and "Autoclave" in SYSTEM_PROMPT
    assert SYSTEM_PROMPT.count('"animation_state": "error"') >= 1
    assert SYSTEM_PROMPT.count('"animation_state": "celebrating"') >= 1
    assert SYSTEM_PROMPT.count('"animation_state": "idle"') >= 1
    assert len(SYSTEM_PROMPT) < 3500


# --- math profile -------------------------------------------------------------------------------------------

SPEECH_MODULE = "biobench_sar.llm.mathspeak"
BENCH = Path(__file__).resolve().parent.parent / "scripts" / "bench_vlm.py"
NOTATION = re.compile(r"[\^/*=+<>]|\d")
MATH_GOOD = {
    "spoken_text": "Subtract five from both sides.",
    "animation_state": "pointing",
    "target_x": 7,
    "target_y": 9,
}
LEARN_OFFER = {"kind": "learn", "topic": "solving fractions", "stage": "offer"}


@pytest.fixture
def no_mathspeak(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, SPEECH_MODULE, None)  # a None entry makes the import raise ImportError


@pytest.fixture
def fake_mathspeak(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    mod = types.ModuleType(SPEECH_MODULE)

    def to_speech(text: str) -> str:
        calls.append(text)
        return text.replace("2/3", "two thirds")

    mod.to_speech = to_speech  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, SPEECH_MODULE, mod)
    return calls


def _math_client(backend: str = "ollama") -> VLMClient:
    return VLMClient(backend, "m", "http://x:11434", profile="math")


def _ask(c: VLMClient, reply: object, query: str = "help me", **kwargs: Any) -> tuple[AvatarAction, dict[str, Any]]:
    response = reply if isinstance(reply, Exception) else _ollama_resp(json.dumps(reply))
    side_effect = [response] if isinstance(response, Exception) else None
    with patch.object(c.session, "post", return_value=response, side_effect=side_effect) as post:
        action = c.evaluate(query, JPEG, CHUNKS, REGIONS, SIZE, **kwargs)
    return action, post.call_args.kwargs["json"]


def _few_shot(prompt: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in prompt.splitlines() if line.startswith('{"') and ": str" not in line]


def test_biology_system_prompt_ignores_mode() -> None:
    for mode in (*vc.MODES, "learn", "bogus"):
        assert system_prompt("biology", mode) is SYSTEM_PROMPT
        assert system_prompt(" Biology ", mode) is SYSTEM_PROMPT


def test_unknown_profile_and_mode_fall_back_to_math_help() -> None:
    assert system_prompt("chemistry", "help") == system_prompt("math", "help")
    assert system_prompt("math", "bogus") == system_prompt("math", "help")
    assert system_prompt("math", " QUIZ ") == system_prompt("math", "quiz")
    assert vc.MODES == ("help", "practice", "quiz")
    assert len({system_prompt("math", kind) for kind in vc.PROMPT_KINDS}) == 4


@pytest.mark.parametrize("kind", vc.PROMPT_KINDS)
def test_math_prompts_have_phonetic_rules_schema_and_stay_compact(kind: str) -> None:
    prompt = system_prompt("math", kind)
    assert "CRITICAL" in prompt and "NEVER use symbols" in prompt
    for phrase in ("two thirds", "x squared", "three x plus five equals twelve", "negative four"):
        assert phrase in prompt
    for key in ("spoken_text", "animation_state", "target_x", "target_y"):
        assert key in prompt
    assert "Conversation so far" in prompt
    assert ("board_answer" in prompt) == (kind == "quiz")
    # Every token of system prompt is paid on each question; the scan rule for small symbols is worth its 160.
    assert len(prompt) < 3200


@pytest.mark.parametrize(
    ("kind", "states"),
    [
        ("help", {"pointing"}),
        ("practice", {"idle"}),
        ("quiz", {"idle", "pointing", "error", "celebrating"}),
        ("learn", {"idle"}),
    ],
)
def test_math_few_shot_examples_are_json_in_words(kind: str, states: set[str]) -> None:
    examples = _few_shot(system_prompt("math", kind))
    assert len(examples) >= 2
    assert {e["animation_state"] for e in examples} == states
    for e in examples:
        assert e["animation_state"] in ANIMATION_STATES
        assert isinstance(e["target_x"], int) and isinstance(e["target_y"], int)
        assert not NOTATION.search(e["spoken_text"]), e["spoken_text"]
        assert set(e) - {"board_answer", "board_problem"} == {"spoken_text", "animation_state", "target_x", "target_y"}


def test_math_mode_rules() -> None:
    help_prompt, practice, quiz, learn = (system_prompt("math", k) for k in ("help", "practice", "quiz", "learn"))
    assert "NEVER give the final answer" in help_prompt and "next single operation" in help_prompt
    assert "board is empty" in help_prompt
    assert "same type and difficulty" in practice and "different from the board" in practice
    assert "NEVER say the correct answer" in quiz and '"celebrating"' in quiz and '"error"' in quiz
    assert f'ask exactly: "{OFFER_QUESTION}"' in learn
    assert any(OFFER_QUESTION in e["spoken_text"] for e in _few_shot(learn))
    for stage in vc.LEARN_STAGES:
        assert f"Stage {stage}" in learn


def test_build_prompt_math_help() -> None:
    prompt = build_prompt("I'm stuck", CHUNKS, REGIONS, SIZE, profile="math")
    assert prompt.startswith("Mode: help (step-by-step).")
    assert "Student said: I'm stuck" in prompt and "Screen center: (960, 540)" in prompt
    assert "center=(600, 450)" in prompt and "protocol.pdf" in prompt and "x" * 900 not in prompt
    assert "Write all math in words, numbers too (five, not 5)." in prompt and "Never say the final answer" in prompt
    # "x=500" would read as the student's answer; math boxes avoid the letter.
    assert "1. left=0 top=0 width=10 height=10, center=(5, 5)" in prompt and "x=500" not in prompt
    assert "Conversation so far" not in prompt and "Quiz stage" not in prompt


def test_build_prompt_math_empty_board() -> None:
    prompt = build_prompt("What is a fraction?", [], [], SIZE, profile="math", mode="practice")
    assert "Mode: practice (similar problem). Make ONE new problem like the one on the board" in prompt
    assert "Ink regions detected on the whiteboard: none." in prompt and "Reference context: none." in prompt


def test_task_kind_overrides_session_mode() -> None:
    prompt = build_prompt("q", [], [], SIZE, profile="math", mode="help", task={"kind": "practice"})
    assert "Mode: practice" in prompt
    assert "Mode: help" in build_prompt("q", [], [], SIZE, profile="math", mode="nonsense", task={"kind": "x"})
    assert vc.resolve_mode("quiz", {"kind": "learn"}) == "learn" and vc.resolve_mode("quiz", None) == "quiz"


def test_quiz_dictate_prompt() -> None:
    task = {"kind": "quiz", "stage": "dictate", "number": 2, "total": 5, "problem": "3x + 5 = 20", "answer": "x = 5"}
    prompt = build_prompt("start", [], [], SIZE, profile="math", mode="quiz", task=task)
    assert "Mode: quiz." in prompt and "Quiz stage: dictate. Problem 2 of 5: 3x + 5 = 20" in prompt
    assert "Do not solve it" in prompt and "x = 5" not in prompt  # the expected answer is only shown when checking
    # Asked to invent one, the 3B model echoed the student 12 times out of 12: code picks the problem.
    invented = vc.quiz_task_text({"kind": "quiz", "stage": "dictate", "number": 1, "total": 10, "problem": None})
    problem = invented.split("Problem 1 of 10: ", 1)[1].splitlines()[0]
    assert vc.solve_problem(problem) and "Read this problem out in words" in invented
    assert invented == vc.quiz_task_text({"kind": "quiz", "stage": "dictate", "number": 1, "total": 10}), "stable"


def test_quiz_check_prompt() -> None:
    task = {
        "kind": "quiz",
        "stage": "check",
        "number": 1,
        "total": 5,
        "problem": "3x + 5 = 20",
        "answer": "x = 5",
        "next_problem": "2x - 4 = 10",
    }
    text = vc.quiz_task_text(task)
    assert "Quiz stage: check. Problem 1 of 5: 3x + 5 = 20" in text
    # Shown the next problem, the model copied it as the student's answer: code reads it out after a correct check.
    assert "2x - 4 = 10" not in text and "Next problem" not in text and "do not read out any problem" in text
    assert "board_answer" in text and "never the answer" in text
    # A numeric answer is graded in code; shown to the model, it gets copied into board_answer.
    assert "x = 5" not in text and "Expected answer" not in text
    symbolic = vc.quiz_task_text({**task, "answer": "x = 2 or x = -2"})
    assert "Expected answer" not in symbolic and "x = 2" not in symbolic, "every answer is graded in code now"
    last = vc.quiz_task_text({**task, "number": 5, "next_problem": None})
    assert "last problem" in last and "Next problem" not in last
    unknown = vc.quiz_task_text({**task, "problem": None, "answer": None, "next_problem": None, "number": 3})
    assert "the problem on the board" in unknown and "solve the problem yourself" not in unknown, "code grades it"
    assert "make up" not in unknown, "the next problem is the tutor's, not the model's"
    invented = vc.quiz_task_text({**task, "problem": "4x - 3 = 13", "answer": None})
    assert "Expected answer" not in invented and "x = 4" not in invented, "code solves it, so it is graded in code"
    # The bottom line, and every line of an answer that takes more than one: the model read only "x = 5" of "x = 5"
    # over "x = -3". Asked for "every answer line" instead, it wrote the solution of a problem with no answer yet.
    assert "bottom line of the student's writing" in text and "not a problem from this message" in text
    assert "more than one line (both roots, or x and y), copy each of them" in text and 'joined with " ; "' in text
    spanish = vc.quiz_task_text(task, "es")
    assert "última línea" in spanish and "más de una línea" in spanish and 'unidas con " ; "' in spanish
    for language in ("en", "es"):
        prompt = system_prompt("math", "quiz", language)
        assert '" ; "' in prompt and ("the problem itself if no answer" in prompt or "el mismo problema" in prompt)
    defaults = vc.quiz_task_text({"kind": "quiz", "stage": "check", "number": "x", "total": None})
    assert "Problem 1 of 10" in defaults


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("offer", OFFER_QUESTION),
        ("example", "dictate ONE easy starter problem on solving fractions"),
        ("own", "write it on the board and then ask you for help"),
        ("bogus", OFFER_QUESTION),
    ],
)
def test_build_prompt_learn_stages(stage: str, expected: str) -> None:
    task = {"kind": "learn", "topic": "solving fractions", "stage": stage}
    prompt = build_prompt("teach me", [], [], SIZE, profile="math", mode="quiz", task=task)
    assert prompt.startswith("Mode: learn a topic.")
    assert f"Learn stage: {'offer' if stage == 'bogus' else stage}. Topic: solving fractions" in prompt
    assert expected in prompt and "Quiz stage" not in prompt
    assert "Ink regions detected on the whiteboard: none." in prompt


def test_learn_prompt_without_topic() -> None:
    assert "Topic: this math topic" in vc.learn_task_text({"kind": "learn", "topic": "  ", "stage": "offer"})


def test_recent_turns_in_math_prompt() -> None:
    turns = [{"q": f"q{i}", "a": f"a{i}"} for i in range(5)]
    prompt = build_prompt("now?", [], [], SIZE, profile="math", recent_turns=turns)
    assert (
        "Conversation so far (oldest first):\nStudent: q2\nYou: a2\nStudent: q3\nYou: a3\nStudent: q4\nYou: a4"
        in prompt
    )
    assert "q1" not in prompt
    assert prompt.index("Conversation so far") < prompt.index("Student said: now?")


def test_recent_turns_are_cleaned_and_capped() -> None:
    long = "word " * 200
    text = vc.turns_text(["junk", {"q": "  make\n one   up ", "a": ""}, {"q": None, "a": long}])  # type: ignore[list-item]
    assert text is not None
    assert "Student: make one up" in text and "You: " in text and "junk" not in text
    assert len(text.splitlines()[-1]) <= len("You: ") + vc.MAX_TURN_CHARS and text.endswith("...")
    assert vc.turns_text([]) is None and vc.turns_text(None) is None and vc.turns_text([{"q": "", "a": " "}]) is None


def test_biology_prompt_unchanged_and_takes_recent_turns() -> None:
    legacy = "\n\n".join(
        [
            "User query: why turbid?",
            "Projector size: 1920x1080 pixels (origin top-left).",
            "Ink regions detected on the whiteboard: none.",
            "Reference context: none.",
            "Answer with the JSON object only.",
        ]
    )
    assert build_prompt("why turbid?", [], [], SIZE) == legacy
    assert build_prompt("why turbid?", [], [], SIZE, task={"kind": "quiz"}, mode="quiz") == legacy
    prompt = build_prompt("and then?", [], [], SIZE, recent_turns=[{"q": "why turbid?", "a": "Runoff."}])
    assert prompt.startswith("User query: and then?")
    history = "Conversation so far (oldest first):\nUser: why turbid?\nYou: Runoff.\nEarlier turns are only for"
    assert history in prompt and "judge the CURRENT board" in prompt
    assert prompt.index("Reference context") < prompt.index("Conversation so far"), "after the board and notes"
    assert "Mode:" not in prompt


@pytest.mark.parametrize("query", ["Check my flowchart.", "Is the procedure correct and complete?", "Is this right?"])
def test_biology_check_is_judged_on_the_current_board_only(query: str) -> None:
    """Shown an earlier verdict, the 3B model repeated it: AB 1/6 with the history first, BA 1/6 with it last."""
    turns = [{"q": "Check it.", "a": "Not yet. There is no autoclave step."}]
    assert build_prompt(query, [], [], SIZE, recent_turns=turns) == build_prompt(query, [], [], SIZE)


def test_profile_kwarg_is_normalized() -> None:
    assert VLMClient("mock", "m", "u").profile == "biology"
    assert VLMClient("mock", "m", "u", profile="Biology").profile == "biology"
    assert VLMClient("mock", "m", "u", profile="math").profile == "math"
    assert VLMClient("mock", "m", "u", profile="poetry").profile == "math"


@pytest.mark.usefixtures("no_mathspeak")
def test_math_evaluate_sends_mode_prompt_and_turns() -> None:
    turns = [{"q": "I want to learn fractions", "a": "Do you have an exercise in mind?"}]
    action, payload = _ask(_math_client(), MATH_GOOD, query="another one", mode="practice", recent_turns=turns)
    assert payload["messages"][0] == {"role": "system", "content": system_prompt("math", "practice")}
    user = payload["messages"][1]["content"]
    assert "Mode: practice" in user and "Student said: another one" in user and "Student: I want to learn" in user
    assert payload["options"] == {"temperature": 0.6, "num_predict": 200} and payload["keep_alive"] == "30m"
    assert payload["format"] == "json" and payload["messages"][1]["images"] == [B64]
    # dictating, not pointing
    assert action.to_dict() == AvatarAction("Subtract five from both sides.", "idle", 7, 9).to_dict()
    assert action.display_text == "Subtract five from both sides."  # type: ignore[attr-defined]


@pytest.mark.usefixtures("no_mathspeak")
def test_math_evaluate_task_picks_the_prompt() -> None:
    quiz = {"kind": "quiz", "stage": "check", "number": 1, "total": 3, "problem": "x + 1 = 2", "answer": "x = 1"}
    _, payload = _ask(_math_client(), MATH_GOOD, mode="help", task=quiz)
    assert payload["messages"][0]["content"] == system_prompt("math", "quiz")
    assert "Quiz stage: check. Problem 1 of 3: x + 1 = 2" in payload["messages"][1]["content"]
    assert payload["options"]["temperature"] == 0.2
    _, payload = _ask(_math_client(), MATH_GOOD, mode="quiz", task=LEARN_OFFER)
    assert payload["messages"][0]["content"] == system_prompt("math", "learn")


@pytest.mark.usefixtures("no_mathspeak")
def test_openai_math_uses_math_prompt() -> None:
    c = VLMClient("openai", "m", "https://api.example.com", profile="math")
    body = {"choices": [{"message": {"content": json.dumps(MATH_GOOD)}}]}
    with patch.object(c.session, "post", return_value=_resp(body)) as post:
        c.evaluate("q", None, [], [], SIZE, mode="practice")
    p = post.call_args.kwargs["json"]
    assert p["messages"][0]["content"] == system_prompt("math", "practice") and p["temperature"] == 0.6


def test_biology_evaluate_accepts_new_kwargs_unchanged(fake_mathspeak: list[str]) -> None:
    reply = {**GOOD, "spoken_text": "Plate 2/3 of it."}
    action, payload = _ask(_client("ollama"), reply, mode="quiz", task={"kind": "quiz"}, recent_turns=[{"q": "a"}])
    assert payload["messages"][0]["content"] is SYSTEM_PROMPT and payload["options"]["temperature"] == 0.2
    assert "User: a" in payload["messages"][1]["content"]
    assert action.spoken_text == "Plate 2/3 of it." and fake_mathspeak == []


def test_mathspeak_applied_for_math(fake_mathspeak: list[str]) -> None:
    action, _ = _ask(_math_client(), {**MATH_GOOD, "spoken_text": "You have 2/3 left."})
    assert action.spoken_text == "You have two thirds left." and fake_mathspeak[0] == "You have 2/3 left."


def test_mathspeak_applied_on_mock_backend(fake_mathspeak: list[str]) -> None:
    _math_client("mock").evaluate("what is 2/3?", None, [], [], SIZE)
    assert fake_mathspeak and "two thirds" not in fake_mathspeak[0]


@pytest.mark.usefixtures("no_mathspeak")
def test_mathspeak_missing_is_identity() -> None:
    action, _ = _ask(_math_client(), {**MATH_GOOD, "spoken_text": "You have 2/3 left."})
    assert action.spoken_text == "You have 2/3 left."


class _ExplodingModule(types.ModuleType):
    def __getattr__(self, name: str) -> Any:
        raise RuntimeError("broken module")


@pytest.mark.parametrize(
    "module",
    [
        types.SimpleNamespace(to_speech=lambda text: 1 / 0),
        types.SimpleNamespace(to_speech=lambda text: "   "),
        types.ModuleType(SPEECH_MODULE),  # no to_speech attribute: ImportError
        _ExplodingModule(SPEECH_MODULE),
    ],
)
def test_mathspeak_failures_keep_raw_text(monkeypatch: pytest.MonkeyPatch, module: object) -> None:
    monkeypatch.setitem(sys.modules, SPEECH_MODULE, module)
    action, _ = _ask(_math_client(), {**MATH_GOOD, "spoken_text": "You have 2/3 left."})
    assert action.spoken_text == "You have 2/3 left." and action.animation_state == "pointing"


@pytest.mark.usefixtures("no_mathspeak")
def test_learn_offer_adds_missing_question_and_idles() -> None:
    reply = {
        "spoken_text": "A fraction is part of a whole.",
        "animation_state": "pointing",
        "target_x": 7,
        "target_y": 9,
    }
    action, _ = _ask(_math_client(), reply, query="I want to learn how to solve fractions", task=LEARN_OFFER)
    assert action.spoken_text == f"A fraction is part of a whole. {OFFER_QUESTION}"
    assert action.animation_state == "idle"


def test_apply_task_rules_learn() -> None:
    asked = AvatarAction(f"Fractions are parts. {OFFER_QUESTION}", "idle", 1, 2)
    assert vc.apply_task_rules(asked, "help", LEARN_OFFER) == asked
    lower = AvatarAction("Want me to make one up for you?", "celebrating", 1, 2)
    assert vc.apply_task_rules(lower, "quiz", LEARN_OFFER) == AvatarAction(lower.spoken_text, "idle", 1, 2)
    assert vc.apply_task_rules(AvatarAction("...", "idle", 1, 2), "help", LEARN_OFFER).spoken_text == OFFER_QUESTION
    example = AvatarAction("Try one half plus one fourth.", "pointing", 1, 2)
    learn_example = {**LEARN_OFFER, "stage": "example"}
    assert vc.apply_task_rules(example, "help", learn_example) == AvatarAction(example.spoken_text, "idle", 1, 2)
    failed = vc.failed_action("The AI model took too long to answer.", (2, 4))
    assert vc.apply_task_rules(failed, "help", LEARN_OFFER) is failed
    # The model picking "error" is not a failed call: a lesson offer is still idle and still asks the question.
    chosen = vc.apply_task_rules(AvatarAction("Fractions are parts.", "error", 1, 2), "help", LEARN_OFFER)
    assert chosen == AvatarAction(f"Fractions are parts. {OFFER_QUESTION}", "idle", 1, 2)


def test_apply_task_rules_states() -> None:
    pointing = AvatarAction("Try four x plus seven equals nineteen.", "pointing", 1, 2)
    idle = AvatarAction(pointing.spoken_text, "idle", 1, 2)
    assert vc.apply_task_rules(pointing, "help", None) == pointing  # help points at the equation
    assert vc.apply_task_rules(pointing, "practice", None) == idle
    assert vc.apply_task_rules(pointing, "help", {"kind": "practice"}) == idle
    dictate = {"kind": "quiz", "stage": "dictate", "problem": "4x + 7 = 19"}
    assert _base(vc.apply_task_rules(pointing, "quiz", dictate)) == _base(idle)
    for chosen in ("error", "pointing", "walking"):
        practice = vc.apply_task_rules(replace(pointing, animation_state=chosen), "practice", {"kind": "practice"})
        assert practice == idle, "a practice reply is idle even when the model picks error"


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("x = 5", Fraction(5)),
        ("5", Fraction(5)),
        (" y=-4. ", Fraction(-4)),
        ("x = \u22124", Fraction(-4)),
        ("x = 2/3", Fraction(2, 3)),
        ("0.5", Fraction(1, 2)),
        ("x = 1 1/2", Fraction(3, 2)),
        ("-2 1/4", Fraction(-9, 4)),
        ("3 / 4", Fraction(3, 4)),
        ("3x + 5 = 20", None),
        ("3x = 15", None),
        ("x = 2 or x = -2", None),
        ("x = 5 = 5", None),
        ("x =", None),
        ("five", None),
        ("1/0", None),
        ("", None),
        (None, None),
    ],
)
def test_answer_value(text: str | None, value: Fraction | None) -> None:
    assert vc.answer_value(text) == value


def test_board_answer_extraction() -> None:
    assert vc.board_answer('{"board_answer": "x = 7", "spoken_text": "Hi"}') == "x = 7"
    assert vc.board_answer('```json\n{"board_answer" : " x = \\"7\\" ", "a": 1}\n```') == 'x = "7"'
    assert vc.board_answer('{"board_answer": "", "spoken_text": "Hi"}') is None
    assert vc.board_answer('{"spoken_text": "Hi"}') is None and vc.board_answer(None) is None
    assert vc.board_answer('{"board_answer": "bad \\q escape"}') is None


QUIZ_CHECK = {
    "kind": "quiz",
    "stage": "check",
    "number": 1,
    "total": 5,
    "problem": "3x + 5 = 20",
    "answer": "x = 5",
    "next_problem": "2x - 4 = 10",
}
PRAISE = AvatarAction("Correct, great job! Problem two. Two x minus four equals ten.", "celebrating", 8, 9)
HINT = AvatarAction("Not quite. What can you subtract from both sides?", "error", 8, 9)


def _base(action: AvatarAction) -> AvatarAction:
    return AvatarAction(action.spoken_text, action.animation_state, action.target_x, action.target_y)


def test_quiz_check_keeps_model_text_when_it_agrees() -> None:
    praised = vc.check_quiz_answer(PRAISE, QUIZ_CHECK, "x = 5")
    assert _base(praised) == PRAISE and (praised.verdict, praised.dictated) == ("correct", "2x - 4 = 10")
    hinted = vc.check_quiz_answer(HINT, QUIZ_CHECK, "7")
    assert _base(hinted) == HINT and hinted.verdict == "wrong" and hinted.dictated is None
    symbolic = {**QUIZ_CHECK, "problem": "x^2 = 4", "answer": "x = 2 or x = -2"}
    assert vc.check_quiz_answer(PRAISE, symbolic, "x = 2, x = -2").verdict == "correct", "not a plain value"
    solved = vc.check_quiz_answer(PRAISE, {**QUIZ_CHECK, "answer": None}, "x = 7")
    assert solved.animation_state == "error", "no answer given: code solves 3x + 5 = 20 itself and grades it"


def test_quiz_check_overrides_wrong_praise() -> None:
    fixed = vc.check_quiz_answer(PRAISE, QUIZ_CHECK, "x = 7")
    assert fixed.animation_state == "error" and (fixed.target_x, fixed.target_y) == (8, 9)
    # Code's own hint for this problem's first step ("subtract 5"), never its answer (x = 5) or the next problem.
    assert fixed.spoken_text == "Not quite. Subtract 5 from both sides first." and "2x" not in fixed.spoken_text
    assert not grading.states_solution(fixed.spoken_text, grading.solve_text("3x + 5 = 20"))
    assert fixed.hint_needed and fixed.board_answer == "x = 7"  # type: ignore[attr-defined]
    generic = vc.check_quiz_answer(PRAISE, {**QUIZ_CHECK, "problem": "a word problem", "answer": "x = 5"}, "x = 7")
    assert generic.spoken_text == vc.WRONG_TEXT, "no hint code can build: the generic one"


def test_quiz_check_overrides_wrong_rejection() -> None:
    fixed = vc.check_quiz_answer(HINT, QUIZ_CHECK, "x=5")
    assert _base(fixed) == AvatarAction("Correct, great job! Problem 2. 2x - 4 = 10", "celebrating", 8, 9)
    last = vc.check_quiz_answer(HINT, {**QUIZ_CHECK, "number": 5, "next_problem": None}, "5")
    assert last.spoken_text == "Correct, great job! That was the last problem. You finished the quiz!"
    fraction = vc.check_quiz_answer(HINT, {**QUIZ_CHECK, "answer": "x = 2/3", "next_problem": None}, "x = 4/6")
    assert fraction.animation_state == "celebrating" and fraction.spoken_text == "Correct, great job!"


@pytest.mark.usefixtures("no_mathspeak")
def test_evaluate_grades_the_board_answer_in_code() -> None:
    wrong = {"board_answer": "x = 7", **PRAISE.to_dict()}
    action, _ = _ask(_math_client(), wrong, mode="quiz", task=QUIZ_CHECK)
    assert action.animation_state == "error" and action.spoken_text.startswith("Not quite")
    right = {"board_answer": "x = 5", **HINT.to_dict()}
    action, _ = _ask(_math_client(), right, mode="quiz", task=QUIZ_CHECK)
    assert action.animation_state == "celebrating" and "Problem 2. 2x - 4 = 10" in action.spoken_text
    action, _ = _ask(_math_client(), PRAISE.to_dict(), mode="quiz", task=QUIZ_CHECK)
    # No board_answer: nothing was read off the board, so never approved and never counted as a miss.
    assert action.animation_state == "idle" and action.verdict is None  # type: ignore[attr-defined]
    assert "can't see your answer" in action.spoken_text


def test_evaluate_quiz_override_is_spoken_in_words(fake_mathspeak: list[str]) -> None:
    right = {"board_answer": "x = 5", **HINT.to_dict()}
    action, _ = _ask(_math_client(), right, mode="quiz", task=QUIZ_CHECK)
    assert "Correct, great job! Problem 2. 2x - 4 = 10" in fake_mathspeak
    assert action.spoken_text == "Correct, great job! Problem 2. 2x - 4 = 10", "the fake speaks only 2/3"


def test_raw_evaluate_returns_board_answer() -> None:
    c = _math_client()
    reply = {"board_answer": "x = 7", **PRAISE.to_dict()}
    with patch.object(c.session, "post", return_value=_ollama_resp(json.dumps(reply))):
        raw = c._evaluate("q", None, [], [], SIZE, task=QUIZ_CHECK)
    assert raw.action == PRAISE and raw.board_answer == "x = 7" and not raw.failed and raw.board_problem is None
    assert _math_client("mock")._evaluate("q", None, [], [], SIZE)[1] is None


@pytest.mark.usefixtures("no_mathspeak")
def test_math_failures_still_give_error_action() -> None:
    action, _ = _ask(_math_client(), requests.Timeout("slow"), task=LEARN_OFFER)
    assert action.animation_state == "error" and OFFER_QUESTION not in action.spoken_text
    garbage = _math_client()
    with patch.object(garbage.session, "post", return_value=_ollama_resp("not json")):
        assert garbage.evaluate("q", None, [], [], SIZE, mode="quiz").animation_state == "error"


@pytest.mark.usefixtures("no_mathspeak")
def test_evaluate_ignores_a_malformed_task() -> None:
    action, payload = _ask(_math_client(), MATH_GOOD, mode="quiz", task="quiz")  # type: ignore[arg-type]
    assert "Quiz stage: dictate" in payload["messages"][1]["content"] and action.animation_state == "idle"


@pytest.mark.usefixtures("no_mathspeak")
def test_evaluate_survives_post_processing_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any) -> AvatarAction:
        raise RuntimeError("bug")

    monkeypatch.setattr(vc, "apply_task_rules", boom)
    action, _ = _ask(_math_client(), MATH_GOOD, task=LEARN_OFFER)
    assert action == AvatarAction("Subtract five from both sides.", "pointing", 7, 9)


# --- quiz answers never reach the model (except a non-numeric expected answer in the check stage) ---------------

SEED = Path(__file__).resolve().parent.parent / "biobench_sar" / "knowledge" / "math"


def _seed_chunks() -> list[RetrievedChunk]:
    from biobench_sar.rag.ingestor import ingest_file

    names = ("quiz_bank.csv", "worksheet_linear_equations.md")
    return [RetrievedChunk(c.text, c.source, 0.8, {}) for n in names for c in ingest_file(SEED / n)]


@pytest.mark.parametrize(
    "task",
    [
        None,
        {"kind": "practice"},
        {"kind": "learn", "topic": "equations", "stage": "offer"},
        {"kind": "quiz", "stage": "dictate", "number": 5, "total": 12, "problem": "2x + 3 = 11", "answer": "x = 4"},
        {"kind": "quiz", "stage": "hint", "number": 5, "total": 12, "problem": "2x + 3 = 11", "answer": "x = 4"},
        {"kind": "quiz", "stage": "check", "number": 5, "total": 12, "problem": "2x + 3 = 11", "answer": "x = 4"},
    ],
)
def test_seeded_answers_are_stripped_from_math_prompts(task: dict[str, Any] | None) -> None:
    chunks = _seed_chunks()
    assert any("Answer: x = 4" in c.text for c in chunks) and any("answer is x = 5" in c.text for c in chunks)
    prompt = build_prompt("I got x = 7", chunks, [], SIZE, profile="math", mode="help", task=task)
    assert "2x + 3 = 11" in prompt and "Reference context:" in prompt, "the problems stay"
    assert not re.search(r"\banswer\s*(?:is|:)\s*x", prompt, re.I), "no worksheet answer line or CSV answer column"
    assert "Answer: x = 4" not in prompt and "answer is x = 5" not in prompt


def test_biology_context_is_not_stripped() -> None:
    chunk = RetrievedChunk("Answer: autoclave first.", "notes.md", 0.9, {})
    assert "Answer: autoclave first." in build_prompt("q", [chunk], [], SIZE)


def test_context_of_only_answers_is_none() -> None:
    chunk = RetrievedChunk("Answer: x = 4", "w.md", 0.9, {})
    assert "Reference context: none." in build_prompt("q", [chunk], [], SIZE, profile="math")


# --- quiz stages --------------------------------------------------------------------------------------------

QUIZ_HINT = {**QUIZ_CHECK, "stage": "hint", "next_problem": None}


def test_quiz_hint_prompt_has_the_problem_but_not_the_answer() -> None:
    prompt = build_prompt("I'm stuck", [], REGIONS, SIZE, profile="math", mode="quiz", task=QUIZ_HINT)
    assert "Quiz stage: hint. Problem 1 of 5: 3x + 5 = 20" in prompt and "NEVER say the answer" in prompt
    assert "x = 5" not in prompt and "board_answer" not in vc.quiz_task_text(QUIZ_HINT)
    assert "Stage hint" in system_prompt("math", "quiz")


@pytest.mark.usefixtures("no_mathspeak")
def test_quiz_hint_points_and_is_never_graded() -> None:
    reply = {
        "spoken_text": "Subtract five from both sides first.",
        "animation_state": "error",
        "target_x": 7,
        "target_y": 9,
    }
    action, _ = _ask(_math_client(), reply, query="I'm stuck", mode="quiz", task=QUIZ_HINT)
    assert action.animation_state == "pointing" and getattr(action, "verdict", None) is None


def test_quiz_hint_that_gives_the_answer_is_asked_again() -> None:
    c = _math_client()
    leak = {"spoken_text": "So x equals five.", "animation_state": "pointing", "target_x": 7, "target_y": 9}
    good = {**leak, "spoken_text": "Subtract five from both sides first."}
    replies = [_ollama_resp(json.dumps(leak)), _ollama_resp(json.dumps(good))]
    with patch.object(c.session, "post", side_effect=replies) as post:
        action = c.evaluate("I'm stuck", JPEG, [], REGIONS, SIZE, mode="quiz", task=QUIZ_HINT)
    assert post.call_count == 2 and "gave away the value" in post.call_args.kwargs["json"]["messages"][1]["content"]
    assert action.spoken_text == "Subtract five from both sides first."


@pytest.mark.usefixtures("no_mathspeak")
def test_quiz_dictation_echo_is_replaced_by_the_problem() -> None:
    task = {"kind": "quiz", "stage": "dictate", "number": 2, "total": 5, "problem": "2x + 3 = 11"}
    for spoken in (
        "Quiz me.",
        "Problem one of five. What is a fraction?",
        "Two x plus three equals eleven, x is four.",
    ):
        reply = {"spoken_text": spoken, "animation_state": "pointing", "target_x": 7, "target_y": 9}
        action, _ = _ask(_math_client(), reply, query="Quiz me.", mode="quiz", task=task)
        assert action.spoken_text == "Problem 2 of 5. 2x + 3 = 11. Write your answer on the board.", spoken
        assert action.animation_state == "idle" and action.dictated == "2x + 3 = 11"  # type: ignore[attr-defined]
    good = {"spoken_text": "Problem two. 2x + 3 = 11. Write it down.", "animation_state": "idle", "target_x": 1}
    action, _ = _ask(_math_client(), good, query="Quiz me.", mode="quiz", task=task)
    assert action.spoken_text == good["spoken_text"], "a reading of the right problem is kept"


@pytest.mark.usefixtures("no_mathspeak")
def test_quiz_without_a_problem_dictates_one_picked_in_code() -> None:
    task = {"kind": "quiz", "stage": "dictate", "number": 1, "total": 10, "problem": None, "answer": None}
    echo = {"spoken_text": "Start a quiz.", "animation_state": "idle", "target_x": 1, "target_y": 1}
    action, payload = _ask(_math_client(), echo, query="Start a quiz.", mode="quiz", task=task)
    problem = action.dictated  # type: ignore[attr-defined]
    assert problem and vc.solve_problem(problem) and problem in payload["messages"][1]["content"]
    assert action.spoken_text == f"Problem 1 of 10. {problem}. Write your answer on the board."


def test_quiz_check_echo_and_leaks_are_replaced() -> None:
    hint = "Not quite. Subtract 5 from both sides first."
    echo = AvatarAction("I'm stuck.", "error", 1, 2)
    assert vc.check_quiz_answer(echo, QUIZ_CHECK, "x = 7", query="I'm stuck.").spoken_text == hint
    leak = AvatarAction("Not quite, x should be five.", "error", 1, 2)
    assert vc.check_quiz_answer(leak, QUIZ_CHECK, "x = 7").spoken_text == hint
    stated = AvatarAction("Not quite. Fifteen divided by three is five, so x is 5.", "error", 1, 2)
    assert vc.check_quiz_answer(stated, QUIZ_CHECK, "x = 7").spoken_text == hint
    reads_next = AvatarAction("Not quite. Problem two. 2x - 4 = 10", "error", 1, 2)
    assert vc.check_quiz_answer(reads_next, QUIZ_CHECK, "x = 7").spoken_text == hint
    praise = AvatarAction("Correct, great job!", "error", 1, 2)
    assert vc.check_quiz_answer(praise, QUIZ_CHECK, "x = 7").spoken_text == hint, "praise for a wrong answer"


def test_quiz_check_praise_without_the_next_problem_gets_it_appended() -> None:
    short = AvatarAction("Correct, well done!", "celebrating", 1, 2)
    fixed = vc.check_quiz_answer(short, QUIZ_CHECK, "x = 5")
    assert fixed.spoken_text == "Correct, well done! Problem 2. 2x - 4 = 10" and fixed.dictated == "2x - 4 = 10"
    worded = AvatarAction("Correct! Problem two. Two x minus four equals ten.", "celebrating", 1, 2)
    assert vc.check_quiz_answer(worded, QUIZ_CHECK, "x = 5").spoken_text == worded.spoken_text


@pytest.mark.parametrize("read", [None, "", "3x + 5 = 20", "3x = 15", "3 x plus five equals twenty"])
def test_quiz_check_praise_without_a_readable_answer_is_not_approved(read: str | None) -> None:
    praise = AvatarAction("Correct, great job!", "celebrating", 1, 2)
    action = vc.check_quiz_answer(praise, QUIZ_CHECK, read)
    assert action.animation_state == "idle" and action.verdict is None and action.spoken_text == vc.NO_ANSWER_TEXT


def _arithmetic_check(problem: str) -> dict[str, Any]:
    return {"kind": "quiz", "stage": "check", "number": 1, "total": 5, "problem": problem, "answer": None}


@pytest.mark.parametrize(
    ("problem", "read"),
    [
        ("1/2 + 1/3", "1/2 + 1/3"),  # only the problem on the board: solved in code it was "correct"
        ("3/4 - 1/2", "3/4 - 1/2"),
        ("12 ÷ 4", "12 ÷ 4"),
        ("2/3 × 3/4", "2/3 × 3/4"),
        ("0.5 + 0.25", "0.5 + 0.25"),
        ("1/5 + 2/5", "1/5 + 2/5"),  # the seeded starter
        ("1/2 + 1/3", "2/6 + 3/6"),  # worked part of the way
        ("1/2 + 1/3", "one half plus one third"),
        ("12 ÷ 4 = ?", "12 ÷ 4 ="),
        ("Simplify 6/8", "6/8"),  # the problem copied back, though it reads as one value
    ],
)
def test_quiz_check_of_arithmetic_not_worked_out_is_not_graded(problem: str, read: str) -> None:
    task = _arithmetic_check(problem)
    for language in ("en", "es"):
        action = vc.check_quiz_answer(CHEER, task, read, language=language)
        assert action.verdict is None and action.animation_state == "idle"  # type: ignore[attr-defined]
        assert action.spoken_text == vc.TEXTS[language]["unfinished"]
        assert not grading.states_solution(action.spoken_text, vc.expected_answers(task), language)


@pytest.mark.parametrize(
    ("read", "verdict"),
    [
        ("5/6", "correct"),
        ("1/2 + 1/3 = 5/6", "correct"),
        ("2/6 + 3/6 = 5/6", "correct"),
        ("10/12", "correct"),
        ("1/2 + 1/3 = 2/5", "wrong"),
        ("2/5", "wrong"),
    ],
)
def test_quiz_check_of_worked_arithmetic_is_graded(read: str, verdict: str) -> None:
    action = vc.check_quiz_answer(CHEER, _arithmetic_check("1/2 + 1/3"), read)
    assert action.verdict == verdict  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("problem", "answer", "check"),
    [
        ("3x + 5 = 20", "x = 5", "3(5) + 5 = 20"),
        ("3x + 5 = 20", "x = 5", "15 + 5 = 20"),
        ("2x - 7 = 9", "x = 8", "2(8) - 7 = 9"),
        ("x + 7 = 12", "x = 5", "5 + 7 = 12"),
        ("5x + 4 = -11", "x = -3", "5(-3) + 4 = -11"),
    ],
)
def test_quiz_check_of_a_substitution_check_on_the_bottom_line_is_never_wrong(
    problem: str, answer: str, check: str
) -> None:
    task = {**QUIZ_CHECK, "problem": problem, "answer": answer}
    for language in ("en", "es"):
        for model in (PRAISE, HINT):
            alone = vc.check_quiz_answer(model, task, check, language=language)
            assert alone.verdict is None and alone.animation_state == "idle", "no point, no miss"  # type: ignore[attr-defined]
            assert alone.spoken_text == vc.TEXTS[language]["reread"] and not getattr(alone, "hint_needed", False)
        both = vc.check_quiz_answer(PRAISE, task, f"{answer} ; {check}", language=language)
        assert both.verdict == "correct", "with the answer line copied too, that line is graded"  # type: ignore[attr-defined]
        other = {**task, "answer": "x = 99"}
        assert vc.check_quiz_answer(PRAISE, other, f"{answer} ; {check}", language=language).verdict == "wrong"  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("problem", "answer", "check"),
    [
        ("Despeja x: 3x + 5 = 20", "x = 5", "3(5) + 5 = 20"),
        ("2x - 7 = 9. Find x.", "x = 8", "2(8) - 7 = 9"),
        ("Encuentra el valor de x: 5x + 4 = -11", "x = -3", "5(-3) + 4 = -11"),
        ("Resuelve la ecuación 7 - 2x = 1", "x = 3", "7 - 2(3) = 1"),
        ("A farmer has some hens. Three times the hens plus 5 is 20. How many?", "h = 5", "3(5) + 5 = 20"),
        ("A farmer has some hens. Three times the hens plus 5 is 20. How many?", "5", "3(5) + 5 = 20"),
    ],
)
def test_quiz_check_of_a_substitution_check_under_a_worded_problem_is_never_wrong(
    problem: str, answer: str, check: str
) -> None:
    task = {**QUIZ_CHECK, "problem": problem, "answer": answer}
    for language in ("en", "es"):
        for model in (PRAISE, HINT):
            alone = vc.check_quiz_answer(model, task, check, language=language)
            assert alone.verdict is None and alone.animation_state == "idle", (problem, language)  # type: ignore[attr-defined]
            assert alone.spoken_text == vc.TEXTS[language]["reread"] and not getattr(alone, "hint_needed", False)
        both = vc.check_quiz_answer(PRAISE, task, f"{answer} ; {check}", language=language)
        assert both.verdict == "correct", (problem, language)  # type: ignore[attr-defined]


@pytest.mark.parametrize("mark", [" ✓", " ✔", " √", " true", " ok", " correct", " sí", " ✅", " →", " :)"])
def test_quiz_check_of_a_check_line_with_a_tick_after_it_is_never_wrong(mark: str) -> None:
    for check in ("3(5) + 5 = 20", "15 + 5 = 20", "20 = 20"):
        for language in ("en", "es"):
            action = vc.check_quiz_answer(HINT, QUIZ_CHECK, check + mark, language=language)
            assert action.verdict is None and action.spoken_text == vc.TEXTS[language]["reread"], (check, mark)  # type: ignore[attr-defined]
            both = vc.check_quiz_answer(PRAISE, QUIZ_CHECK, f"x = 5 ; {check}{mark}", language=language)
            assert both.verdict == "correct"  # type: ignore[attr-defined]
    fraction = vc.check_quiz_answer(PRAISE, _arithmetic_check("1/2 + 1/3"), "1/2 + 1/3 = 5/6" + mark)
    assert fraction.verdict == "correct", "the arithmetic problem worked out, ticked"  # type: ignore[attr-defined]


def test_a_check_under_a_worded_problem_asks_again_for_the_answer_line() -> None:
    # Live, problem "3x + 5 = 20. Find x.": the first read was the check line in 2/2 runs, graded as a miss.
    task = {**QUIZ_CHECK, "problem": "3x + 5 = 20. Find x."}
    c = _math_client()
    replies = _check_replies("3(5) + 5 = 20", "x = 5 ; 3(5) + 5 = 20", spoken="Not quite.", state="error")
    with patch.object(c.session, "post", side_effect=replies) as post:
        action = c.evaluate("Check my answer.", JPEG, [], REGIONS, SIZE, mode="quiz", task=task)
    assert post.call_count == 2
    assert vc.RETRY_NOTES["en"]["check_line"] in post.call_args.kwargs["json"]["messages"][1]["content"]
    assert action.verdict == "correct"  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("problem", "read", "verdict"),
    [
        ("7 - 2x = 1", "3 = x", "correct"),
        ("7 - 2x = 1", "6 = 2x\n3 = x", "correct"),
        ("7 - 2x = 1", "6 = 2x ; 3 = x", "correct"),
        ("7 - 2x = 1", "4 = x", "wrong"),
        ("2(3x - 4) = 4x + 6", "7 = x", "correct"),
    ],
)
def test_quiz_check_of_an_answer_with_the_letter_on_the_right_is_graded(problem: str, read: str, verdict: str) -> None:
    task = {**QUIZ_CHECK, "problem": problem, "answer": None}
    for language in ("en", "es"):
        action = vc.check_quiz_answer(PRAISE, task, read, language=language)
        assert action.verdict == verdict, (read, language)  # type: ignore[attr-defined]
        assert action.spoken_text != vc.TEXTS[language]["unfinished"]


def test_quiz_check_of_a_step_with_the_letter_on_the_right_is_no_answer_yet() -> None:
    task = {**QUIZ_CHECK, "problem": "7 - 2x = 1", "answer": "x = 3"}
    for language in ("en", "es"):
        action = vc.check_quiz_answer(PRAISE, task, "6 = 2x", language=language)
        assert action.verdict is None and action.spoken_text == vc.TEXTS[language]["no_answer"]  # type: ignore[attr-defined]


def test_evaluate_of_a_substitution_check_records_no_miss() -> None:
    reply = {"board_answer": "3(5) + 5 = 20", **CHEER.to_dict()}
    action, _ = _ask(_math_client(), reply, query="Check my answer.", mode="quiz", task=QUIZ_CHECK)
    assert action.verdict is None and action.animation_state == "idle"  # type: ignore[attr-defined]
    assert action.spoken_text == vc.TEXTS["en"]["reread"]


@pytest.mark.parametrize(
    ("language", "query", "praise"),
    [("en", "Check my answer.", "Correct, great job!"), ("es", "Revisa mi respuesta.", "¡Correcto, muy bien!")],
)
def test_a_check_that_read_only_the_check_line_asks_again_for_the_answer_line(
    language: str, query: str, praise: str
) -> None:
    # Live, qwen2.5vl:3b copied only the bottom line "3(5) + 5 = 20" of "x = 5" over its check, every time.
    c = _math_client()
    replies = _check_replies("3(5) + 5 = 20", "x = 5 ; 3(5) + 5 = 20", spoken=praise)
    with patch.object(c.session, "post", side_effect=replies) as post:
        action = c.evaluate(query, JPEG, [], REGIONS, SIZE, mode="quiz", task=QUIZ_CHECK, language=language)
    assert post.call_count == 2
    assert vc.RETRY_NOTES[language]["check_line"] in post.call_args.kwargs["json"]["messages"][1]["content"]
    assert action.verdict == "correct"  # type: ignore[attr-defined]


def test_a_check_still_reading_only_the_check_line_is_not_graded() -> None:
    c = _math_client()
    replies = _check_replies("3(5) + 5 = 20", "15 + 5 = 20", spoken="Correct, great job!")
    with patch.object(c.session, "post", side_effect=replies) as post:
        action = c.evaluate("Check my answer.", JPEG, [], REGIONS, SIZE, mode="quiz", task=QUIZ_CHECK)
    assert post.call_count == 2, "asked again once, never more"
    assert action.verdict is None and action.spoken_text == vc.TEXTS["en"]["reread"]  # type: ignore[attr-defined]
    for read in ("x = 5", "x = 4", "3x = 15", "x = 5 ; 3(5) + 5 = 20"):
        assert not grading.only_checks(read, QUIZ_CHECK["problem"]), read


def test_evaluate_never_approves_the_problem_read_back() -> None:
    reply = {"board_answer": "1/2 + 1/3", **CHEER.to_dict()}
    action, _ = _ask(_math_client(), reply, query="Check my answer.", mode="quiz", task=_arithmetic_check("1/2 + 1/3"))
    assert action.verdict is None and action.animation_state == "idle"  # type: ignore[attr-defined]
    assert "final answer" in action.spoken_text


@pytest.mark.parametrize("read", ["x = something", "I can't tell", "x = 2y"])
def test_quiz_check_of_an_unreadable_answer_asks_again(read: str) -> None:
    praise = AvatarAction("Correct, great job!", "celebrating", 1, 2)
    action = vc.check_quiz_answer(praise, QUIZ_CHECK, read)
    assert action.animation_state == "idle" and action.verdict is None  # type: ignore[attr-defined]
    assert action.spoken_text == "I couldn't read your answer clearly. Ask me to check it again."
    spanish = vc.check_quiz_answer(praise, QUIZ_CHECK, read, language="es")
    assert spanish.spoken_text == "No pude leer bien tu respuesta. Pídeme que la revise otra vez."


def test_quiz_check_reads_an_answer_written_in_words() -> None:
    praise = AvatarAction("Correct, great job!", "celebrating", 1, 2)
    assert vc.check_quiz_answer(praise, QUIZ_CHECK, "five").verdict == "correct"  # type: ignore[attr-defined]
    es = vc.check_quiz_answer(praise, QUIZ_CHECK, "x es igual a cinco", language="es")
    assert es.verdict == "correct"  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("reply", "read"),
    [
        ('{"board_answer": 7, "spoken_text": "Hi"}', "7"),
        ('{"board_answer": -2.5, "spoken_text": "Hi"}', "-2.5"),
        ('{"board_answer": true}', None),
        ('{"board_answer": ["x = 7"]}', None),
        ('{"board_answer": 7, "spoken_text": "cut off', "7"),
        ('{"board_answer": "x = 7", "spoken_text": "cut off', "x = 7"),
    ],
)
def test_board_answer_accepts_numbers(reply: str, read: str | None) -> None:
    assert vc.board_answer(reply) == read


@pytest.mark.usefixtures("no_mathspeak")
def test_numeric_board_answer_is_graded_in_code() -> None:
    wrong = {"board_answer": 7, **PRAISE.to_dict()}
    action, _ = _ask(_math_client(), wrong, mode="quiz", task=QUIZ_CHECK)
    assert action.animation_state == "error" and action.verdict == "wrong"  # type: ignore[attr-defined]


# --- failed calls are marked, the model's own choices are not ------------------------------------------------


@pytest.mark.parametrize("profile", ["math", "biology"])
def test_failed_calls_are_marked(profile: str) -> None:
    c = VLMClient("ollama", "m", "http://x:11434", profile=profile)
    for error in (requests.Timeout("slow"), requests.ConnectionError("down")):
        with patch.object(c.session, "post", side_effect=error):
            action = c.evaluate("q", None, [], [], SIZE, mode="quiz", task=QUIZ_CHECK)
        assert action.animation_state == "error" and action.failed  # type: ignore[attr-defined]
        assert set(action.to_dict()) == {"spoken_text", "animation_state", "target_x", "target_y"}
    with patch.object(c.session, "post", return_value=_ollama_resp("not json")):
        assert c.evaluate("q", None, [], [], SIZE).failed  # type: ignore[attr-defined]
    with patch.object(c.session, "post", return_value=_ollama_resp(json.dumps(MATH_GOOD))):
        assert not getattr(c.evaluate("q", None, [], [], SIZE), "failed", False)
    assert VLMClient("bogus", "m", "u", profile=profile).evaluate("q", None, [], [], SIZE).failed  # type: ignore[attr-defined]


# --- conversation memory: never copy an earlier reply ----------------------------------------------------------

TURNS = [{"q": "I'm stuck", "a": "Subtract five from both sides, then write that step."}]


def test_repeat_of_the_previous_answer_is_asked_again() -> None:
    c = _math_client()
    same = {**MATH_GOOD, "spoken_text": "Subtract five from both sides, then write that step."}
    new = {**MATH_GOOD, "spoken_text": "Now divide both sides by three."}
    with patch.object(c.session, "post", side_effect=[_ollama_resp(json.dumps(same)), _ollama_resp(json.dumps(new))]):
        action = c.evaluate("what now?", JPEG, [], REGIONS, SIZE, recent_turns=TURNS)
    assert action.spoken_text == "Now divide both sides by three."
    assert "Never copy an earlier reply" in system_prompt("math", "help")


def test_repeat_after_the_retry_is_trimmed_or_replaced() -> None:
    c = _math_client()
    same = {**MATH_GOOD, "spoken_text": "Subtract five from both sides, then write that step."}
    mixed = {**MATH_GOOD, "spoken_text": "Subtract five from both sides, then write that step. Good work so far!"}
    with patch.object(c.session, "post", side_effect=[_ollama_resp(json.dumps(same))] * 2):
        action = c.evaluate("what now?", JPEG, [], REGIONS, SIZE, recent_turns=TURNS)
    assert action.spoken_text == vc.STEP_TEXT.replace("x", "x"), "nothing new left: the task's fallback"
    with patch.object(c.session, "post", side_effect=[_ollama_resp(json.dumps(mixed))] * 2):
        action = c.evaluate("what now?", JPEG, [], REGIONS, SIZE, recent_turns=TURNS)
    assert action.spoken_text == "Good work so far!", "the copied sentence is dropped"


def test_help_reply_giving_the_value_is_asked_again() -> None:
    c = _math_client()
    leak = {**MATH_GOOD, "spoken_text": "The answer is 5."}
    with patch.object(c.session, "post", side_effect=[_ollama_resp(json.dumps(leak))] * 2) as post:
        action = c.evaluate("just tell me x", JPEG, [], REGIONS, SIZE)
    assert post.call_count == 2 and "five" not in action.spoken_text.lower() and "5" not in action.spoken_text


def test_echo_of_the_question_is_asked_again() -> None:
    c = _math_client()
    echo = {**MATH_GOOD, "spoken_text": "What is a fraction?"}
    good = {**MATH_GOOD, "spoken_text": "A fraction is a part of a whole.", "animation_state": "idle"}
    with patch.object(c.session, "post", side_effect=[_ollama_resp(json.dumps(echo)), _ollama_resp(json.dumps(good))]):
        action = c.evaluate("What is a fraction?", None, [], [], SIZE)
    assert action.spoken_text == "A fraction is a part of a whole."


def test_good_replies_are_not_asked_again() -> None:
    c = _math_client()
    with patch.object(c.session, "post", return_value=_ollama_resp(json.dumps(MATH_GOOD))) as post:
        c.evaluate("what now?", JPEG, [], REGIONS, SIZE, recent_turns=TURNS)
    assert post.call_count == 1


@pytest.mark.parametrize(
    ("spoken", "reveals"),
    [
        ("So x equals five.", True),
        ("x = 5", True),
        ("The answer is 5.", True),
        ("Not quite, x should be seven.", True),
        ("Yes, x is negative four!", True),
        ("Three x equals fifteen. Now divide.", False),
        ("Subtract five from both sides.", False),
        ("Is x equal to five?", False),
        ("If x equals five, what is three x?", False),
        # the letter's value said another way
        ("x turns out to be five.", True),
        ("x ends up being five.", True),
        ("That makes x five.", True),
        ("x has to be five.", True),
        ("The roots are five and negative three.", True),
        ("What makes x alone on one side?", False),
        ("Divide both sides by three.", False),
        ("What do you get when you divide fifteen by three?", False),
    ],
)
def test_reveals_value(spoken: str, reveals: bool) -> None:
    assert vc.reveals_value(spoken) is reveals


@pytest.mark.parametrize(
    ("spoken", "problem", "reads"),
    [
        ("Problem two. Two x minus four equals ten.", "2x - 4 = 10", True),
        ("2x - 4 = 10", "2x - 4 = 10", True),
        ("Problem two. Two x plus four equals ten.", "2x - 4 = 10", False),
        ("One fourth plus two fourths.", "1/4 + 2/4", True),
        ("One quarter plus two quarters.", "1/4 + 2/4", True),
        ("x divided by five equals four.", "x/5 = 4", True),
        ("Correct, great job!", "2x - 4 = 10", False),
        ("anything", None, False),
    ],
)
def test_reads_problem(spoken: str, problem: str | None, reads: bool) -> None:
    assert vc.reads_problem(spoken, problem) is reads


# --- lessons and practice ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spoken",
    [
        "Fractions are parts of a whole. Do you have a problem in mind, or should I create one for you?",
        "Fractions are parts of a whole. Do you have an exercise you want to try?",
    ],
)
def test_offer_question_in_other_words_is_not_asked_twice(spoken: str) -> None:
    action = vc.apply_task_rules(AvatarAction(spoken, "idle", 1, 2), "help", LEARN_OFFER)
    assert action.spoken_text == spoken


def test_offer_question_replaces_another_closing_question() -> None:
    action = vc.apply_task_rules(AvatarAction("Fractions are parts. Ready?", "idle", 1, 2), "help", LEARN_OFFER)
    assert action.spoken_text == f"Fractions are parts. {OFFER_QUESTION}"


@pytest.mark.parametrize(
    ("task", "spoken", "said"),
    [
        (
            {**LEARN_OFFER, "stage": "example"},
            "Let's start easy. One half plus one fourth equals three quarters. Write it down.",
            "Let's start easy. One half plus one fourth. Write it down.",
        ),
        (
            {"kind": "practice"},
            "Try this one. One half minus one third equals one sixth.",
            "Try this one. One half minus one third.",
        ),
        (
            {"kind": "practice"},
            "Try this. Four x plus seven equals nineteen.",
            "Try this. Four x plus seven equals nineteen.",
        ),
    ],
)
def test_dictated_problems_lose_the_result_the_model_added(task: dict[str, Any], spoken: str, said: str) -> None:
    assert vc.apply_task_rules(AvatarAction(spoken, "idle", 1, 2), "help", task).spoken_text == said


PRACTICE = {"kind": "practice"}
CENTER = {"animation_state": "idle", "target_x": 960, "target_y": 540}


def _practice(c: VLMClient, *replies: dict[str, Any]) -> tuple[AvatarAction, Any]:
    with patch.object(c.session, "post", side_effect=[_ollama_resp(json.dumps(r)) for r in replies]) as post:
        action = c.evaluate("another one like this", JPEG, [], REGIONS, SIZE, mode="practice", task=PRACTICE)
    return action, post


def test_practice_reading_the_board_back_is_asked_again() -> None:
    back = {"board_problem": "3/4 - 1/2", "spoken_text": "Try this one. Three fourths minus one half.", **CENTER}
    new = {**back, "spoken_text": "Try this one. Five sixths minus one third."}
    action, post = _practice(_math_client(), back, new)
    assert post.call_count == 2 and "read the board's problem (3/4 - 1/2) back" in str(post.call_args)
    assert action.spoken_text == "Try this one. Five sixths minus one third."


def test_practice_reading_the_board_back_twice_gets_a_problem_made_in_code() -> None:
    back = {"board_problem": "3x + 5 = 20", "spoken_text": "Try three x plus five equals twenty.", **CENTER}
    action, _ = _practice(_math_client(), back, back)
    assert action.spoken_text.startswith("Try this one.") and "equals" in action.spoken_text
    assert not vc.reads_back(action.spoken_text, "3x + 5 = 20") and action.animation_state == "idle"


def test_practice_without_a_whole_problem_is_asked_again() -> None:
    half = {"board_problem": "3x + 5 = 20", "spoken_text": "Five times something plus six. Can you solve it?", **CENTER}
    full = {**half, "spoken_text": "Try this one. Four x plus three equals fifteen."}
    action, post = _practice(_math_client(), half, full)
    assert post.call_count == 2 and action.spoken_text == "Try this one. Four x plus three equals fifteen."


def test_practice_new_problem_is_not_asked_again() -> None:
    good = {"board_problem": "3x + 5 = 20", "spoken_text": "Try this one. Four x plus seven equals nineteen.", **CENTER}
    _, post = _practice(_math_client(), good)
    assert post.call_count == 1
    assert '"board_problem": str' in system_prompt("math", "practice")


@pytest.mark.parametrize(
    ("spoken", "problem", "back"),
    [
        ("Try this one. Three fourths minus one half.", "3/4 - 1/2", True),
        ("Try this one. Three quarters minus one half.", "3/4 - 1/2", True),
        ("Try this one. Three fourths plus one half.", "3/4 - 1/2", False),
        ("Two x plus three equals eleven.", "3x + 2 = 11", False),
        ("anything", "", False),
    ],
)
def test_reads_back(spoken: str, problem: str, back: bool) -> None:
    assert vc.reads_back(spoken, problem) is back


# --- the offline mock backend follows the task ------------------------------------------------------------------


def test_mock_backend_runs_a_whole_quiz() -> None:
    c = _math_client("mock")
    dictate = {**QUIZ_CHECK, "stage": "dictate", "next_problem": None}
    first = c.evaluate("quiz me", None, [], [], SIZE, mode="quiz", task=dictate)
    assert first.animation_state == "idle" and "three x plus five equals twenty" in first.spoken_text
    check = c.evaluate("check", None, [], REGIONS, SIZE, mode="quiz", task=QUIZ_CHECK)
    assert check.animation_state == "celebrating" and check.verdict == "correct"  # type: ignore[attr-defined]
    assert "two x minus four equals ten" in check.spoken_text and check.dictated == "2x - 4 = 10"  # type: ignore[attr-defined]
    hint = c.evaluate("I'm stuck", None, [], REGIONS, SIZE, mode="quiz", task=QUIZ_HINT)
    assert hint.animation_state == "pointing"
    offer = c.evaluate("teach me", None, [], [], SIZE, task=LEARN_OFFER)
    assert offer.animation_state == "idle" and OFFER_QUESTION in offer.spoken_text
    practice = c.evaluate("another", None, [], REGIONS, SIZE, mode="practice", task={"kind": "practice"})
    assert practice.animation_state == "idle" and "equals" in practice.spoken_text
    plain = c.evaluate("what is this?", None, [], REGIONS, SIZE)
    assert plain.animation_state == "pointing" and "I heard" in plain.spoken_text


def test_mock_backend_biology_is_unchanged() -> None:
    action = _client("mock").evaluate("q", None, [], REGIONS, SIZE, mode="quiz", task=QUIZ_CHECK)
    assert action.animation_state == "pointing" and action.spoken_text.startswith('I heard: "q".')


# --- scripts/bench_vlm.py -----------------------------------------------------------------------------------


def _bench(monkeypatch: pytest.MonkeyPatch) -> Any:
    spec = importlib.util.spec_from_file_location("bench_vlm", BENCH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "bench_vlm", module)
    monkeypatch.setattr(sys, "path", list(sys.path))  # the script prepends the repo root
    spec.loader.exec_module(module)
    return module


def test_bench_script_runs_from_any_directory(tmp_path: Path) -> None:
    out = subprocess.run(
        [sys.executable, str(BENCH), "--help"], cwd=tmp_path, capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 0, out.stderr
    assert "--profile" in out.stdout and "--raw" in out.stdout


def test_bench_math_heuristics(monkeypatch: pytest.MonkeyPatch) -> None:
    b = _bench(monkeypatch)
    leaks = ("So x equals five.", "x = 5", "Fifteen divided by three is five.", "The answer is 5.", "x would be 5")
    # Solved in code (grading.py): "x is fifteen over three" says x is 5, asked or not.
    for leak in (*leaks, "x is fifteen over three?", "You get x = 15/3."):
        assert b.leaks_answer(leak), leak
    fine_replies = (
        "Subtract five from both sides.",
        "Three x plus five equals twenty.",
        "What is five less than twenty?",
        "Next, what is 3x equal to?",
    )
    for fine in fine_replies:
        assert not b.leaks_answer(fine), fine
    assert b.dictates_new_equation("Try this. Four x plus seven equals nineteen.")
    assert not b.dictates_new_equation("Three x plus five equals twenty. Solve it.")
    assert b.reads_next_problem("Correct! Problem two. Two x minus four equals ten.")
    assert b.reads_next_problem("Problem two. Two times x minus four equals ten.") and b.reads_next_problem(
        "2x - 4 = 10"
    )
    assert not b.reads_next_problem("Correct, great job!")
    assert b.asks_offer(OFFER_QUESTION) and not b.asks_offer("Fractions are parts of a whole.")
    assert b.dictates_fraction("Try one half plus one fourth.") and not b.dictates_fraction("Try x plus two.")
    assert b.has_notation("x^2 = 4") and b.has_notation("x plus 2") and not b.has_notation("x squared")


def test_bench_math_scenarios_are_well_formed(monkeypatch: pytest.MonkeyPatch) -> None:
    b = _bench(monkeypatch)
    assert set(b.MATH_SCENARIOS) == {
        *("H", "HA", "P", "PF", "QD", "QH", "QC", "QW", "QN", "QDI", "QCI", "QWI", "LO", "LE", "HF"),
        *("ELO", "ELE", "EH", "EHA", "EQD", "EQC", "EQW"),
        *("CR", "CRW", "CRH", "CRHW"),
        *("EQCN", "EQWN", "QCN", "QWN", "HS", "EHS"),
        *("QQC", "QQW", "QQP", "QSC", "QSW", "EQQW"),
        *("QQ2", "EQQ2", "QFN", "QFC"),
        *("QK", "QT", "QXR", "HR", "EHR", "QS2", "EQS2"),
    }
    assert all(b.MATH_SCENARIOS[k].language == "es" for k in ("ELO", "ELE", "EH", "EHA", "EQD", "EQC", "EQW"))
    assert set(b.BIO_SCENARIOS) == {"A", "B", "AB", "BA"}
    assert b.BIO_SCENARIOS["AB"].turns and b.BIO_SCENARIOS["BA"].turns
    follow_up = b.MATH_SCENARIOS["HF"]
    assert follow_up.turns and follow_up.check("Add them to get three fourths.") == "gave the final answer"
    assert follow_up.check("Find a common denominator first. What do two and four both go into?") is None
    offer = b.MATH_SCENARIOS["LO"]
    jpeg, regions = offer.board()
    assert jpeg[:2] == b"\xff\xd8" and regions == []
    assert offer.task == LEARN_OFFER and offer.expected == "idle"
    assert b.judge(offer, AvatarAction(OFFER_QUESTION, "idle", 960, 540)) == []
    assert b.judge(offer, AvatarAction("Fractions.", "pointing", 1, 1)) == ["state pointing", offer.check("Fractions.")]
    assert b.MATH_SCENARIOS["LE"].turns and b.MATH_SCENARIOS["QC"].task["answer"] == "x = 5"


def test_bench_no_files_and_stored_notes_scenarios(monkeypatch: pytest.MonkeyPatch) -> None:
    b = _bench(monkeypatch)
    for key in ("EQCN", "EQWN", "QCN", "QWN"):
        assert b.MATH_SCENARIOS[key].chunks() == [], "every file switched off: no notes at all"
    assert all(b.MATH_SCENARIOS[k].language == "es" for k in ("EQCN", "EQWN", "EHS"))
    right = b.MATH_SCENARIOS["EQCN"]
    assert right.check("¡Correcto, muy bien! Problema dos. dos x menos cuatro es igual a diez") is None
    assert right.check(vc.TEXTS["es"]["no_answer"]), "the copied next problem made a right answer ungraded"
    assert right.check(vc.TEXTS["es"]["reread"])
    assert b.MATH_SCENARIOS["QCN"].check(vc.TEXTS["en"]["reread"])
    notes = b.MATH_SCENARIOS["HS"].chunks()
    assert notes and all("answer is" not in c.text and "Answer:" not in c.text for c in notes), "stored without them"
    assert any("problem is 3x + 5 = 20" in c.text for c in notes)


def test_bench_spanish_heuristics(monkeypatch: pytest.MonkeyPatch) -> None:
    b = _bench(monkeypatch)
    for leak in ("Entonces x es igual a cinco.", "La respuesta es 5.", "x vale cinco", "Quince entre tres es cinco."):
        assert b._no_leak_es(leak), leak
    for fine in ("Resta cinco en ambos lados.", "Tres x más cinco es igual a veinte.", "¿Cuánto es tres por cinco?"):
        assert b._no_leak_es(fine) is None, fine
    assert b.spanish("Resta cinco en ambos lados.") is None
    assert b.spanish("Subtract five from both sides.") == "not Spanish"
    assert b.spanish("Resta 5 en ambos lados.") == "symbols/digits"
    assert b._offer_es(vc.OFFER_QUESTION_ES) is None and b._offer_es("Las fracciones son partes.")
    assert b._fraction_es("Un medio más un cuarto.") is None
    assert b._fraction_es("Un medio más un cuarto es igual a tres cuartos.") == "solved the starter problem"
    assert b.MATH_SCENARIOS["EQC"].check("¡Correcto! Problema dos. Dos x menos cuatro es igual a diez.") is None
    assert b.MATH_SCENARIOS["EQD"].check("Problem two. Two x plus three equals eleven.")


def test_bench_crop_scenario_uses_the_pipeline_crop(monkeypatch: pytest.MonkeyPatch) -> None:
    b = _bench(monkeypatch)
    jpeg, regions, crop = b.crop_board()
    assert jpeg == crop.jpeg and regions and crop.roi[2] < b.BOARD[0] // 2, "a crop of the small equation"
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape[1] == crop.size[0] and crop.scale > 1, "rendered from the raw frame at camera detail"
    whole, ink = b.whole_board()
    assert ink and cv2.imdecode(np.frombuffer(whole, np.uint8), cv2.IMREAD_COLOR).shape[:2] == (800, 1280)
    scenario = b.MATH_SCENARIOS["CR"]
    assert scenario.reads and scenario.check("3.5 - x squared equals 0.25") is None
    assert scenario.check("3.5 + x^2 = 0.25") and scenario.check("35 - x^2 = 0.25") and scenario.check("")
    assert b.judge(scenario, AvatarAction("ok", "walking", 1, 1), "3.5 - x^2 = 0.25") == []
    help_check = b.MATH_SCENARIOS["CRH"].check
    assert help_check("First, add x squared to both sides.") is None
    assert help_check("Subtract three point five from both sides.") is None
    assert help_check("Subtract x squared from both sides.") == "a step for a misread sign"
    assert help_check("Find the value of x.") == "no step on x squared, 3.5 or 0.25"


# ---- Spanish replies -------------------------------------------------------------------------------------------

ES_QUIZ = {
    "kind": "quiz",
    "stage": "check",
    "number": 1,
    "total": 5,
    "problem": "3x + 5 = 20",
    "answer": "x = 5",
    "next_problem": "2x - 4 = 10",
}


def _digits_or_symbols(text: str) -> bool:
    return bool(re.search(r"[\d^/*=+<>]", text))


@pytest.mark.parametrize("kind", vc.PROMPT_KINDS)
def test_spanish_math_prompts_are_spanish_with_the_scan_rule(kind: str) -> None:
    prompt = system_prompt("math", kind, "es")
    assert vc.SCAN_RULE in prompt and vc.SCAN_RULE_ES in prompt
    assert "Responde SIEMPRE en español de Latinoamérica" in prompt and "háblale de tú" in prompt
    for phrase in ("dos tercios", "x al cuadrado", "tres x más cinco es igual a doce", "menos cuatro"):
        assert phrase in prompt
    assert ("board_answer" in prompt) == (kind == "quiz")
    assert ("board_problem" in prompt) == (kind in ("practice", "help")), "help copies it so code can solve it"
    for example in _few_shot(prompt):
        assert set(example) >= {"spoken_text", "animation_state", "target_x", "target_y"}
        assert not _digits_or_symbols(example["spoken_text"]), example
    assert len(prompt) < 3800


@pytest.mark.parametrize("kind", vc.PROMPT_KINDS)
def test_english_math_prompt_carries_the_scan_rule(kind: str) -> None:
    prompt = system_prompt("math", kind)
    assert (
        "Scan carefully for easily missed math symbols: negative signs (-), decimal points (.), fraction bars and "
        "small exponents (^2). Read exactly what the ink says." in prompt
    )
    assert vc.SCAN_RULE_ES not in prompt


def test_spanish_offer_question_in_the_learn_prompt() -> None:
    assert vc.OFFER_QUESTION_ES == "¿Tienes un ejercicio en mente o quieres que invente uno?"
    assert vc.OFFER_QUESTION_ES in system_prompt("math", "learn", "es")


def test_biology_spanish_asks_for_a_spanish_reply() -> None:
    assert system_prompt("biology", "help", "es").startswith(SYSTEM_PROMPT)
    assert "Reply in Spanish" in system_prompt("biology", "help", "es")
    prompt = build_prompt("¿Está completo?", [], [], SIZE, profile="biology", language="es")
    assert prompt.endswith("spoken_text in Spanish.")


def test_spanish_user_prompt_labels() -> None:
    turns = [{"q": "Quiero aprender fracciones", "a": vc.OFFER_QUESTION_ES}]
    prompt = build_prompt(
        "Invéntame uno", CHUNKS, REGIONS, (1280, 800), profile="math", mode="help", recent_turns=turns, language="es"
    )
    for label in (
        "Modo: ayuda (paso a paso).",
        "Conversación hasta ahora",
        "Estudiante: Quiero aprender fracciones",
        "El estudiante dijo: Invéntame uno",
        "Centro de la pantalla: (640, 400).",
        "Regiones de tinta detectadas en el pizarrón:",
        "Contexto de referencia:",
        "en español",
    ):
        assert label in prompt
    quiz = build_prompt("¿Está bien?", [], [], SIZE, profile="math", mode="quiz", task=ES_QUIZ, language="es")
    assert "Etapa del quiz: revisar. Problema 1 de 5: 3x + 5 = 20" in quiz
    assert "2x - 4 = 10" not in quiz and "Siguiente problema" not in quiz, "never shown: it was copied as the answer"
    assert "x = 5" not in quiz, "a numeric answer is checked in code, never shown"
    learn = vc.learn_task_text({"kind": "learn", "topic": "fracciones", "stage": "offer"}, "es")
    assert learn.startswith("Etapa de aprender: ofrecer. Tema: fracciones") and vc.OFFER_QUESTION_ES in learn


def test_spanish_reply_goes_through_mathspeak_es_and_keeps_display_text() -> None:
    reply = {
        "spoken_text": "Resta 5 en ambos lados: 3x = 15.",
        "animation_state": "pointing",
        "target_x": 7,
        "target_y": 9,
    }
    action, payload = _ask(_math_client(), reply, query="¿Qué hago?", language="es")
    assert payload["messages"][0]["content"] == system_prompt("math", "help", "es")
    assert action.spoken_text == "Resta cinco en ambos lados: tres x es igual a quince."
    assert action.display_text == "Resta 5 en ambos lados: 3x = 15."  # type: ignore[attr-defined]
    english, _ = _ask(_math_client(), {**reply, "spoken_text": "Subtract 5: 3x = 15."}, query="help")
    assert english.spoken_text == "Subtract five: three x equals fifteen."
    assert english.display_text == "Subtract 5: 3x = 15."  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("text", "reveals"),
    [
        ("Entonces x es igual a cinco.", True),
        ("La respuesta es cinco.", True),
        ("Muy bien, x vale 5.", True),
        ("Ahora tienes tres x es igual a quince.", False),
        ("¿Cuánto vale x?", False),
        ("Resta cinco en ambos lados.", False),
        ("x más tres es igual a ocho.", False),
        ("Recuerda que la respuesta es una fracción.", False),
        ("Entonces x es igual a menos cuatro.", True),
        ("x tiene que ser cinco.", True),
        ("Las raíces son cinco y menos tres.", True),
        ("Eso hace que x sea cinco.", True),
        ("Tienes que restar cinco en ambos lados.", False),
        ("¿Cuánto tiene que ser x?", False),
    ],
)
def test_reveals_value_in_spanish(text: str, reveals: bool) -> None:
    assert vc.reveals_value(text, "es") is reveals


def test_reads_problem_in_spanish_words() -> None:
    assert vc.reads_problem("Problema dos. Dos x menos cuatro es igual a diez.", "2x - 4 = 10", "es")
    assert vc.reads_problem("dos x menos 4 es igual a diez", "2x - 4 = 10", "es")
    assert not vc.reads_problem("Dos x más cuatro es igual a diez.", "2x - 4 = 10", "es")
    assert vc.reads_back("Prueba un medio más un cuarto, por favor.", "1/2 + 1/4", "es")
    assert not vc.dictates_problem("Escríbelo por favor.", None, "es"), "'por favor' is no product"


def test_spanish_quiz_check_correct_reads_the_next_problem_in_spanish_words() -> None:
    action = AvatarAction("¡Excelente trabajo!", "celebrating", 1, 2)
    checked = vc.speakable(vc.check_quiz_answer(action, ES_QUIZ, "x = 5", language="es"), "es")
    assert checked.animation_state == "celebrating" and checked.verdict == "correct"  # type: ignore[attr-defined]
    assert checked.display_text == "¡Excelente trabajo! Problema 2. 2x - 4 = 10"  # type: ignore[attr-defined]
    assert checked.spoken_text == "¡Excelente trabajo! Problema dos. dos x menos cuatro es igual a diez"
    assert not _digits_or_symbols(checked.spoken_text)


def test_spanish_quiz_check_wrong_and_missing_answers() -> None:
    leak = AvatarAction("Casi. La respuesta es cinco.", "error", 1, 2)
    wrong = vc.check_quiz_answer(leak, ES_QUIZ, "x = 7", language="es")
    assert wrong.spoken_text == "Casi. Primero resta 5 en ambos lados." and wrong.verdict == "wrong"  # type: ignore[attr-defined]
    hint = AvatarAction("Casi. ¿Qué restas en ambos lados?", "error", 1, 2)
    assert vc.check_quiz_answer(hint, ES_QUIZ, "x = 7", language="es").spoken_text == hint.spoken_text
    none = vc.check_quiz_answer(hint, ES_QUIZ, None, language="es")
    assert none.spoken_text == vc.TEXTS["es"]["no_answer"] and none.animation_state == "idle"
    english = AvatarAction("Correct, great job!", "celebrating", 1, 2)
    praised = vc.check_quiz_answer(english, {**ES_QUIZ, "next_problem": None, "number": 5}, "x = 5", language="es")
    assert praised.spoken_text == "¡Correcto, muy bien! Ese fue el último problema. ¡Terminaste el quiz!"


def test_spanish_quiz_dictation_is_spanish_even_when_the_model_answers_in_english() -> None:
    task = {"kind": "quiz", "stage": "dictate", "number": 2, "total": 10, "problem": "2x + 3 = 11"}
    english = AvatarAction("Problem two of ten. Two x plus three equals eleven. Write your answer.", "idle", 1, 2)
    dictated = vc.speakable(vc.apply_task_rules(english, "quiz", task, language="es"), "es")
    assert dictated.display_text == "Problema 2 de 10. 2x + 3 = 11. Escribe tu respuesta en el pizarrón."  # type: ignore[attr-defined]
    assert (
        dictated.spoken_text
        == "Problema dos de diez. dos x más tres es igual a once. Escribe tu respuesta en el pizarrón."
    )
    spanish = AvatarAction("Problema dos de diez. Dos x más tres es igual a once. Escríbelo.", "pointing", 1, 2)
    kept = vc.apply_task_rules(spanish, "quiz", task, language="es")
    assert kept.spoken_text == spanish.spoken_text and kept.animation_state == "idle"


def test_spanish_learn_offer_asks_the_spanish_question() -> None:
    offer = {"kind": "learn", "topic": "fracciones", "stage": "offer"}
    intro = AvatarAction("Una fracción es una parte de un todo. ¿Quieres saber más?", "pointing", 1, 2)
    asked = vc.apply_task_rules(intro, "help", offer, language="es")
    assert asked.spoken_text == f"Una fracción es una parte de un todo. {vc.OFFER_QUESTION_ES}"
    own = AvatarAction("Una fracción es una parte. ¿Tienes un ejercicio o lo invento yo?", "idle", 1, 2)
    assert vc.apply_task_rules(own, "help", offer, language="es").spoken_text == own.spoken_text


def test_spanish_starter_problem_drops_the_result_and_keeps_notation_elsewhere() -> None:
    example = {"kind": "learn", "topic": "fracciones", "stage": "example"}
    solved = AvatarAction("Prueba 1/2 + 1/4. Un medio más un cuarto es igual a tres cuartos.", "idle", 1, 2)
    out = vc.apply_task_rules(solved, "help", example, language="es").spoken_text
    assert out == "Prueba 1/2 + 1/4. Un medio más un cuarto."
    equation = AvatarAction("Resuelve x más tres es igual a ocho.", "idle", 1, 2)
    assert vc.apply_task_rules(equation, "help", example, language="es").spoken_text == equation.spoken_text


def test_spanish_reply_in_english_is_asked_again_then_salvaged() -> None:
    english = {
        "spoken_text": "First subtract five from both sides.",
        "animation_state": "pointing",
        "target_x": 7,
        "target_y": 9,
    }
    c = _math_client()
    with patch.object(c.session, "post", return_value=_ollama_resp(json.dumps(english))) as post:
        action = c.evaluate("¿Qué hago?", JPEG, CHUNKS, REGIONS, SIZE, language="es")
    assert post.call_count == 2, "asked again"
    assert vc.RETRY_NOTES["es"]["language"] in post.call_args.kwargs["json"]["messages"][1]["content"]
    assert action.display_text == vc.TEXTS["es"]["step"]  # type: ignore[attr-defined]
    assert action.spoken_text == vc.TEXTS["es"]["step"]


def test_spanish_help_answer_leak_is_retried_in_spanish() -> None:
    leak = {"spoken_text": "Entonces x es igual a cinco.", "animation_state": "pointing", "target_x": 7, "target_y": 9}
    step = {"spoken_text": "Resta cinco en ambos lados.", "animation_state": "pointing", "target_x": 7, "target_y": 9}
    c = _math_client()
    responses = [_ollama_resp(json.dumps(leak)), _ollama_resp(json.dumps(step))]
    with patch.object(c.session, "post", side_effect=responses) as post:
        action = c.evaluate("Dime cuánto vale x", JPEG, CHUNKS, REGIONS, SIZE, language="es")
    assert vc.RETRY_NOTES["es"]["reveal"] in post.call_args.kwargs["json"]["messages"][1]["content"]
    assert action.spoken_text == "Resta cinco en ambos lados."


def test_spanish_failures_are_spanish() -> None:
    action, _ = _ask(_math_client(), requests.Timeout("slow"), language="es")
    assert action.spoken_text == vc.TEXTS["es"]["timeout"] and action.failed  # type: ignore[attr-defined]
    c = _math_client()
    with patch.object(c.session, "post", return_value=_ollama_resp("no json")):
        garbled = c.evaluate("hola", JPEG, [], [], SIZE, language="es")
    assert garbled.spoken_text == vc.TEXTS["es"]["not_json"]


def test_mock_backend_speaks_spanish() -> None:
    c = _math_client("mock")
    task = {"kind": "quiz", "stage": "dictate", "number": 1, "total": 3, "problem": "x + 2 = 5"}
    dictated = c.evaluate("Hazme un quiz", None, [], [], SIZE, mode="quiz", task=task, language="es")
    assert (
        dictated.spoken_text == "Problema uno de tres. x más dos es igual a cinco. Escribe tu respuesta en el pizarrón."
    )
    plain = c.evaluate("hola", None, [], [], SIZE, language="es")
    assert plain.spoken_text.startswith('Escuché: "hola".')


def test_unknown_language_is_english() -> None:
    assert vc.normalize_reply_language("fr") == "en" and vc.normalize_reply_language("es-MX") == "es"
    assert vc.normalize_reply_language(None) == "en"


# ---- the high-res ink crop -------------------------------------------------------------------------------------


class _Crop:
    """InkCrop's shape: a 600x300 crop of the board area (400, 300, 300, 150) at 2 crop px per projector px."""

    def __init__(self, jpeg: bytes = JPEG) -> None:
        self.jpeg, self.roi, self.size, self.scale = jpeg, (400, 300, 300, 150), (600, 300), 2.0

    def to_projector(self, x: float, y: float) -> tuple[float, float]:
        return self.roi[0] + x / self.scale, self.roi[1] + y / self.scale


def _wide_jpeg() -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((900, 1800, 3), 200, np.uint8))
    assert ok
    return buf.tobytes()


def test_crop_is_sent_unshrunk_and_located_in_the_prompt() -> None:
    big = _wide_jpeg()
    reply = {"spoken_text": "Look here.", "animation_state": "pointing", "target_x": 500, "target_y": 350}
    c = VLMClient("ollama", "m", "http://x:11434", profile="math", max_image_side=1024)
    with patch.object(c.session, "post", return_value=_ollama_resp(json.dumps(reply))) as post:
        action = c.evaluate("help", big, [], REGIONS, (1280, 800), crop=_Crop(big))
    sent = post.call_args.kwargs["json"]["messages"][1]
    assert base64.b64decode(sent["images"][0]) == big, "never shrunk again"
    assert "left=400 top=300 width=300 height=150" in sent["content"] and "not photo pixels" in sent["content"]
    assert (action.target_x, action.target_y) == (500, 350), "inside the crop's board area: projector pixels"
    with patch.object(c.session, "post", return_value=_ollama_resp(json.dumps(reply))) as post:
        c.evaluate("help", big, [], REGIONS, (1280, 800))
    whole = base64.b64decode(post.call_args.kwargs["json"]["messages"][1]["images"][0])
    assert cv2.imdecode(np.frombuffer(whole, np.uint8), cv2.IMREAD_COLOR).shape[1] == 1024, "a whole board shrinks"


@pytest.mark.parametrize(
    ("state", "target", "expected"),
    [
        ("pointing", (100, 60), (450, 330)),  # crop pixels -> projector pixels
        ("celebrating", (590, 290), (695, 445)),
        ("pointing", (450, 320), (450, 320)),  # already projector pixels inside the crop's area
        ("idle", (100, 60), (100, 60)),  # idle targets are the screen center by rule: never moved
        ("pointing", (1000, 700), (1000, 700)),  # outside both: left alone
    ],
)
def test_crop_targets_end_in_projector_pixels(state: str, target: tuple[int, int], expected: tuple[int, int]) -> None:
    reply = {"spoken_text": "Mira aquí.", "animation_state": state, "target_x": target[0], "target_y": target[1]}
    c = VLMClient("ollama", "m", "http://x:11434", profile="biology")
    with patch.object(c.session, "post", return_value=_ollama_resp(json.dumps(reply))) as post:
        action = c.evaluate("check", JPEG, [], REGIONS, (1280, 800), crop=_Crop())
    assert (action.target_x, action.target_y) == expected
    assert "left=400 top=300" in post.call_args.kwargs["json"]["messages"][1]["content"], "biology gets it too"


def test_crop_text_absent_without_a_crop() -> None:
    assert vc.crop_text(None) is None
    assert "píxeles del proyector" in str(vc.crop_text(_Crop(), "es"))
    assert "not photo pixels" not in build_prompt("q", [], REGIONS, SIZE, profile="math")


def test_context_strips_an_answer_key_spread_over_chunks() -> None:
    chunks = [
        RetrievedChunk("7. x = 14\n8. x = 2\n\n## Notes\nShow each step.", "ws.md", 0.9, {}),
        RetrievedChunk("# Worksheet\n7. x/2 - 4 = 3\n## Answer key\n1. x = 6", "ws.md", 0.8, {}),
    ]
    prompt = build_prompt("help", chunks, [], SIZE, profile="math")
    assert "x = 14" not in prompt and "x = 6" not in prompt and "x/2 - 4 = 3" in prompt and "Show each step." in prompt


def test_lesson_starter_without_a_problem_is_asked_again_then_falls_back() -> None:
    story = {
        "spoken_text": "Imagina que tienes cinco manzanas y las repartes en dos partes iguales. ¿Cuántas hay en cada parte?",
        "animation_state": "idle",
        "target_x": 1,
        "target_y": 2,
    }
    task = {"kind": "learn", "topic": "resolver fracciones", "stage": "example"}
    c = _math_client()
    with patch.object(c.session, "post", return_value=_ollama_resp(json.dumps(story))) as post:
        action = c.evaluate("Invéntame uno", None, [], [], SIZE, task=task, language="es")
    assert post.call_count == 2 and "no dictó un problema" in post.call_args.kwargs["json"]["messages"][1]["content"]
    assert (
        action.display_text == "Empecemos con uno fácil. 1/5 + 2/5. Escríbelo en el pizarrón e intenta el primer paso."
    )  # type: ignore[attr-defined]
    assert action.spoken_text.startswith("Empecemos con uno fácil. un quinto más dos quintos.")
    good = {**story, "spoken_text": "Empecemos con uno fácil. Un tercio más un tercio. Escríbelo."}
    with patch.object(c.session, "post", return_value=_ollama_resp(json.dumps(good))) as post:
        kept = c.evaluate("Invéntame uno", None, [], [], SIZE, task=task, language="es")
    assert post.call_count == 1 and kept.spoken_text == good["spoken_text"]


def test_lesson_starter_rule_only_for_fractions_and_equations() -> None:
    reply = {"spoken_text": "Find the square root of sixteen.", "animation_state": "idle", "target_x": 1, "target_y": 2}
    task = {"kind": "learn", "topic": "square roots", "stage": "example"}
    c = _math_client()
    with patch.object(c.session, "post", return_value=_ollama_resp(json.dumps(reply))) as post:
        action = c.evaluate("make one up", None, [], [], SIZE, task=task)
    assert post.call_count == 1 and action.spoken_text == "Find the square root of sixteen."


# ---- the next quiz problem, copied as the student's answer ----------------------------------------------------


@pytest.mark.parametrize(
    ("read", "copies"),
    [
        ("2x - 4 = 10", True),
        ("2x-4=10.", True),
        ("2x − 4 = 10", True),
        ("two x minus four equals ten", True),
        ("3x + 5 = 20, 2x - 4 = 10", True),
        ("x = 5", False),
        ("3x + 5 = 20", False),
        ("x = 7", False),
        ("", False),
        (None, False),
    ],
)
def test_copies_problem(read: str | None, copies: bool) -> None:
    assert vc.copies_problem(read, "2x - 4 = 10") is copies
    assert vc.copies_problem(read, None) is False


@pytest.mark.parametrize("read", ["3x + 5 = 20", "3x + 5 = 20, x = 5", "3x + 5 = 20 luego x = 5", "x = 5"])
def test_copies_problem_needs_the_whole_problem(read: str) -> None:
    # The next problem's text inside the current one's is no copy of it.
    assert vc.copies_problem(read, "x + 5 = 20", "es") is False
    assert vc.copies_problem("x + 5 = 20", "x + 5 = 20", "es") is True


@pytest.mark.parametrize("language", ["en", "es"])
def test_board_answer_that_is_the_next_problem_is_not_graded(language: str) -> None:
    praise = AvatarAction("¡Correcto!" if language == "es" else "Correct!", "celebrating", 1, 2)
    for read in ("2x - 4 = 10", "dos x menos cuatro es igual a diez" if language == "es" else "2x-4=10"):
        out = vc.check_quiz_answer(praise, ES_QUIZ, read, language=language)
        assert out.verdict is None and out.animation_state == "idle"  # type: ignore[attr-defined]
        assert out.spoken_text == vc.TEXTS[language]["reread"] and out.dictated is None  # type: ignore[attr-defined]


def _check_replies(*reads: str, state: str = "celebrating", spoken: str = "¡Correcto, muy bien!") -> list[MagicMock]:
    return [
        _ollama_resp(
            json.dumps(
                {"board_answer": r, "spoken_text": spoken, "animation_state": state, "target_x": 9, "target_y": 9}
            )
        )
        for r in reads
    ]


def test_spanish_check_that_copies_the_next_problem_is_asked_again_and_graded() -> None:
    c = _math_client()
    with patch.object(c.session, "post", side_effect=_check_replies("2x - 4 = 10", "x = 5")) as post:
        action = c.evaluate("¿Está bien?", JPEG, [], REGIONS, SIZE, mode="quiz", task=ES_QUIZ, language="es")
    assert post.call_count == 2
    first, retry = (call.kwargs["json"]["messages"][1]["content"] for call in post.call_args_list)
    assert "2x - 4 = 10" not in first, "the next problem is never in a check prompt"
    assert vc.RETRY_NOTES["es"]["misread"] in retry and "2x - 4 = 10" not in retry
    assert action.verdict == "correct" and action.dictated == "2x - 4 = 10"  # type: ignore[attr-defined]
    assert action.spoken_text == "¡Correcto, muy bien! Problema dos. dos x menos cuatro es igual a diez"


MULTI_LINE = {"en": 'more than one line (both roots, or x and y), copy each of them, joined with " ; "'}
MULTI_LINE["es"] = 'más de una línea (las dos raíces, o la x y la y), copia cada una, unidas con " ; "'


@pytest.mark.parametrize("language", ["en", "es"])
def test_every_check_prompt_asks_for_every_answer_line(language: str) -> None:
    # A system or two roots written one per line lost the point when only the bottom line was copied.
    stage = " ".join(system_prompt("math", "quiz", language).split())
    note = vc.quiz_task_text(QUIZ_CHECK, language)
    retry = vc.RETRY_NOTES[language]["misread"]
    for text in (stage, note, retry):
        assert MULTI_LINE[language].lower() in " ".join(text.split()).lower(), text


def _two_line_problems() -> list[tuple[str, str]]:
    import csv

    bank = Path(__file__).resolve().parent.parent / "biobench_sar" / "knowledge" / "math" / "quiz_bank.csv"
    rows = list(csv.DictReader(bank.open(encoding="utf-8")))
    return [(r["problem"], r["answer"]) for r in rows if re.search(r"\b(?:and|or)\b", r["answer"])]


def test_two_line_answers_of_every_bundled_system_and_quadratic_get_the_point() -> None:
    problems = _two_line_problems()
    assert len(problems) == 7, "the 3 systems and 4 quadratics of the bundled bank"
    for problem, answer in problems:
        task = {**QUIZ_CHECK, "problem": problem, "answer": answer, "next_problem": None}
        lines = re.split(r"\s+(?:and|or)\s+", answer)
        for language in ("en", "es"):
            # Copied joined with " ; ", or one per line, and a check line under them is no answer (R1).
            for read in (" ; ".join(lines), "\n".join(lines), " ; ".join([*lines, "7 + 3 = 10"])):
                action = vc.check_quiz_answer(PRAISE, task, read, language=language)
                assert action.verdict == "correct", (problem, read, language)  # type: ignore[attr-defined]
            half = vc.check_quiz_answer(PRAISE, task, lines[-1], language=language)
            assert half.verdict == "wrong" and half.spoken_text == vc.TEXTS[language]["incomplete"]  # type: ignore[attr-defined]


def test_a_misread_check_asks_again_for_every_answer_line_and_grades_them() -> None:
    c = _math_client()
    task = {**QUIZ_CHECK, "problem": "x + y = 10 and x - y = 4", "answer": "x = 7 and y = 3"}
    replies = _check_replies("2x - 4 = 10", "x = 7 ; y = 3 ; 7 + 3 = 10", spoken="Correct, great job!")
    with patch.object(c.session, "post", side_effect=replies) as post:
        action = c.evaluate("Is it right?", JPEG, [], REGIONS, SIZE, mode="quiz", task=task)
    assert post.call_count == 2
    retry = post.call_args.kwargs["json"]["messages"][1]["content"]
    assert vc.RETRY_NOTES["en"]["misread"] in retry and MULTI_LINE["en"] in retry
    assert action.verdict == "correct"  # type: ignore[attr-defined]


@pytest.mark.parametrize("language", ["en", "es"])
def test_check_that_copies_the_next_problem_twice_stays_ungraded(language: str) -> None:
    c = _math_client()
    replies = _check_replies("2x - 4 = 10", "2x-4=10", state="error", spoken="Casi." if language == "es" else "No.")
    with patch.object(c.session, "post", side_effect=replies) as post:
        action = c.evaluate("check it", JPEG, [], REGIONS, SIZE, mode="quiz", task=ES_QUIZ, language=language)
    assert post.call_count == 2
    assert action.verdict is None and action.animation_state == "idle"  # type: ignore[attr-defined]
    assert action.spoken_text == vc.TEXTS[language]["reread"]


def test_a_right_reading_is_not_asked_again() -> None:
    c = _math_client()
    with patch.object(c.session, "post", side_effect=_check_replies("x = 7", state="error", spoken="Casi.")) as post:
        action = c.evaluate("¿Está bien?", JPEG, [], REGIONS, SIZE, mode="quiz", task=ES_QUIZ, language="es")
    assert post.call_count == 1 and action.verdict == "wrong"  # type: ignore[attr-defined]


def test_correct_praise_that_reads_a_problem_of_its_own_is_replaced() -> None:
    # The old few-shot example read "Problem two. Two x equals fourteen."; only the quiz's next problem is read out.
    own = AvatarAction("¡Correcto, muy bien! Problema dos. Dos x es igual a catorce.", "celebrating", 1, 2)
    out = vc.check_quiz_answer(own, ES_QUIZ, "x = 5", language="es")
    assert out.spoken_text == "¡Correcto, muy bien! Problema 2. 2x - 4 = 10"
    for praise in ("¡Excelente trabajo!", "¡Correcto! Quitaste el más cinco.", "¡Bien! x es igual a cinco."):
        kept = vc.check_quiz_answer(AvatarAction(praise, "celebrating", 1, 2), ES_QUIZ, "x = 5", language="es")
        assert kept.spoken_text == f"{praise} Problema 2. 2x - 4 = 10", "praise without a problem of its own stays"


def test_check_system_prompts_never_show_a_next_problem() -> None:
    for language in ("en", "es"):
        prompt = system_prompt("math", "quiz", language)
        assert "Next problem" not in prompt and "Siguiente problema" not in prompt
        assert "2x = 14" not in prompt and "catorce" not in prompt and "fourteen" not in prompt


# --- quiz answers graded in code: several values, systems, words ----------------------------------------------

QUADRATIC = {
    "kind": "quiz",
    "stage": "check",
    "number": 10,
    "total": 10,
    "problem": "x^2 - 2x - 15 = 0",
    "answer": "x = 5 or x = -3",
    "next_problem": None,
}
SYSTEM = {**QUADRATIC, "problem": "x + y = 10 and x - y = 4", "answer": "x = 7 and y = 3"}
CHEER = AvatarAction("Correct, great job!", "celebrating", 8, 9)


@pytest.mark.parametrize(
    ("task", "read", "verdict"),
    [
        (QUADRATIC, "x = 5 or x = -3", "correct"),
        (QUADRATIC, "x = -3, x = 5", "correct"),
        (QUADRATIC, "x = 5 y x = -3", "correct"),
        (QUADRATIC, "x = 7 or x = -3", "wrong"),
        (QUADRATIC, "x = 3 or x = -5", "wrong"),
        (QUADRATIC, "x = 5", "wrong"),
        ({**QUADRATIC, "answer": None}, "x = 5 or x = -3", "correct"),  # solved in code
        ({**QUADRATIC, "answer": None}, "x = 4 or x = -3", "wrong"),
        (SYSTEM, "x = 7, y = 3", "correct"),
        (SYSTEM, "(7, 3)", "correct"),
        (SYSTEM, "x = 8, y = 2", "wrong"),
        (SYSTEM, "(3, 7)", "wrong"),
        ({**QUIZ_CHECK, "answer": "x = 5"}, "x es igual a cinco", "correct"),
    ],
)
def test_quiz_check_grades_every_answer_in_code(task: dict[str, Any], read: str, verdict: str) -> None:
    """The 3B model praised every wrong answer to a quadratic or a system: its verdict never awards a point."""
    action = vc.check_quiz_answer(CHEER, task, read)
    assert action.verdict == verdict  # type: ignore[attr-defined]
    assert action.animation_state == ("celebrating" if verdict == "correct" else "error")
    if verdict == "wrong":
        assert not praises_answer(action.spoken_text) and action.spoken_text.startswith("Not quite")
        assert not grading.states_solution(action.spoken_text, vc.expected_answers(task))


def praises_answer(text: str) -> bool:
    return vc.praises(text)


def test_quiz_check_rejection_of_a_right_answer_is_overruled() -> None:
    rejected = AvatarAction("Not quite. Try again.", "error", 8, 9)
    assert vc.check_quiz_answer(rejected, QUADRATIC, "x = -3 or x = 5").verdict == "correct"  # type: ignore[attr-defined]


def test_quiz_check_says_an_answer_is_incomplete() -> None:
    action = vc.check_quiz_answer(CHEER, QUADRATIC, "x = 5")
    assert action.spoken_text == vc.TEXTS["en"]["incomplete"] and action.hint_needed  # type: ignore[attr-defined]
    es = vc.check_quiz_answer(CHEER, QUADRATIC, "x = 5", language="es")
    assert es.spoken_text == vc.TEXTS["es"]["incomplete"]


def test_expected_answers_and_answer_for_problems_code_solves() -> None:
    assert vc.expected_answer({**QUADRATIC, "answer": None}) == "x = 5 or x = -3"
    assert vc.expected_answer({**SYSTEM, "answer": None}) == "x = 7 and y = 3"
    assert grading.format_answers(vc.expected_answers(QUADRATIC)) == "x = 5 or x = -3"
    assert vc.expected_answers({"problem": "a word problem"}) == frozenset()


def test_check_prompt_never_shows_an_expected_answer() -> None:
    for task in (QUADRATIC, SYSTEM, {**QUADRATIC, "answer": None}):
        for language in ("en", "es"):
            text = vc.quiz_task_text(task, language)
            assert "x = 5" not in text and "x = 7" not in text and "Expected" not in text and "esperada" not in text


@pytest.mark.parametrize(
    ("text", "praise", "language"),
    [
        ("Correct, great job!", True, "en"),
        ("Well done.", True, "en"),
        ("Not quite. That is not correct.", False, "en"),
        ("Not quite. Subtract five first.", False, "en"),
        ("¡Muy bien!", True, "es"),
        ("¡Correcto!", True, "es"),
        ("Casi. No es correcto todavía.", False, "es"),
        ("Great!", True, "en"),
        ("Right!", True, "en"),
        ("You're right.", True, "en"),
        ("You are right, x is five.", True, "en"),
        ("The answer is right.", True, "en"),
        ("Nice!", True, "en"),
        ("Good!", True, "en"),
        ("¡Buen trabajo!", True, "es"),
        ("¡Bien!", True, "es"),
        ("Está bien.", True, "es"),
        ("Tienes razón.", True, "es"),
        ("¡Genial!", True, "es"),
        # a verdict against the answer, a step, a word that only looks like praise
        ("Not quite.", False, "en"),
        ("That's not right.", False, "en"),
        ("That is not quite right, check the sign.", False, "en"),
        ("Right, now subtract five from both sides.", False, "en"),
        ("Look at the right side of the equation.", False, "en"),
        ("Good question: what undoes times three?", False, "en"),
        ("Is that right?", False, "en"),
        ("Casi.", False, "es"),
        ("Casi, no está bien.", False, "es"),
        ("No está del todo bien.", False, "es"),
        ("Revisa bien el signo.", False, "es"),
    ],
)
def test_praises(text: str, praise: bool, language: str) -> None:
    assert vc.praises(text, language) is praise


def test_praise_is_left_out_of_a_hint_for_a_wrong_answer() -> None:
    solutions = vc.expected_answers(QUADRATIC)
    hint = vc.hint_after_wrong("Not quite. Great! Now factor it.", QUADRATIC, solutions=solutions)
    assert hint == "Not quite. Now factor it."
    es = vc.hint_after_wrong("Casi. ¡Buen trabajo! Ahora factoriza.", QUADRATIC, language="es", solutions=solutions)
    assert es == "Casi. Ahora factoriza."
    praised = AvatarAction("Not quite. You're right that it factors. Look for two numbers.", "error", 1, 2)
    assert "right" not in vc.check_quiz_answer(praised, QUADRATIC, "x = 7 or x = -3").spoken_text


def test_hint_after_wrong_cleans_the_models_hint() -> None:
    solutions = vc.expected_answers(QUIZ_CHECK)
    cleaned = vc.hint_after_wrong(
        "Great job! Not quite, subtract five from both sides first. So x is 5.", QUIZ_CHECK, solutions=solutions
    )
    assert cleaned == "Not quite. Subtract five from both sides first."
    assert vc.hint_after_wrong("Correct!", QUIZ_CHECK, solutions=solutions) is None
    assert vc.hint_after_wrong("Not quite.", QUIZ_CHECK, solutions=solutions) is None
    assert vc.hint_after_wrong("Entonces x es igual a cinco.", QUIZ_CHECK, language="es", solutions=solutions) is None
    es = vc.hint_after_wrong("Casi. Resta cinco en ambos lados.", QUIZ_CHECK, language="es", solutions=solutions)
    assert es == "Casi. Resta cinco en ambos lados."


def _posts(c: VLMClient, *replies: object) -> Any:
    side = [r if isinstance(r, Exception) else _ollama_resp(json.dumps(r)) for r in replies]
    return patch.object(c.session, "post", side_effect=side)


@pytest.mark.usefixtures("no_mathspeak")
def test_wrong_quiz_answer_asks_the_model_for_one_hint_on_this_problem() -> None:
    c = _math_client()
    praise = {"board_answer": "x = 7", **PRAISE.to_dict()}
    hint = {"spoken_text": "Great job! First take away five on both sides.", "animation_state": "pointing"}
    with _posts(c, praise, hint) as post:
        action = c.evaluate("Is it right?", JPEG, [], REGIONS, SIZE, mode="quiz", task=QUIZ_CHECK)
    assert post.call_count == 2
    content = post.call_args.kwargs["json"]["messages"][1]["content"]
    assert 'The student\'s answer "x = 7" is WRONG' in content and "Quiz stage: hint" in content
    assert "A good next step for this problem: Subtract 5 from both sides first." in content, "code's step as a guide"
    assert "2x - 4 = 10" not in content, "the hint call never sees the next problem"
    assert action.spoken_text == "Not quite. First take away five on both sides."
    assert (action.animation_state, action.verdict) == ("error", "wrong")  # type: ignore[attr-defined]


@pytest.mark.usefixtures("no_mathspeak")
@pytest.mark.parametrize(
    "second",
    [
        {"spoken_text": "So x is 5.", "animation_state": "pointing"},  # the answer
        {"spoken_text": "Correct, well done!", "animation_state": "celebrating"},  # praise
        requests.Timeout("slow"),  # the call failed
    ],
)
def test_unusable_wrong_answer_hint_keeps_codes_own(second: object) -> None:
    c = _math_client()
    praise = {"board_answer": "x = 7", **PRAISE.to_dict()}
    with _posts(c, praise, second):
        action = c.evaluate("Is it right?", JPEG, [], REGIONS, SIZE, mode="quiz", task=QUIZ_CHECK)
    assert action.spoken_text == "Not quite. Subtract 5 from both sides first."
    assert action.verdict == "wrong" and not action.failed  # type: ignore[attr-defined]


@pytest.mark.usefixtures("no_mathspeak")
def test_a_good_hint_of_the_models_own_needs_no_second_call() -> None:
    c = _math_client()
    reply = {"board_answer": "x = 7", **HINT.to_dict()}
    with _posts(c, reply) as post:
        action = c.evaluate("Is it right?", JPEG, [], REGIONS, SIZE, mode="quiz", task=QUIZ_CHECK)
    assert post.call_count == 1 and action.spoken_text == HINT.spoken_text


def test_incomplete_answer_tells_the_hint_call() -> None:
    c = _math_client()
    reply = {"board_answer": "x = 5", **PRAISE.to_dict()}
    hint = {"spoken_text": "There is a second answer. Look at the other factor.", "animation_state": "pointing"}
    with _posts(c, reply, hint) as post:
        action = c.evaluate("Check it", JPEG, [], REGIONS, SIZE, mode="quiz", task=QUADRATIC)
    assert vc.RETRY_NOTES["en"]["incomplete"] in post.call_args.kwargs["json"]["messages"][1]["content"]
    assert action.verdict == "wrong" and "second answer" in action.spoken_text  # type: ignore[attr-defined]


def test_spanish_wrong_answer_hint_is_spanish() -> None:
    c = _math_client()
    reply = {"board_answer": "x = 7", "spoken_text": "¡Correcto, muy bien!", "animation_state": "celebrating"}
    hint = {"spoken_text": "Primero resta cinco en ambos lados.", "animation_state": "pointing"}
    with _posts(c, reply, hint) as post:
        action = c.evaluate("¿Está bien?", JPEG, [], REGIONS, SIZE, mode="quiz", task=ES_QUIZ, language="es")
    assert 'La respuesta del estudiante "x = 7" está MAL' in post.call_args.kwargs["json"]["messages"][1]["content"]
    assert action.spoken_text == "Casi. Primero resta cinco en ambos lados."


def test_mock_backend_never_asks_for_a_hint() -> None:
    mock = _math_client("mock")
    action = mock.evaluate("check", None, [], REGIONS, SIZE, mode="quiz", task=QUADRATIC)
    assert action.verdict == "correct", "the mock reads the expected answer off the board"  # type: ignore[attr-defined]


# --- help mode never states the solution, checked against the problem solved in code ---------------------------


def test_solutions_for() -> None:
    assert grading.format_answers(vc.solutions_for("help", None, "3x + 5 = 20") or frozenset()) == "x = 5"
    assert vc.solutions_for("help", None, "x = 5") is None, "the board already shows the answer"
    assert vc.solutions_for("help", None, None) is None and vc.solutions_for("practice", None, "3x + 5 = 20") is None
    assert grading.format_answers(vc.solutions_for("quiz", QUADRATIC, None) or frozenset()) == "x = 5 or x = -3"


def test_help_prompt_asks_for_the_board_problem() -> None:
    for language in ("en", "es"):
        prompt = system_prompt("math", "help", language)
        assert '"board_problem": str' in prompt
        assert all("board_problem" in example for example in _few_shot(prompt))


def test_help_reply_stating_the_solution_is_asked_again() -> None:
    c = _math_client()
    # "and you get x = 5" is no "so x equals five" for the old check: only solving the board's problem catches it.
    assert not vc.reveals_value("Divide by three and you get x = 5.")
    leak = {**MATH_GOOD, "board_problem": "3x + 5 = 20", "spoken_text": "Divide by three and you get x = 5."}
    good = {**MATH_GOOD, "board_problem": "3x + 5 = 20"}
    with _posts(c, leak, good) as post:
        action = c.evaluate("help", JPEG, [], REGIONS, SIZE)
    assert (
        post.call_count == 2
        and vc.RETRY_NOTES["en"]["reveal"] in post.call_args.kwargs["json"]["messages"][1]["content"]
    )
    assert action.spoken_text == "Subtract five from both sides."


def test_help_reply_still_stating_the_solution_loses_that_sentence() -> None:
    c = _math_client()
    leak = {
        **MATH_GOOD,
        "board_problem": "x^2 - 2x - 15 = 0",
        "spoken_text": "Try factoring the left side. You will find that x equals negative three.",
    }
    with _posts(c, leak, leak):
        action = c.evaluate("help", JPEG, [], REGIONS, SIZE)
    assert action.spoken_text == "Try factoring the left side."


@pytest.mark.usefixtures("no_mathspeak")
@pytest.mark.parametrize(
    ("language", "query", "reply", "kept"),
    [
        (
            "en",
            "Is this right?",
            "Not quite. One half plus one third equals five sixths, not two fifths. Find a common denominator first.",
            "Not quite. Find a common denominator first.",
        ),
        (
            "en",
            "Is this right?",
            "Not quite: 1/2 + 1/3 = 5/6. Find a common denominator.",
            "Find a common denominator.",
        ),
        (
            "es",
            "¿Está bien?",
            "Casi. Un medio más un tercio es igual a cinco sextos. Busca un denominador común.",
            "Casi. Busca un denominador común.",
        ),
    ],
)
def test_help_reply_on_a_board_with_the_students_own_result_never_states_the_solution(
    language: str, query: str, reply: str, kept: str
) -> None:
    c = _math_client()
    leak = {**MATH_GOOD, "board_problem": "1/2 + 1/3 = 2/5", "spoken_text": reply}
    with _posts(c, leak, leak) as post:
        action = c.evaluate(query, JPEG, [], REGIONS, SIZE, language=language)
    assert post.call_count == 2, "asked again once"
    assert vc.RETRY_NOTES[language]["reveal"] in post.call_args.kwargs["json"]["messages"][1]["content"]
    assert action.spoken_text == kept
    assert not grading.states_solution(action.spoken_text, grading.parse_answers("5/6"), language)


def test_help_reply_on_a_board_with_a_wrong_answer_below_never_states_the_solution() -> None:
    c = _math_client()
    leak = {**MATH_GOOD, "board_problem": "2x + 3 = 11\nx = 5", "spoken_text": "Not quite. You should get four."}
    good = {**MATH_GOOD, "board_problem": "2x + 3 = 11\nx = 5", "spoken_text": "Check the subtraction step first."}
    with _posts(c, leak, good) as post:
        action = c.evaluate("Is this right?", JPEG, [], REGIONS, SIZE)
    assert post.call_count == 2 and action.spoken_text == "Check the subtraction step first."


def test_spanish_help_reply_stating_the_solution_is_removed() -> None:
    c = _math_client()
    leak = {
        **MATH_GOOD,
        "board_problem": "3x + 5 = 20",
        "spoken_text": "Resta cinco en ambos lados. Al final te va a quedar que x es igual a cinco.",
    }
    assert not vc.reveals_value(leak["spoken_text"], "es")
    with _posts(c, leak, leak):
        action = c.evaluate("ayúdame", JPEG, [], REGIONS, SIZE, language="es")
    assert action.spoken_text == "Resta cinco en ambos lados."


def test_help_on_a_board_already_solved_is_not_censored() -> None:
    c = _math_client()
    reply = {**MATH_GOOD, "board_problem": "x = 5", "spoken_text": "Check it: put it back into the equation."}
    with _posts(c, reply) as post:
        action = c.evaluate("is this right?", JPEG, [], REGIONS, SIZE)
    assert post.call_count == 1 and action.spoken_text == reply["spoken_text"]


@pytest.mark.parametrize(
    ("leak", "language"),
    [
        ("Then divide 15 by 3 to get 5.", "en"),
        ("Fifteen divided by three is five.", "en"),
        ("You should get five.", "en"),
        ("Quince entre tres es cinco.", "es"),
        ("Obtienes cinco.", "es"),
    ],
)
def test_help_reply_stating_the_solution_another_way_is_asked_again(leak: str, language: str) -> None:
    """Said without the letter, the value is still the solution: code solves the board's problem and compares."""
    c = _math_client()
    step = "Divide both sides by three." if language == "en" else "Divide ambos lados entre tres."
    first = {**MATH_GOOD, "board_problem": "3x = 15", "spoken_text": f"{step} {leak}"}
    with _posts(c, first, first) as post:
        action = c.evaluate("help", JPEG, [], REGIONS, SIZE, language=language)
    assert post.call_count == 2, "asked again"
    assert action.spoken_text == step, "what still states it is cut"


def test_quiz_hint_stating_the_solution_is_asked_again() -> None:
    c = _math_client()
    assert not vc.reveals_value("Divide and you get x = 5.")
    leak = {"spoken_text": "Divide and you get x = 5.", "animation_state": "pointing", "target_x": 7, "target_y": 9}
    with _posts(c, leak, MATH_GOOD) as post:
        action = c.evaluate("I'm stuck", JPEG, [], REGIONS, SIZE, mode="quiz", task=QUIZ_HINT)
    assert post.call_count == 2 and action.spoken_text == "Subtract five from both sides."


def test_bench_graded_scenarios(monkeypatch: pytest.MonkeyPatch) -> None:
    b = _bench(monkeypatch)
    wrong = b.MATH_SCENARIOS["QQW"]
    assert wrong.expected == "error" and wrong.task["answer"] == "x = 5 or x = -3"
    assert wrong.check("Not quite. Look for two numbers that multiply to negative fifteen.") is None
    assert wrong.check("Not quite. One answer is x = 5.") == "gave the solution"
    assert wrong.check("Correct, great job!") == "no 'Not quite' / 'Casi'"
    assert b.MATH_SCENARIOS["QSC"].expected == "celebrating" and b.MATH_SCENARIOS["EQQW"].language == "es"
    board, regions = b.draw_board(["x = 5"], (1280, 800))
    assert board.shape == (800, 1280, 3) and regions[0]["x"] == 200


def test_bench_answer_line_and_unfinished_scenarios(monkeypatch: pytest.MonkeyPatch) -> None:
    b = _bench(monkeypatch)
    for key in ("QQ2", "EQQ2"):
        two_lines = b.MATH_SCENARIOS[key]
        assert two_lines.expected == "celebrating" and two_lines.task["answer"] == "x = 5 or x = -3"
        _, regions = two_lines.board()
        assert len(regions) == 3, "the problem and one root per line"
    assert b.MATH_SCENARIOS["EQQ2"].language == "es"
    problem_only = b.MATH_SCENARIOS["QFN"]
    assert problem_only.expected == "idle" and problem_only.task["answer"] is None
    assert problem_only.check(vc.TEXTS["en"]["unfinished"]) is None
    assert problem_only.check("Correct, great job!") == "praised a problem with no answer"
    assert problem_only.check("Not quite. It is five sixths.") == "gave the result"
    assert b.MATH_SCENARIOS["QFC"].expected == "celebrating"
    steps = [s.board for s in b.FLOWS["en"]] + [s.board for s in b.FLOWS["es"]]
    assert steps.count([b.Q2, "x = 5", "x = -3"]) == 2, "both flows check a root per line"


def test_bench_landing_review_scenarios(monkeypatch: pytest.MonkeyPatch) -> None:
    b = _bench(monkeypatch)
    check_line = b.MATH_SCENARIOS["QK"]
    assert check_line.expected == "celebrating", "a read of only the check is asked again for the answer line"
    assert check_line.check(vc.TEXTS["en"]["reread"]) is None and check_line.check("Correct, great job!") is None
    assert check_line.check("Not quite. Subtract 5 from both sides first.") == "graded the check line wrong"
    assert b.MATH_SCENARIOS["QT"].check("Correct, great job!") and b.MATH_SCENARIOS["QT"].check("Ask me again.") is None
    assert b.MATH_SCENARIOS["QXR"].expected == "celebrating" and b.MATH_SCENARIOS["QXR"].task["answer"] == "x = 3"
    for key in ("HR", "EHR"):
        sum_board = b.MATH_SCENARIOS[key]
        assert sum_board.mode == "help" and sum_board.check("Not quite. It is five sixths.")
        assert sum_board.check("Casi. Un medio más un tercio es igual a cinco sextos.") == "gave the result"
    assert b.MATH_SCENARIOS["HR"].check("Not quite. Find a common denominator first.") is None
    for key in ("QS2", "EQS2"):
        _, regions = b.MATH_SCENARIOS[key].board()
        assert len(regions) == 4, "two equations, then one value per line"
    assert b.MATH_SCENARIOS["EQS2"].language == "es" and b.MATH_SCENARIOS["EHR"].language == "es"


@pytest.mark.parametrize("language", ["en", "es"])
def test_bench_flow_runs_the_whole_pipeline(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], language: str
) -> None:
    b = _bench(monkeypatch)
    b.run_flow(_math_client("mock"), language, 1)
    out = capsys.readouterr().out
    assert out.count(f"[flow {language} run 1]") == len(b.FLOWS[language])
    assert f"flow {language} run 1:" in out and "hl=[" in out, "the checks highlight the answer line"


def test_a_check_whose_post_processing_fails_is_never_graded_by_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> AvatarAction:
        raise RuntimeError("bug")

    monkeypatch.setattr(vc, "check_quiz_answer", boom)
    praise = {"board_answer": "x = 7", **PRAISE.to_dict()}
    action, _ = _ask(_math_client(), praise, mode="quiz", task=QUIZ_CHECK)
    assert action.animation_state == "idle" and action.verdict is None  # type: ignore[attr-defined]
    assert action.spoken_text == vc.TEXTS["en"]["reread"]
