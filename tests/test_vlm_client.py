from __future__ import annotations

import base64
import importlib.util
import json
import re
import subprocess
import sys
import types
from fractions import Fraction
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import requests

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
    assert len(prompt) < 3000  # every token of system prompt is paid on each question


@pytest.mark.parametrize(
    ("kind", "states"),
    [("help", {"pointing"}), ("practice", {"idle"}), ("quiz", {"idle", "error", "celebrating"}), ("learn", {"idle"})],
)
def test_math_few_shot_examples_are_json_in_words(kind: str, states: set[str]) -> None:
    examples = _few_shot(system_prompt("math", kind))
    assert len(examples) >= 2
    assert {e["animation_state"] for e in examples} == states
    for e in examples:
        assert e["animation_state"] in ANIMATION_STATES
        assert isinstance(e["target_x"], int) and isinstance(e["target_y"], int)
        assert not NOTATION.search(e["spoken_text"]), e["spoken_text"]
        assert set(e) - {"board_answer"} == {"spoken_text", "animation_state", "target_x", "target_y"}


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
    invented = vc.quiz_task_text({"kind": "quiz", "stage": "dictate", "number": 1, "total": 10, "problem": None})
    assert "make up one yourself" in invented and "easiest" in invented


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
    assert "Quiz stage: check. Problem 1 of 5: 3x + 5 = 20" in text and "Next problem: 2x - 4 = 10" in text
    assert "board_answer" in text and "never the answer" in text
    # A numeric answer is graded in code; shown to the model, it gets copied into board_answer.
    assert "x = 5" not in text and "Expected answer" not in text
    symbolic = vc.quiz_task_text({**task, "answer": "x = 2 or x = -2"})
    assert "Expected answer: x = 2 or x = -2" in symbolic
    last = vc.quiz_task_text({**task, "number": 5, "next_problem": None})
    assert "last problem" in last and "Next problem" not in last
    invented = vc.quiz_task_text({**task, "problem": None, "answer": None, "next_problem": None, "number": 3})
    assert "made up earlier" in invented and "solve the problem yourself" in invented and "problem 4" in invented
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
    assert prompt.startswith("Conversation so far (oldest first):\nUser: why turbid?\nYou: Runoff.")
    assert "Mode:" not in prompt


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
    assert action == AvatarAction("Subtract five from both sides.", "idle", 7, 9)  # dictating, not pointing


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
    assert action.spoken_text == "You have two thirds left." and fake_mathspeak == ["You have 2/3 left."]


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
    failed = AvatarAction("The AI model took too long to answer.", "error", 1, 2)
    assert vc.apply_task_rules(failed, "help", LEARN_OFFER) == failed


def test_apply_task_rules_states() -> None:
    pointing = AvatarAction("Try four x plus seven equals nineteen.", "pointing", 1, 2)
    idle = AvatarAction(pointing.spoken_text, "idle", 1, 2)
    assert vc.apply_task_rules(pointing, "help", None) == pointing  # help points at the equation
    assert vc.apply_task_rules(pointing, "practice", None) == idle
    assert vc.apply_task_rules(pointing, "help", {"kind": "practice"}) == idle
    assert vc.apply_task_rules(pointing, "quiz", {"kind": "quiz", "stage": "dictate"}) == idle
    unchecked = {"kind": "quiz", "stage": "check", "answer": None}
    assert vc.apply_task_rules(pointing, "quiz", unchecked, "x = 4") == pointing


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


def test_quiz_check_keeps_model_text_when_it_agrees() -> None:
    assert vc.check_quiz_answer(PRAISE, QUIZ_CHECK, "x = 5") == PRAISE
    assert vc.check_quiz_answer(HINT, QUIZ_CHECK, "7") is HINT
    assert vc.check_quiz_answer(PRAISE, QUIZ_CHECK, "3x = 15") is PRAISE  # not a plain value: model decides
    assert vc.check_quiz_answer(PRAISE, {**QUIZ_CHECK, "answer": None}, "x = 7") is PRAISE


def test_quiz_check_overrides_wrong_praise() -> None:
    fixed = vc.check_quiz_answer(PRAISE, QUIZ_CHECK, "x = 7")
    assert fixed.animation_state == "error" and (fixed.target_x, fixed.target_y) == (8, 9)
    assert fixed.spoken_text.startswith("Not quite") and "5" not in fixed.spoken_text
    assert "five" not in fixed.spoken_text.lower() and "2x" not in fixed.spoken_text


def test_quiz_check_overrides_wrong_rejection() -> None:
    fixed = vc.check_quiz_answer(HINT, QUIZ_CHECK, "x=5")
    assert fixed == AvatarAction("Correct, great job! Problem 2. 2x - 4 = 10", "celebrating", 8, 9)
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
    assert action == PRAISE  # no board_answer: the model's verdict stands


def test_evaluate_quiz_override_is_spoken_in_words(fake_mathspeak: list[str]) -> None:
    right = {"board_answer": "x = 5", **HINT.to_dict()}
    _ask(_math_client(), right, mode="quiz", task=QUIZ_CHECK)
    assert fake_mathspeak == ["Correct, great job! Problem 2. 2x - 4 = 10"]


def test_raw_evaluate_returns_board_answer() -> None:
    c = _math_client()
    reply = {"board_answer": "x = 7", **PRAISE.to_dict()}
    with patch.object(c.session, "post", return_value=_ollama_resp(json.dumps(reply))):
        action, read = c._evaluate("q", None, [], [], SIZE, task=QUIZ_CHECK)
    assert action == PRAISE and read == "x = 7"
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
    for leak in ("So x equals five.", "x = 5", "Fifteen divided by three is five.", "The answer is 5.", "x would be 5"):
        assert b.leaks_answer(leak), leak
    fine_replies = (
        "Subtract five from both sides.",
        "Three x plus five equals twenty.",
        "x is fifteen over three?",
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
    assert set(b.MATH_SCENARIOS) == {"H", "HA", "P", "QC", "QW", "LO", "LE", "HF"}
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
