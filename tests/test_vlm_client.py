from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import requests

from biobench_sar.llm.vlm_client import SYSTEM_PROMPT, VLMClient, build_prompt
from biobench_sar.schemas import InkRegion, RetrievedChunk

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
