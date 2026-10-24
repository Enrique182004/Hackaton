from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal, TypedDict, get_args

AnimationState = Literal["idle", "walking", "pointing", "celebrating", "error"]
ANIMATION_STATES: frozenset[str] = frozenset(get_args(AnimationState))


class InkRegion(TypedDict):
    """Bounding box of a marker stroke cluster, in projector coordinates."""

    x: int
    y: int
    w: int
    h: int
    cx: int
    cy: int


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    score: float
    metadata: dict[str, str | int | float]


@dataclass(frozen=True)
class AvatarAction:
    spoken_text: str
    animation_state: AnimationState
    target_x: int
    target_y: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def error_action(message: str, projector_size: tuple[int, int]) -> AvatarAction:
    width, height = projector_size
    return AvatarAction(message, "error", width // 2, height // 2)


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    candidates: list[str] = [m.group(1) for m in _FENCE_RE.finditer(text)]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    candidates.append(text)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate.strip())
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _clamp_int(value: Any, low: int, high: int, fallback: int) -> int:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return fallback
    return max(low, min(high, number))


def parse_vlm_response(text: str, projector_size: tuple[int, int]) -> AvatarAction:
    """Parse VLM output into an AvatarAction. Never raises: VLMs wrap JSON in prose/markdown."""
    width, height = projector_size
    data = _extract_json_object(text or "")
    if data is None:
        return error_action("Sorry, I couldn't understand my own answer. Please ask again.", projector_size)
    spoken = str(data.get("spoken_text") or "").strip() or "..."
    state = str(data.get("animation_state") or "idle").strip().lower()
    if state not in ANIMATION_STATES:
        state = "idle"
    return AvatarAction(
        spoken_text=spoken,
        animation_state=state,  # type: ignore[arg-type]
        target_x=_clamp_int(data.get("target_x"), 0, width - 1, width // 2),
        target_y=_clamp_int(data.get("target_y"), 0, height - 1, height // 2),
    )
