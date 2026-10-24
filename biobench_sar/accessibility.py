"""Accessibility settings for the whole room: captions, text size, voice speed, calm mode and high contrast.

The projector, the voice and every open dashboard follow one set of values, saved as JSON so it survives a restart.
"""

from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from biobench_sar.jsonfile import save_json

TEXT_SIZES: tuple[str, ...] = ("normal", "large", "xlarge")
# Multiples of the voice's normal speed; TTSEngine.set_rate takes the number.
SPEECH_RATES: dict[str, float] = {"slower": 0.75, "normal": 1.0, "faster": 1.25}


@dataclass(frozen=True)
class AccessibilitySettings:
    """The defaults are the room as it was before these settings: timed captions, normal text and voice."""

    captions_stay: bool = False
    text_size: str = "normal"
    speech_rate: str = "normal"
    calm: bool = False
    high_contrast: bool = False

    def to_dict(self) -> dict[str, bool | str]:
        """All five keys in field order: the API body, the `accessibility` event and the file."""
        return asdict(self)

    @property
    def rate_factor(self) -> float:
        return SPEECH_RATES[self.speech_rate]


_KEYS = tuple(f.name for f in fields(AccessibilitySettings))
_FLAGS = frozenset({"captions_stay", "calm", "high_contrast"})
_CHOICES: dict[str, tuple[str, ...]] = {"text_size": TEXT_SIZES, "speech_rate": tuple(SPEECH_RATES)}


def _check(key: str, value: object) -> bool | str:
    """`value` when `key` allows it; ValueError naming the key and what it allows otherwise."""
    if key in _FLAGS:
        if isinstance(value, bool):  # a JSON true/false, not 1 or 0
            return value
        raise ValueError(f"{key} must be true or false")
    choices = _CHOICES[key]
    if isinstance(value, str) and value in choices:  # exact, like /api/language: "Large" is not "large"
        return value
    raise ValueError(f"{key} must be one of: {', '.join(choices)}")


def parse_changes(body: object) -> dict[str, bool | str]:
    """The checked changes in a PUT /api/accessibility body: a JSON object with any non-empty subset of the keys.

    Anything else raises ValueError with an English message for the 400 reply, and nothing is applied.
    """
    if not isinstance(body, dict) or not body:
        raise ValueError(f"send a JSON object with at least one of: {', '.join(_KEYS)}")
    changes: dict[str, bool | str] = {}
    for key, value in body.items():
        if key not in _KEYS:
            raise ValueError(f"unknown setting: {key}")
        changes[key] = _check(key, value)
    return changes


class AccessibilityStore:
    """The room's settings: loaded once, saved on every change. Thread-safe.

    File format: the to_dict() keys. A missing or unreadable file gives the defaults; a key that is missing from the
    file or holds a value parse_changes would refuse gives that key's default, and the valid keys are kept.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._settings = self._load()

    def _load(self) -> AccessibilitySettings:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return AccessibilitySettings()
        if not isinstance(data, dict):
            return AccessibilitySettings()
        kept: dict[str, bool | str] = {}
        for key in _KEYS:
            if key in data:
                with contextlib.suppress(ValueError):
                    kept[key] = _check(key, data[key])
        return AccessibilitySettings(**kept)

    @property
    def current(self) -> AccessibilitySettings:
        with self._lock:
            return self._settings

    def update(self, changes: Mapping[str, bool | str]) -> AccessibilitySettings:
        """Apply changes parse_changes already checked, save them and return the new settings."""
        with self._lock:
            self._settings = replace(self._settings, **changes)
            # save_json logs a failure; the new value still applies, it just won't survive a restart.
            save_json(self._path, self._settings.to_dict(), "accessibility settings")
            return self._settings
