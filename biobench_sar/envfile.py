"""Load per-room settings from a `sar.env` file (KEY=VALUE lines) before Settings is built."""

from __future__ import annotations

import os
from pathlib import Path

from biobench_sar.config import PACKAGE_DIR

DEFAULT_ENV_FILE = PACKAGE_DIR.parent / "sar.env"


def env_file_path() -> Path:
    override = os.environ.get("SAR_ENV_FILE")
    return Path(override) if override else DEFAULT_ENV_FILE


def parse_env_lines(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.split(" #", 1)[0].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key.isidentifier():
            values[key] = value
    return values


def load_env_file(path: Path | None = None) -> list[str]:
    """Apply KEY=VALUE pairs that aren't already set in the environment; returns the keys applied.

    Real environment variables win, so `SAR_PORT=6000 python -m biobench_sar` still overrides the file.
    """
    path = path or env_file_path()
    try:
        values = parse_env_lines(path.read_text(encoding="utf-8"))
    except OSError:
        return []
    applied = [key for key in values if key not in os.environ]
    for key in applied:
        os.environ[key] = values[key]
    return applied
