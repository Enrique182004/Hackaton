"""Load per-room settings from a `sar.env` file (KEY=VALUE lines) before Settings is built."""

from __future__ import annotations

import codecs
import os
from pathlib import Path

# Not from biobench_sar.config: importing it here would read SAR_DATA_DIR before load_env_file applies the file.
DEFAULT_ENV_FILE = Path(__file__).resolve().parent.parent / "sar.env"

# key -> (file, value) for each key a sar.env put into os.environ, so advice can name the file a bad value came from.
_SOURCES: dict[str, tuple[Path, str]] = {}


def env_file_path() -> Path:
    override = os.environ.get("SAR_ENV_FILE")
    return Path(override) if override else DEFAULT_ENV_FILE


def parse_env_lines(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.removeprefix("﻿").splitlines():
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


def decode_env_bytes(data: bytes) -> str:
    """UTF-8 with or without a BOM (Notepad's "UTF-8 with BOM"), or UTF-16 with a BOM (Notepad's "Unicode" and
    Windows PowerShell 5.1's `>` / Out-File). Anything else raises UnicodeDecodeError, so the caller can tell the
    teacher to save the file as UTF-8 instead of silently dropping its settings."""
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16")
    text = data.decode("utf-8-sig")
    if "\x00" in text:  # UTF-16 without a BOM decodes as UTF-8 with a NUL after every letter
        at = data.index(b"\x00")
        raise UnicodeDecodeError("utf-8", data, at, at + 1, "NUL byte, probably UTF-16 without a BOM")
    return text


def load_env_file(path: Path | None = None) -> list[str]:
    """Apply KEY=VALUE pairs that aren't already set in the environment; returns the keys applied.

    Real environment variables win, so `SAR_PORT=6000 python -m biobench_sar` still overrides the file.
    """
    path = path or env_file_path()
    try:
        values = parse_env_lines(decode_env_bytes(path.read_bytes()))
    except OSError:
        return []
    applied = [key for key in values if key not in os.environ]
    for key in applied:
        os.environ[key] = values[key]
        _SOURCES[key] = (path, values[key])
    return applied


def env_source(key: str) -> Path | None:
    """The sar.env file that set `key`, or None when the value came from elsewhere (a real environment variable,
    or something that changed it after the file was read) or `key` is not set."""
    source = _SOURCES.get(key)
    if source is None or os.environ.get(key) != source[1]:
        return None
    return source[0]
