from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from biobench_sar.config import Settings
from biobench_sar.envfile import load_env_file, parse_env_lines

ROOT = Path(__file__).resolve().parents[1]


def test_parse_handles_comments_quotes_export_and_inline_comments() -> None:
    text = """
# comment
SAR_PORT=6001
export SAR_PROFILE="biology"
SAR_VLM_MODEL='qwen2.5vl:3b'   # trailing note
not a line
BAD-KEY=1
EMPTY=
"""
    assert parse_env_lines(text) == {
        "SAR_PORT": "6001",
        "SAR_PROFILE": "biology",
        "SAR_VLM_MODEL": "qwen2.5vl:3b",
        "EMPTY": "",
    }


def test_load_applies_only_unset_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / "sar.env"
    env.write_text("SAR_PROJECTOR_WIDTH=1280\nSAR_PROJECTOR_HEIGHT=800\nSAR_PORT=6001\n")
    monkeypatch.delenv("SAR_PROJECTOR_WIDTH", raising=False)
    monkeypatch.delenv("SAR_PROJECTOR_HEIGHT", raising=False)
    monkeypatch.setenv("SAR_PORT", "7000")  # the real environment wins
    applied = load_env_file(env)
    try:
        assert sorted(applied) == ["SAR_PROJECTOR_HEIGHT", "SAR_PROJECTOR_WIDTH"]
        settings = Settings()
        assert settings.projector_size == (1280, 800) and settings.port == 7000
    finally:
        for key in applied:
            os.environ.pop(key, None)


def test_data_dir_from_the_env_file_is_used(tmp_path: Path) -> None:
    # The same order as __main__: load the file, then import the app's config. A fresh interpreter, because
    # config reads SAR_DATA_DIR once at import and this test process imported it long ago.
    env = tmp_path / "sar.env"
    env.write_text(f"SAR_DATA_DIR={tmp_path / 'room'}\n")
    script = (
        "from biobench_sar.envfile import load_env_file; load_env_file()\n"
        "from biobench_sar.config import Settings; print(Settings().chroma_dir.parent)\n"
    )
    environ = {k: v for k, v in os.environ.items() if k != "SAR_DATA_DIR"} | {"SAR_ENV_FILE": str(env)}
    out = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=environ, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert Path(out.stdout.strip()) == tmp_path / "room"


def test_missing_file_is_harmless(tmp_path: Path) -> None:
    assert load_env_file(tmp_path / "nope.env") == []


def test_example_file_matches_the_classroom_kit() -> None:
    values = parse_env_lines((ROOT / "sar.env.example").read_text(encoding="utf-8"))
    assert (values["SAR_CAMERA_WIDTH"], values["SAR_CAMERA_HEIGHT"]) == ("1920", "1080")  # Logitech C930e
    assert (values["SAR_PROJECTOR_WIDTH"], values["SAR_PROJECTOR_HEIGHT"]) == ("1280", "800")  # Epson WXGA
    assert all(key.startswith("SAR_") for key in values)
