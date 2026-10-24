from __future__ import annotations

import codecs
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from biobench_sar.config import Settings
from biobench_sar.envfile import decode_env_bytes, env_source, load_env_file, parse_env_lines

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


@pytest.fixture
def unset(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Keys the test's sar.env may set; removed again afterwards (load_env_file writes os.environ directly)."""
    keys = ["SAR_PORT", "SAR_CAMERA_INDEX", "SAR_PROFILE"]
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    yield keys
    for key in keys:
        os.environ.pop(key, None)


def test_a_byte_order_mark_does_not_hide_the_first_setting(tmp_path: Path, unset: list[str]) -> None:
    # Notepad's "UTF-8 with BOM" and PowerShell 5.1's `Set-Content -Encoding UTF8` both start the file with one.
    env = tmp_path / "sar.env"
    env.write_bytes("SAR_PORT=6000\nSAR_CAMERA_INDEX=1\n".encode("utf-8-sig"))
    assert load_env_file(env) == ["SAR_PORT", "SAR_CAMERA_INDEX"]
    assert os.environ["SAR_PORT"] == "6000"
    assert parse_env_lines("\ufeffSAR_PORT=6000\n") == {"SAR_PORT": "6000"}


@pytest.mark.parametrize("encoding", ["utf-16", "utf-16-be"])
def test_a_utf16_file_with_a_bom_is_read(tmp_path: Path, unset: list[str], encoding: str) -> None:
    # Notepad's "Unicode" and Windows PowerShell 5.1's `"SAR_PORT=6000" > sar.env` write UTF-16 with a BOM.
    env = tmp_path / "sar.env"
    data = "SAR_PORT=6000\r\nSAR_PROFILE=biology\r\n".encode(encoding)
    if encoding == "utf-16-be":
        data = codecs.BOM_UTF16_BE + data
    env.write_bytes(data)
    assert load_env_file(env) == ["SAR_PORT", "SAR_PROFILE"]
    assert (os.environ["SAR_PORT"], os.environ["SAR_PROFILE"]) == ("6000", "biology")


@pytest.mark.parametrize(
    "data",
    [
        "SAR_PORT=6000\n".encode("utf-16-le"),  # UTF-16 without a BOM: every other byte is NUL
        "# cámara del aula\nSAR_CAMERA_INDEX=1\n".encode("cp1252"),  # an ANSI code page, not UTF-8
    ],
)
def test_a_file_in_another_encoding_raises_instead_of_dropping_settings(
    tmp_path: Path, unset: list[str], data: bytes
) -> None:
    env = tmp_path / "sar.env"
    env.write_bytes(data)
    with pytest.raises(UnicodeDecodeError):
        load_env_file(env)
    assert "SAR_PORT" not in os.environ and "SAR_CAMERA_INDEX" not in os.environ


def test_decode_env_bytes_accepts_plain_utf8() -> None:
    assert decode_env_bytes("SAR_DATA_DIR=C:\\Users\\José\n".encode()) == "SAR_DATA_DIR=C:\\Users\\José\n"


def test_each_key_remembers_the_file_that_set_it(tmp_path: Path, unset: list[str]) -> None:
    app_env, data_env = tmp_path / "app.env", tmp_path / "data.env"
    app_env.write_text("SAR_CAMERA_INDEX=2\n")
    data_env.write_text("SAR_CAMERA_INDEX=5\nSAR_PORT=6060\n")
    load_env_file(app_env)
    load_env_file(data_env)
    assert env_source("SAR_CAMERA_INDEX") == app_env  # the first file wins, and gets the blame
    assert env_source("SAR_PORT") == data_env


def test_a_real_environment_variable_has_no_file(
    tmp_path: Path, unset: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    env = tmp_path / "sar.env"
    env.write_text("SAR_PORT=6000\nSAR_CAMERA_INDEX=1\n")
    monkeypatch.setenv("SAR_PORT", "7000")
    load_env_file(env)
    assert env_source("SAR_PORT") is None, "the environment won, so the file did not set it"
    assert env_source("SAR_CAMERA_INDEX") == env
    os.environ["SAR_CAMERA_INDEX"] = "3"  # changed later (the launcher's --port does this for SAR_PORT)
    assert env_source("SAR_CAMERA_INDEX") is None
    assert env_source("SAR_NEVER_SET") is None
