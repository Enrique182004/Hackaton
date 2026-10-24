from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pytest

from biobench_sar.assistant import DocumentSelection
from biobench_sar.jsonfile import save_json
from biobench_sar.rag.quiz_bank import QuizBank, QuizProblem


def _names(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


def _fail_replace(src: Any, dst: Any) -> None:
    raise OSError("disk full")


# ---- save_json --------------------------------------------------------------------------------------------------


def test_save_json_writes_the_payload_and_creates_folders(tmp_path: Path) -> None:
    path = tmp_path / "state" / "deep" / "thing.json"
    payload = {"all": True, "names": ["a.md", "señal.md"], "n": 3}
    assert save_json(path, payload, "thing") is True
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert _names(path.parent) == ["thing.json"], "no temp file left behind"


def test_save_json_replaces_an_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "thing.json"
    path.write_text('{"old": true, "padding": "a longer file than the new one"}', encoding="utf-8")
    assert save_json(path, [1, 2], "thing") is True
    assert json.loads(path.read_text(encoding="utf-8")) == [1, 2]
    assert _names(tmp_path) == ["thing.json"]


def test_save_json_renames_a_temp_file_from_the_same_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "thing.json"
    calls: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def spy(src: Any, dst: Any) -> None:
        calls.append((Path(src), Path(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    assert save_json(path, {"v": 1}, "thing") is True
    [(src, dst)] = calls
    assert dst == path and src.parent == path.parent, "same folder, so the rename is atomic"
    assert src.name.startswith(".thing.json.") and src.name.endswith(".tmp"), "a hidden temp file"


def test_save_json_failure_keeps_the_old_file_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "thing.json"
    assert save_json(path, {"v": 1}, "thing") is True
    before = path.read_text(encoding="utf-8")
    monkeypatch.setattr(os, "replace", _fail_replace)
    with caplog.at_level(logging.WARNING, logger="biobench_sar.jsonfile"):
        assert save_json(path, {"v": 2}, "quiz bank") is False
    assert path.read_text(encoding="utf-8") == before, "a failed write never leaves a half-written file"
    assert _names(tmp_path) == ["thing.json"], "the temp file is cleaned up"
    assert [r.getMessage() for r in caplog.records] == [f"could not save quiz bank to {path}: disk full"]


def test_save_json_failure_when_the_folder_cannot_be_made(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    blocker = tmp_path / "state"
    blocker.write_text("a file where the folder should be", encoding="utf-8")
    path = blocker / "thing.json"
    with caplog.at_level(logging.WARNING, logger="biobench_sar.jsonfile"):
        assert save_json(path, {"v": 1}, "document selection") is False
    assert blocker.read_text(encoding="utf-8") == "a file where the folder should be"
    assert _names(tmp_path) == ["state"]
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    assert caplog.records[0].getMessage().startswith(f"could not save document selection to {path}: ")


def test_selection_and_quiz_bank_save_through_save_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(os, "replace", _fail_replace)
    with caplog.at_level(logging.WARNING, logger="biobench_sar.jsonfile"):
        DocumentSelection(tmp_path / "selection.json").select(["a.md"])
        QuizBank(tmp_path / "quiz_bank.json").replace_source("a.md", [QuizProblem("1 + 1", "2", "", 1, "a.md")])
    assert [r.getMessage() for r in caplog.records if r.name == "biobench_sar.jsonfile"] == [
        f"could not save document selection to {tmp_path / 'selection.json'}: disk full",
        f"could not save quiz bank to {tmp_path / 'quiz_bank.json'}: disk full",
    ]
    assert _names(tmp_path) == [], "neither caller leaves a temp file"
