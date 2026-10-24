from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

FETCH_PATH = Path(__file__).resolve().parent.parent / "packaging" / "fetch_models.py"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sar_fetch_models", FETCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fetch = _load()


def _can_symlink(tmp_path: Path) -> bool:
    try:
        (tmp_path / "probe-link").symlink_to(tmp_path)
    except (OSError, NotImplementedError):
        return False
    return True


class FakeModel:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]


class FakeHub:
    """Writes the Hugging Face cache layout the way huggingface_hub does: snapshot files link into blobs/."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def __call__(self, model: str, cache: Path, local_files_only: bool) -> FakeModel:
        self.calls.append((model, local_files_only))
        repo = fetch.repo_dir(cache, model)
        if local_files_only:
            if not fetch.is_complete(cache, model):
                raise OSError("not cached")
            return FakeModel()
        snapshot = repo / "snapshots" / REVISION
        (snapshot / "1_Pooling").mkdir(parents=True)
        (repo / "blobs").mkdir()
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text(REVISION)
        (cache / ".locks").mkdir(exist_ok=True)
        files = {"modules.json": "[]", "model.safetensors": "weights", "1_Pooling/config.json": "{}"}
        for i, (name, content) in enumerate(files.items()):
            blob = repo / "blobs" / f"blob{i}"
            blob.write_text(content)
            (snapshot / name).symlink_to(os.path.relpath(blob, (snapshot / name).parent))
        return FakeModel()


@pytest.fixture
def hub(tmp_path: Path) -> FakeHub:
    if not _can_symlink(tmp_path):
        pytest.skip("symlinks not permitted here")
    return FakeHub()


def test_bare_names_resolve_to_the_sentence_transformers_org(tmp_path: Path) -> None:
    assert fetch.repo_id("all-MiniLM-L6-v2") == "sentence-transformers/all-MiniLM-L6-v2"
    assert fetch.repo_id("BAAI/bge-small-en") == "BAAI/bge-small-en"
    assert fetch.repo_dir(tmp_path, "all-MiniLM-L6-v2").name == "models--sentence-transformers--all-MiniLM-L6-v2"


def test_download_leaves_plain_files_and_no_blobs(tmp_path: Path, hub: FakeHub) -> None:
    dest = tmp_path / "models"
    message = fetch.fetch_embedding_model(dest, "all-MiniLM-L6-v2", load=hub)
    assert "downloaded" in message and "3 links replaced" in message
    snapshot = fetch.snapshot_dir(dest, "all-MiniLM-L6-v2")
    assert snapshot is not None
    assert (snapshot / "model.safetensors").read_text() == "weights"
    assert (snapshot / "1_Pooling" / "config.json").read_text() == "{}"
    assert not any(p.is_symlink() for p in dest.rglob("*"))
    assert not (fetch.repo_dir(dest, "all-MiniLM-L6-v2") / "blobs").exists()
    assert not (dest / ".locks").exists()
    assert hub.calls == [("all-MiniLM-L6-v2", False), ("all-MiniLM-L6-v2", True)]


def test_second_run_only_rechecks(tmp_path: Path, hub: FakeHub) -> None:
    dest = tmp_path / "models"
    fetch.fetch_embedding_model(dest, load=hub)
    hub.calls.clear()
    assert "already" in fetch.fetch_embedding_model(dest, load=hub)
    assert hub.calls == [("all-MiniLM-L6-v2", True)]


def test_a_half_downloaded_model_is_fetched_again(tmp_path: Path, hub: FakeHub) -> None:
    dest = tmp_path / "models"
    fetch.fetch_embedding_model(dest, load=hub)
    snapshot = fetch.snapshot_dir(dest, "all-MiniLM-L6-v2")
    assert snapshot is not None
    (snapshot / "model.safetensors").unlink()
    assert not fetch.is_complete(dest, "all-MiniLM-L6-v2")
    hub.calls.clear()
    fetch.fetch_embedding_model(dest, load=hub)
    assert hub.calls[0] == ("all-MiniLM-L6-v2", False)
    assert fetch.is_complete(dest, "all-MiniLM-L6-v2")


def test_a_model_that_wont_load_offline_is_fetched_again(tmp_path: Path, hub: FakeHub) -> None:
    dest = tmp_path / "models"
    fetch.fetch_embedding_model(dest, load=hub)
    broken = [True]

    def flaky(model: str, cache: Path, local_files_only: bool) -> Any:
        if local_files_only and broken:
            broken.clear()
            raise RuntimeError("corrupt weights")
        return hub(model, cache, local_files_only)

    assert "downloaded" in fetch.fetch_embedding_model(dest, load=flaky)


def test_an_empty_embedding_fails_the_fetch(tmp_path: Path, hub: FakeHub) -> None:
    class Empty:
        def encode(self, texts: list[str]) -> list[list[float]]:
            return [[]]

    def load(model: str, cache: Path, local_files_only: bool) -> Any:
        hub(model, cache, local_files_only)
        return Empty()

    with pytest.raises(RuntimeError, match="no embedding"):
        fetch.fetch_embedding_model(tmp_path / "models", load=load)


def test_whisper_checkpoint_is_fetched_once(tmp_path: Path) -> None:
    downloads: list[str] = []

    def download(name: str, root: Path) -> None:
        downloads.append(name)
        (root / f"{name}.pt").write_bytes(b"checkpoint")

    assert "downloaded" in fetch.fetch_whisper_model(tmp_path, "base", download=download)
    assert "already" in fetch.fetch_whisper_model(tmp_path, "base", download=download)
    assert downloads == ["base"]
    assert (tmp_path / "whisper" / "base.pt").is_file()


def test_whisper_download_without_a_checkpoint_fails(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=r"no base\.pt"):
        fetch.fetch_whisper_model(tmp_path, "base", download=lambda name, root: None)


def test_main_reports_failure_with_exit_code_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def offline(dest: Path, model: str) -> str:
        raise OSError("no network")

    monkeypatch.setattr(fetch, "fetch_embedding_model", offline)
    assert fetch.main(["--dest", str(tmp_path)]) == 1
    assert "no network" in capsys.readouterr().err


def test_main_skips_whisper_unless_asked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetch, "fetch_embedding_model", lambda dest, model: "ok")
    monkeypatch.setattr(fetch, "fetch_whisper_model", lambda dest, name: pytest.fail("not asked for"))
    assert fetch.main(["--dest", str(tmp_path)]) == 0
