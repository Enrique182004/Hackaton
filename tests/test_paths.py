from __future__ import annotations

import sys
from pathlib import Path

import pytest

from biobench_sar import __version__, paths

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "home"
    folder.mkdir()
    monkeypatch.setattr(paths.Path, "home", lambda: folder)
    return folder


def _freeze(monkeypatch: pytest.MonkeyPatch, meipass: Path, executable: Path) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(meipass), raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))


def test_version_is_a_plain_semver_string() -> None:
    parts = __version__.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts)


def test_source_checkout_is_not_frozen_and_resources_are_the_repo() -> None:
    assert not paths.is_frozen()
    assert paths.resource_dir() == REPO
    assert (paths.resource_dir() / "biobench_sar" / "templates").is_dir()
    assert paths.executable_dir() == REPO


def test_frozen_resources_come_from_meipass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = tmp_path / "_internal"
    _freeze(monkeypatch, bundle, tmp_path / "SAR Whiteboard Assistant.exe")
    assert paths.is_frozen()
    assert paths.resource_dir() == bundle
    assert paths.executable_dir() == tmp_path.resolve()


def test_frozen_flag_without_meipass_is_not_a_pyinstaller_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    assert not paths.is_frozen()


def test_macos_bundle_settings_live_next_to_the_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contents = tmp_path / "Apps" / "SAR Whiteboard Assistant.app" / "Contents"
    (contents / "MacOS").mkdir(parents=True)
    _freeze(monkeypatch, contents / "Frameworks", contents / "MacOS" / "SAR Whiteboard Assistant")
    assert paths.executable_dir() == (tmp_path / "Apps").resolve()


def test_windows_data_dir_uses_localappdata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    folder = paths.user_data_dir()
    assert folder == tmp_path / "Local" / "SAR Whiteboard Assistant"
    assert folder.is_dir()


def test_windows_data_dir_without_localappdata_falls_back_to_home(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    assert paths.user_data_dir() == home / "AppData" / "Local" / "SAR Whiteboard Assistant"


def test_macos_data_dir_is_application_support(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    folder = paths.user_data_dir()
    assert folder == home / "Library" / "Application Support" / "SAR Whiteboard Assistant"
    assert folder.is_dir()


def test_linux_data_dir_honours_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert paths.user_data_dir() == tmp_path / "xdg" / "sar-whiteboard-assistant"


@pytest.mark.parametrize("xdg", ["", "relative/dir"])
def test_linux_data_dir_ignores_unset_or_relative_xdg(xdg: str, monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", xdg)
    folder = paths.user_data_dir()
    assert folder == home / ".local" / "share" / "sar-whiteboard-assistant"
    assert folder.is_dir()


def test_data_dir_creation_is_idempotent(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    assert paths.user_data_dir() == paths.user_data_dir()
