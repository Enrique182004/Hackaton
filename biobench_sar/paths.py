"""Where the app's files live, from a source checkout and from a PyInstaller build."""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "SAR Whiteboard Assistant"
APP_SLUG = "sar-whiteboard-assistant"


def is_frozen() -> bool:
    """True inside a PyInstaller build."""
    return bool(getattr(sys, "frozen", False)) and hasattr(sys, "_MEIPASS")


def resource_dir() -> Path:
    """Root holding the biobench_sar package and its data: sys._MEIPASS when frozen, the checkout otherwise."""
    if is_frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent


def executable_dir() -> Path:
    """Folder the teacher sees the app in: next to the .exe, or next to the .app bundle on macOS."""
    if not is_frozen():
        return resource_dir()
    folder = Path(sys.executable).resolve().parent
    # <name>.app/Contents/MacOS/<exe>: a sar.env inside the bundle would be lost on every update.
    if folder.name == "MacOS" and folder.parent.name == "Contents" and folder.parent.parent.suffix == ".app":
        return folder.parent.parent.parent
    return folder


def user_data_dir() -> Path:
    """Per-user writable folder (calibration, uploads, vector store, sar.env); created on demand."""
    home = Path.home()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        folder = Path(base) / APP_NAME if base else home / "AppData" / "Local" / APP_NAME
    elif sys.platform == "darwin":
        folder = home / "Library" / "Application Support" / APP_NAME
    else:
        xdg = os.environ.get("XDG_DATA_HOME", "")
        # The XDG spec says relative values are invalid and must be ignored.
        base_dir = Path(xdg) if xdg and Path(xdg).is_absolute() else home / ".local" / "share"
        folder = base_dir / APP_SLUG
    folder.mkdir(parents=True, exist_ok=True)
    return folder
