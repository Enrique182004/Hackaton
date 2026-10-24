from __future__ import annotations

import ctypes
import importlib.util
import subprocess
import types
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hardware_check", ROOT / "scripts" / "hardware_check.py")
hc = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(hc)

DISPLAYS = """Graphics/Displays:
    Apple M4:
      Displays:
        Color LCD:
          Resolution: 2560 x 1664 Retina
        EPSON PJ:
          Resolution: 1280 x 800 (WXGA - Widescreen eXtended Graphics Array)
"""

CAMERAS = """Camera:

    MacBook Air Camera:

      Model ID: MacBook Air Camera

    Logitech Webcam C930e:

      Model ID: UVC Camera VendorID_1133 ProductID_2115
"""


def test_parse_resolutions() -> None:
    assert hc.parse_resolutions(DISPLAYS) == [(2560, 1664), (1280, 800)]


def test_projector_found_at_native_resolution() -> None:
    assert hc.projector_display_result([(2560, 1664), (1280, 800)], (1280, 800))[1] == "PASS"


def test_projector_wrong_resolution_warns_with_fix() -> None:
    _name, status, detail = hc.projector_display_result([(2560, 1664), (1920, 1080)], (1280, 800))
    assert status == "WARN" and "SAR_PROJECTOR_WIDTH" in detail


def test_projector_missing_warns() -> None:
    assert hc.projector_display_result([(2560, 1664)], (1280, 800))[1] == "WARN"


def test_parse_camera_names() -> None:
    assert hc.parse_camera_names(CAMERAS) == ["MacBook Air Camera", "Logitech Webcam C930e"]


def test_resolution_status() -> None:
    assert hc.resolution_status((1920, 1080), (1920, 1080)) == "PASS"
    assert hc.resolution_status((1280, 720), (1920, 1080)) == "WARN"


# ---- Windows displays (ctypes, Win32) ----------------------------------------------------------------------------


def test_win32_structures_have_the_windows_sizes() -> None:
    # sizeof(DEVMODEW) == 220 and sizeof(MONITORINFOEXW) == 104 in the Windows SDK; a wrong layout would read
    # dmPelsWidth from the wrong bytes (or let Windows write past the buffer).
    assert ctypes.sizeof(hc.DEVMODEW) == 220
    assert hc.DEVMODEW.dmPelsWidth.offset == 172 and hc.DEVMODEW.dmPelsHeight.offset == 176
    assert ctypes.sizeof(hc.MONITORINFOEXW) == 104 and hc.MONITORINFOEXW.szDevice.offset == 40


def test_wide_string_stops_at_the_nul() -> None:
    chars = (ctypes.c_uint16 * 32)(*map(ord, "\\\\.\\DISPLAY2"))
    assert hc.wide_string(chars) == "\\\\.\\DISPLAY2"


class FakeUser32:
    """user32 for two monitors: the laptop panel (whose mode query fails) and the Epson projector."""

    def __init__(self, enum_ok: bool = True, fail_in_callback: bool = False) -> None:
        self.monitors = {
            11: ("\\\\.\\DISPLAY1", (0, 0, 1536, 960)),  # 1920x1200 panel at 125 % scaling: a scaled rectangle
            22: ("\\\\.\\DISPLAY2", (1536, 0, 2816, 800)),
        }
        self.modes = {"\\\\.\\DISPLAY2": (1280, 800)}
        self.asked: list[tuple[str, int, int]] = []
        self.enum_ok = enum_ok
        self.fail_in_callback = fail_in_callback
        self.EnumDisplayMonitors = self._enum_monitors
        self.GetMonitorInfoW = self._monitor_info
        self.EnumDisplaySettingsW = self._settings

    def _enum_monitors(self, hdc: Any, clip: Any, callback: Any, data: int) -> int:
        for handle in self.monitors:
            if not callback(handle, None, None, data):
                break
        return int(self.enum_ok)

    def _monitor_info(self, handle: int, ref: Any) -> int:
        if self.fail_in_callback:
            raise RuntimeError("bad handle")
        info = ref._obj
        assert info.cbSize == ctypes.sizeof(hc.MONITORINFOEXW)
        name, (left, top, right, bottom) = self.monitors[handle]
        info.rcMonitor.left, info.rcMonitor.top, info.rcMonitor.right, info.rcMonitor.bottom = left, top, right, bottom
        for i, ch in enumerate(name):
            info.szDevice[i] = ord(ch)
        return 1

    def _settings(self, device: str, mode_number: int, ref: Any) -> int:
        mode = ref._obj
        self.asked.append((device, mode_number, mode.dmSize))
        if device not in self.modes:
            return 0
        mode.dmPelsWidth, mode.dmPelsHeight = self.modes[device]
        return 1


def _user32(**kwargs: Any) -> Any:
    # Real ctypes functions take argtypes; plain methods can't, so expose them as attribute-friendly functions.
    fake = FakeUser32(**kwargs)
    return types.SimpleNamespace(
        EnumDisplayMonitors=lambda *a: fake.EnumDisplayMonitors(*a),
        GetMonitorInfoW=lambda *a: fake.GetMonitorInfoW(*a),
        EnumDisplaySettingsW=lambda *a: fake.EnumDisplaySettingsW(*a),
        fake=fake,
    )


def test_windows_display_sizes_reads_each_monitors_current_mode() -> None:
    user32 = _user32()
    sizes = hc.windows_display_sizes(user32, ctypes.CFUNCTYPE)
    assert sizes == [(1536, 960), (1280, 800)]  # DISPLAY1 falls back to its rectangle
    assert user32.fake.asked == [
        ("\\\\.\\DISPLAY1", hc.ENUM_CURRENT_SETTINGS, 220),
        ("\\\\.\\DISPLAY2", hc.ENUM_CURRENT_SETTINGS, 220),
    ]
    assert hc.projector_display_result(sizes, (1280, 800))[1] == "PASS"


def test_windows_display_sizes_raises_what_went_wrong() -> None:
    with pytest.raises(RuntimeError, match="bad handle"):
        hc.windows_display_sizes(_user32(fail_in_callback=True), ctypes.CFUNCTYPE)
    with pytest.raises(OSError, match="EnumDisplayMonitors"):
        hc.windows_display_sizes(_user32(enum_ok=False), ctypes.CFUNCTYPE)


def test_windows_projector_is_compared_like_on_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hc, "screeninfo_sizes", lambda: pytest.fail("ctypes worked; screeninfo is not needed"))
    monkeypatch.setattr(hc, "windows_display_sizes", lambda: [(1920, 1200), (1280, 800)])
    assert hc.check_displays((1280, 800), system="Windows") == [
        ("projector", "PASS", "a display runs at 1280x800 (1920x1200; 1280x800)")
    ]
    monkeypatch.setattr(hc, "windows_display_sizes", lambda: [(1920, 1200), (1024, 768)])
    name, status, detail = hc.check_displays((1280, 800), system="Windows")[0]
    assert (name, status) == ("projector", "WARN") and "SAR_PROJECTOR_WIDTH" in detail


def test_windows_falls_back_to_screeninfo_then_explains(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> list[tuple[int, int]]:
        raise OSError("no user32")

    def missing() -> list[tuple[int, int]]:
        raise ImportError("screeninfo")

    monkeypatch.setattr(hc, "windows_display_sizes", broken)
    monkeypatch.setattr(hc, "screeninfo_sizes", lambda: [(1280, 800), (1920, 1080)])
    assert hc.check_displays((1280, 800), system="Windows")[0][1] == "PASS"
    monkeypatch.setattr(hc, "screeninfo_sizes", missing)
    assert hc.check_displays((1280, 800), system="Windows") == [
        ("displays", "WARN", "could not list the displays (OSError: no user32)")
    ]


def test_other_systems_use_screeninfo_when_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing() -> list[tuple[int, int]]:
        raise ImportError("screeninfo")

    monkeypatch.setattr(hc, "screeninfo_sizes", missing)
    assert hc.check_displays((1280, 800), system="Linux")[0][2].startswith("skipped (pip install screeninfo")
    monkeypatch.setattr(hc, "screeninfo_sizes", lambda: [(1920, 1080), (1280, 800)])
    assert hc.check_displays((1280, 800), system="Linux")[0][1] == "PASS"


# ---- Tesseract ------------------------------------------------------------------------------------------------


def _ran(stdout: str = "tesseract 5.5.2\n leptonica-1.85.0\n", stderr: str = "", code: int = 0) -> Any:
    return lambda cmd, **kw: subprocess.CompletedProcess(cmd, code, stdout=stdout, stderr=stderr)


def test_tesseract_missing_says_how_to_install_it_on_this_os() -> None:
    for system, advice in (
        ("Windows", "winget install UB-Mannheim.TesseractOCR"),
        ("Darwin", "brew install tesseract"),
        ("Linux", "apt install tesseract-ocr"),
    ):
        name, status, detail = hc.check_tesseract(find=lambda: None, system=system)[0]
        assert (name, status) == ("tesseract", "FAIL") and advice in detail


def test_tesseract_off_path_passes_and_says_where() -> None:
    exe = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    result = hc.check_tesseract(find=lambda: exe, run=_ran(), which=lambda name: None, system="Windows")
    assert result == [("tesseract", "PASS", f"tesseract 5.5.2 at {exe} (not on PATH; the assistant finds it there)")]


def test_tesseract_on_path_passes_with_its_version() -> None:
    result = hc.check_tesseract(find=lambda: "/usr/bin/tesseract", run=_ran(), which=lambda name: "/usr/bin/tesseract")
    assert result == [("tesseract", "PASS", "tesseract 5.5.2")]
    old = hc.check_tesseract(find=lambda: "/usr/bin/tesseract", run=_ran("", "tesseract 3.05.02\n"), which=lambda n: n)
    assert old == [("tesseract", "PASS", "tesseract 3.05.02")]  # version 3 printed it on stderr


def test_a_tesseract_that_does_not_run_fails() -> None:
    def broken(cmd: Any, **kw: Any) -> Any:
        raise PermissionError("blocked")

    assert hc.check_tesseract(find=lambda: "/x/tesseract", run=broken)[0][1] == "FAIL"
    assert hc.check_tesseract(find=lambda: "/x/tesseract", run=_ran(code=1))[0][2] == "/x/tesseract --version exited 1"


def test_check_tesseract_uses_the_apps_own_search(monkeypatch: pytest.MonkeyPatch) -> None:
    from biobench_sar.rag import ingestor

    monkeypatch.setattr(ingestor, "find_tesseract", lambda: None)
    assert hc.check_tesseract(system="Darwin")[0][1] == "FAIL"
