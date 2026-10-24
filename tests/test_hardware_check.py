from __future__ import annotations

import ctypes
import importlib.util
import itertools
import plistlib
import re
import shutil
import subprocess
import sys
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


BRIO = ("Logitech BRIO", "0x210000046d085e")
BUILT_IN = ("MacBook Air Camera", "6C707041-05AC-0010-000C-000000000001")


def test_opencv_order_is_unique_id_order_not_macos_order() -> None:
    # Hardware day: system_profiler listed the MacBook Air Camera first, but OpenCV opened the BRIO as index 0.
    assert hc.order_like_opencv([BUILT_IN, BRIO]) == ["Logitech BRIO", "MacBook Air Camera"]


def test_opencv_camera_order_is_empty_without_pyobjc_or_off_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "AVFoundation", None)  # import AVFoundation -> ImportError
    monkeypatch.setattr(hc.platform, "system", lambda: "Darwin")
    assert hc.opencv_camera_order() == []
    monkeypatch.setattr(hc.platform, "system", lambda: "Windows")
    assert hc.opencv_camera_order() == []


def test_opencv_camera_order_reads_video_and_muxed_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    class Device:
        def __init__(self, name: str, unique_id: str) -> None:
            self.name, self.unique_id = name, unique_id

        def localizedName(self) -> str:
            return self.name

        def uniqueID(self) -> str:
            return self.unique_id

    by_media = {"vide": [Device(*BUILT_IN)], "muxx": [Device(*BRIO)]}
    capture_device = types.SimpleNamespace(devicesWithMediaType_=by_media.get)
    avf = types.SimpleNamespace(AVMediaTypeVideo="vide", AVMediaTypeMuxed="muxx", AVCaptureDevice=capture_device)
    monkeypatch.setitem(sys.modules, "AVFoundation", avf)
    monkeypatch.setattr(hc.platform, "system", lambda: "Darwin")
    assert hc.opencv_camera_order() == ["Logitech BRIO", "MacBook Air Camera"]


def _fake_cv2(monkeypatch: pytest.MonkeyPatch, working: dict[int, tuple[int, int]]) -> None:
    """cv2 whose VideoCapture(i) opens only the indices in `working`, each giving frames of that (w, h)."""

    class VideoCapture:
        def __init__(self, index: int) -> None:
            self.size = working.get(index)

        def set(self, prop: int, value: float) -> bool:
            return True

        def isOpened(self) -> bool:
            return self.size is not None

        def read(self) -> tuple[bool, Any]:
            assert self.size is not None
            return True, types.SimpleNamespace(shape=(self.size[1], self.size[0], 3))

        def release(self) -> None: ...

    monkeypatch.setitem(
        sys.modules,
        "cv2",
        types.SimpleNamespace(VideoCapture=VideoCapture, CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4),
    )
    clock = itertools.count(0.0, 1.5)  # 90 frames per 1.5 s: 60 fps
    monkeypatch.setattr(hc, "time", types.SimpleNamespace(time=lambda: next(clock)))


def test_no_camera_fails_even_when_camera_names_are_listed(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cv2(monkeypatch, {})
    monkeypatch.setattr(hc, "opencv_camera_order", lambda: ["Logitech BRIO", "MacBook Air Camera"])
    results = hc.check_cameras(None, None)
    assert results[0][0] == "camera names"
    assert ("camera", "FAIL", "no working camera found in indices 0..3") in results
    assert hc.check_cameras(1, None)[1:] == [
        ("camera 1", "FAIL", "cannot open"),
        ("camera", "FAIL", "no working camera found in index 1"),
    ]


def test_too_slow_camera_still_fails_with_a_size_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cv2(monkeypatch, {0: (640, 480)})
    clock = itertools.count(0.0, 30.0)  # 90 frames per 30 s: 3 fps
    monkeypatch.setattr(hc, "time", types.SimpleNamespace(time=lambda: next(clock)))
    monkeypatch.setattr(hc, "opencv_camera_order", lambda: [])
    monkeypatch.setattr(hc, "camera_names", lambda: [])
    statuses = {name: status for name, status, _detail in hc.check_cameras(None, None)}
    assert statuses == {"camera 0": "FAIL", "camera 0 size": "WARN", "camera": "FAIL"}


def test_each_camera_row_names_the_device_at_that_index(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cv2(monkeypatch, {0: (1920, 1080), 1: (1920, 1080)})
    monkeypatch.setattr(hc, "opencv_camera_order", lambda: ["Logitech BRIO", "MacBook Air Camera"])
    results = {name: (status, detail) for name, status, detail in hc.check_cameras(None, None, (1920, 1080))}
    assert results["camera names"][1].startswith("0 Logitech BRIO; 1 MacBook Air Camera")
    assert results["camera 0"][0] == "PASS"
    assert results["camera 0"][1].startswith("Logitech BRIO: 1920x1080 @ 60.0 fps")
    assert results["camera 1"][1].startswith("MacBook Air Camera: 1920x1080")
    assert "camera" not in results


def test_macos_list_order_is_not_claimed_as_index_order(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cv2(monkeypatch, {0: (1920, 1080)})
    monkeypatch.setattr(hc, "opencv_camera_order", lambda: [])
    monkeypatch.setattr(hc, "camera_names", lambda: ["MacBook Air Camera", "Logitech BRIO"])
    results = {name: detail for name, _status, detail in hc.check_cameras(None, None)}
    assert "usually follows" not in results["camera names"]
    assert "not always the index order" in results["camera names"] and "--save-frame" in results["camera names"]
    assert results["camera 0"].startswith("1920x1080 @")


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


# ---- setup scripts ----------------------------------------------------------------------------------------------

SETUP_PS1 = ROOT / "scripts" / "setup_windows.ps1"
SETUP_SH = ROOT / "scripts" / "setup_mac.sh"


def _ps1() -> str:
    return SETUP_PS1.read_text(encoding="utf-8")


def _ps_function(text: str, name: str) -> str:
    start = text.index(f"function {name} {{")
    depth = 0
    for i in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0 and text[i] == "}":
            return text[start : i + 1]
    raise AssertionError(f"unbalanced {name}")


def test_setup_ps1_is_ascii() -> None:
    # Windows PowerShell 5.1 reads a .ps1 without a BOM as the ANSI code page: any non-ASCII letter would garble.
    assert SETUP_PS1.read_bytes().isascii()


def test_setup_ps1_braces_and_parens_balance() -> None:
    code = re.sub(r'"[^"\n]*"|\'[^\'\n]*\'|#.*', "", _ps1())  # drop strings and comments
    assert code.count("{") == code.count("}") and code.count("(") == code.count(")")


def test_setup_ps1_runs_programs_outside_stop_mode() -> None:
    # PowerShell 5.1 + "Stop": the first redirected stderr line of a program would end the script.
    text = _ps1()
    assert '$ErrorActionPreference = "Stop"' in text
    helpers = {name: _ps_function(text, name) for name in ("Invoke-Native", "Get-PythonVersion")}
    for body in helpers.values():
        assert '$ErrorActionPreference = "Continue"' in body and "$ErrorActionPreference = $saved" in body
        assert "finally" in body
    outside = text
    for body in helpers.values():
        outside = outside.replace(body, "")
    outside = re.sub(r"^\s*#.*$", "", outside, flags=re.M)  # the comment that explains the trap quotes it
    assert "2>$null" not in outside and "2>&1" not in outside
    calls = [line.strip() for line in outside.splitlines() if line.strip().startswith("& ")]
    assert calls == [], f"programs started without Invoke-Native: {calls}"
    assert not re.search(r"^\s*(py|winget|ollama|\$python)\s+[^=\s]", outside, re.M), (
        "every program goes through Invoke-Native"
    )


def test_setup_ps1_probes_python_without_assuming_the_py_launcher() -> None:
    text = _ps1()
    probe = _ps_function(text, "Get-PythonVersion")
    assert (
        "Get-Command $Command -ErrorAction SilentlyContinue" in probe
    )  # a missing `py` is not a CommandNotFound crash
    assert '"3.12", "3.11", "3.10"' in text and '"python", "python3"' in text
    assert "winget install -e --id Python.Python.3.12" in text


def test_setup_ps1_puts_tesseract_on_the_user_path() -> None:
    text = _ps1()
    finder = _ps_function(text, "Find-TesseractFolder")
    for root in ("$env:ProgramFiles", "${env:ProgramFiles(x86)}", "$env:LOCALAPPDATA"):
        assert root in finder
    assert '"Programs\\Tesseract-OCR"' in finder and '"Tesseract-OCR"' in finder
    adder = _ps_function(text, "Add-ToUserPath")
    assert '[Environment]::SetEnvironmentVariable("Path"' in adder and '"User")' in adder
    assert "Add-ToUserPath $tesseractDir" in text
    assert "UB-Mannheim.TesseractOCR" in text


def test_setup_ps1_creates_sar_env_once() -> None:
    text = _ps1()
    assert "if (-not (Test-Path sar.env) -and (Test-Path sar.env.example))" in text
    assert "Copy-Item sar.env.example sar.env" in text


def test_setup_ps1_parses_in_powershell() -> None:
    pwsh = shutil.which("pwsh") or shutil.which("powershell")
    if pwsh is None:
        pytest.skip("PowerShell is not installed")
    script = (
        "$errors = $null; [System.Management.Automation.Language.Parser]::ParseFile("
        f"'{SETUP_PS1}', [ref]$null, [ref]$errors) | Out-Null; if ($errors) {{ $errors; exit 1 }}"
    )
    out = subprocess.run([pwsh, "-NoProfile", "-Command", script], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr


def _stub(folder: Path, name: str, body: str) -> None:
    path = folder / name
    path.write_text("#!/bin/sh\n" + body + "\n")
    path.chmod(0o755)


@pytest.fixture
def mac_checkout(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """A copy of the setup script in a fake checkout, with brew/uv/ollama stubbed and nothing else on PATH."""
    if shutil.which("bash") is None:
        pytest.skip("bash is not installed")
    repo, bin_dir = tmp_path / "repo", tmp_path / "bin"
    (repo / "scripts").mkdir(parents=True)
    bin_dir.mkdir()
    shutil.copy(SETUP_SH, repo / "scripts" / "setup_mac.sh")
    shutil.copy(ROOT / "sar.env.example", repo / "sar.env.example")
    (repo / "requirements.txt").write_text("flask\n")
    for tool in ("dirname", "cp", "cat", "mkdir", "chmod", "sh"):
        found = shutil.which(tool)
        assert found, tool
        (bin_dir / tool).symlink_to(found)
    _stub(bin_dir, "brew", 'case "$1" in --prefix) echo /opt/fake ;; *) exit 0 ;; esac')
    _stub(bin_dir, "ollama", "exit 0")
    # `uv venv --seed --python 3.12 .venv`: a python that only records what pip was asked.
    _stub(
        bin_dir,
        "uv",
        'mkdir -p "$5/bin"\nprintf \'#!/bin/sh\\necho "$*" >> pip.log\\n\' > "$5/bin/python"\nchmod +x "$5/bin/python"',
    )
    return repo, {"PATH": str(bin_dir), "HOME": str(tmp_path)}


def test_setup_mac_creates_sar_env_from_the_kit_example(mac_checkout: tuple[Path, dict[str, str]]) -> None:
    repo, env = mac_checkout
    bash = shutil.which("bash")
    assert bash
    out = subprocess.run([bash, "scripts/setup_mac.sh"], cwd=repo, env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stdout + out.stderr
    assert (repo / "sar.env").read_bytes() == (repo / "sar.env.example").read_bytes()
    assert "Created sar.env" in out.stdout
    assert "-m pip install -r requirements.txt" in (repo / "pip.log").read_text()


def test_setup_mac_keeps_an_existing_sar_env(mac_checkout: tuple[Path, dict[str, str]]) -> None:
    repo, env = mac_checkout
    (repo / "sar.env").write_text("SAR_CAMERA_INDEX=0\n")
    bash = shutil.which("bash")
    assert bash
    out = subprocess.run([bash, "scripts/setup_mac.sh"], cwd=repo, env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stdout + out.stderr
    assert (repo / "sar.env").read_text() == "SAR_CAMERA_INDEX=0\n" and "Created sar.env" not in out.stdout


@pytest.fixture
def launcher_checkout(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """A checkout at a path nothing hard-codes, with swiftc/codesign/pgrep/open stubbed; plutil is the real one."""
    bash, plutil = shutil.which("bash"), shutil.which("plutil")
    if bash is None or plutil is None:
        pytest.skip("needs bash and plutil (macOS)")
    repo, bin_dir = tmp_path / "some checkout", tmp_path / "bin"
    (repo / "scripts").mkdir(parents=True)
    bin_dir.mkdir()
    for name in ("mac_dev_launcher.sh", "mac_dev_launcher.swift"):
        shutil.copy(ROOT / "scripts" / name, repo / "scripts" / name)
    for tool in ("dirname", "rm", "mkdir", "cat", "mv"):
        found = shutil.which(tool)
        assert found, tool
        (bin_dir / tool).symlink_to(found)
    (bin_dir / "plutil").symlink_to(plutil)
    _stub(bin_dir, "uname", "echo Darwin")
    _stub(bin_dir, "xcode-select", "echo /Library/Developer/CommandLineTools")
    _stub(bin_dir, "open", 'echo "$*" >> "$HOME/open.log"')
    _stub(bin_dir, "swiftc", 'echo "$*" >> "$HOME/swiftc.log"\n: > "$3"')  # swiftc -O -o OUT SRC
    _stub(bin_dir, "codesign", 'echo "$*" >> "$HOME/codesign.log"')
    _stub(bin_dir, "pgrep", "exit 1")
    return repo, {"PATH": str(bin_dir), "HOME": str(tmp_path), "BASH": bash}


def _build_launcher(repo: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [env["BASH"], "scripts/mac_dev_launcher.sh", *args],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_mac_dev_launcher_builds_for_its_own_checkout(launcher_checkout: tuple[Path, dict[str, str]]) -> None:
    repo, env = launcher_checkout
    out = _build_launcher(repo, env, "--open")
    assert out.returncode == 0, out.stdout + out.stderr
    app = repo / "build" / "sar-launcher" / "SAR Launcher.app"
    info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
    assert info["SARCheckout"] == str(repo.resolve())
    assert info["CFBundleExecutable"] == "SARLauncher" and (app / "Contents" / "MacOS" / "SARLauncher").exists()
    assert info["NSCameraUsageDescription"] and info["NSMicrophoneUsageDescription"]
    home = Path(env["HOME"])
    assert "scripts/mac_dev_launcher.swift" in (home / "swiftc.log").read_text()
    signed = repo.resolve() / "build" / "sar-launcher" / ".new" / "SAR Launcher.app"
    assert (home / "codesign.log").read_text().strip() == f"--force --sign - {signed}"
    assert (home / "open.log").read_text().strip() == str(app.resolve())
    assert not (app.parent / ".new").exists()


def test_mac_dev_launcher_rejects_unknown_options(launcher_checkout: tuple[Path, dict[str, str]]) -> None:
    repo, env = launcher_checkout
    out = _build_launcher(repo, env, "--run")
    assert out.returncode == 2 and "Usage:" in out.stderr


def test_mac_dev_launcher_failed_build_keeps_the_old_app(launcher_checkout: tuple[Path, dict[str, str]]) -> None:
    repo, env = launcher_checkout
    old = repo / "build" / "sar-launcher" / "SAR Launcher.app" / "Contents" / "MacOS" / "SARLauncher"
    old.parent.mkdir(parents=True)
    old.write_text("the launcher that works")
    _stub(Path(env["PATH"]), "swiftc", "exit 1")
    out = _build_launcher(repo, env)
    assert out.returncode != 0
    assert old.read_text() == "the launcher that works"


def test_mac_dev_launcher_will_not_rebuild_a_running_launcher(
    launcher_checkout: tuple[Path, dict[str, str]],
) -> None:
    repo, env = launcher_checkout
    _stub(Path(env["PATH"]), "pgrep", "exit 0")
    out = _build_launcher(repo, env)
    assert out.returncode == 1 and "Stop the assistant first" in out.stderr
    assert not (Path(env["HOME"]) / "swiftc.log").exists()


def test_mac_dev_launcher_reads_the_checkout_from_the_build_not_the_source() -> None:
    source = (ROOT / "scripts" / "mac_dev_launcher.swift").read_text(encoding="utf-8")
    assert 'forInfoDictionaryKey: "SARCheckout"' in source and "/Users/" not in source


def test_mac_dev_launcher_swift_parses() -> None:
    swiftc = shutil.which("swiftc")
    # /usr/bin/swiftc is a stub that offers to install the command line tools when they are missing.
    xcode_select = shutil.which("xcode-select")
    if swiftc is None or xcode_select is None or subprocess.run([xcode_select, "-p"], capture_output=True).returncode:
        pytest.skip("needs swiftc (macOS command line tools)")
    out = subprocess.run(
        [swiftc, "-parse", str(ROOT / "scripts" / "mac_dev_launcher.swift")], capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 0, out.stderr


def test_setup_mac_accepts_python_3_10_and_3_11() -> None:
    text = SETUP_SH.read_text(encoding="utf-8")
    assert "for candidate in python3.12 python3.11 python3.10" in text


@pytest.mark.parametrize(
    ("rms", "status"),
    [(0.0, "FAIL"), (0.00005, "FAIL"), (0.0005, "WARN"), (0.0136, "PASS")],
)
def test_mic_status_fails_on_digital_silence(rms: float, status: str) -> None:
    got, hint = hc.mic_status(rms, "darwin")
    assert got == status
    assert ("Privacy & Security > Microphone" in hint) is (status == "FAIL")


class _FakeStream:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def read(self, count: int, exception_on_overflow: bool = True) -> bytes:
        return self.data[: count * 2]

    def stop_stream(self) -> None: ...
    def close(self) -> None: ...


def _fake_pyaudio(monkeypatch: pytest.MonkeyPatch, sample: int) -> None:
    class PyAudio:
        def get_device_count(self) -> int:
            return 1

        def get_device_info_by_index(self, i: int) -> dict[str, Any]:
            return {"name": "MacBook Air Microphone", "maxInputChannels": 1}

        def open(self, **kwargs: Any) -> _FakeStream:
            return _FakeStream(sample.to_bytes(2, "little", signed=True) * 1024)

        def terminate(self) -> None: ...

    monkeypatch.setitem(sys.modules, "pyaudio", types.SimpleNamespace(PyAudio=PyAudio, paInt16=8))


def test_check_mic_fails_when_the_microphone_sends_only_zeros(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_pyaudio(monkeypatch, 0)
    [(name, status, detail)] = hc.check_mic()
    assert (name, status) == ("microphone", "FAIL")
    assert "RMS 0.0000 over 2 s" in detail and "only digital silence" in detail and "microphone" in detail.lower()


def test_check_mic_passes_with_sound(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_pyaudio(monkeypatch, 1000)
    [(_, status, detail)] = hc.check_mic()
    assert status == "PASS" and "MacBook Air Microphone" in detail
