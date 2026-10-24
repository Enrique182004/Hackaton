#!/usr/bin/env python3
"""Hardware/environment preflight: cameras, mic, TTS, Tesseract, Ollama, displays."""

from __future__ import annotations

import argparse
import ctypes
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biobench_sar.config import Settings
from biobench_sar.envfile import env_file_path, load_env_file

Result = tuple[str, str, str]  # (name, PASS|WARN|FAIL, detail)

CAMERA_WARMUP_FRAMES = 15
CAMERA_MEASURE_FRAMES = 90
CAMERA_PASS_FPS = 28.0
CAMERA_FAIL_FPS = 15.0


def camera_status(fps: float) -> str:
    if fps >= CAMERA_PASS_FPS:
        return "PASS"
    if fps >= CAMERA_FAIL_FPS:
        return "WARN"
    return "FAIL"


def camera_names() -> list[str]:
    """Camera names the OS reports (macOS only); OpenCV can't name devices, so this guides SAR_CAMERA_INDEX."""
    if platform.system() != "Darwin":
        return []
    try:
        out = subprocess.run(["system_profiler", "SPCameraDataType"], capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return []
    return parse_camera_names(out)


def parse_camera_names(system_profiler_output: str) -> list[str]:
    names = []
    for line in system_profiler_output.splitlines():
        if line.startswith("    ") and not line.startswith("      ") and line.rstrip().endswith(":"):
            names.append(line.strip().rstrip(":"))
    return names


def resolution_status(delivered: tuple[int, int], wanted: tuple[int, int]) -> str:
    return "PASS" if delivered[0] >= wanted[0] and delivered[1] >= wanted[1] else "WARN"


def check_cameras(only: int | None, save_path: str | None, wanted: tuple[int, int] = (1280, 720)) -> list[Result]:
    try:
        import cv2
    except ImportError:
        return [("camera", "FAIL", "opencv-python not installed")]
    results: list[Result] = []
    names = camera_names()
    if names:
        results.append(("camera names", "PASS", "; ".join(names) + " (index order usually follows this list)"))
    indices = [only] if only is not None else range(4)
    saved = False
    for index in indices:
        cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, wanted[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, wanted[1])
        if not cap.isOpened():
            cap.release()
            if only is not None:
                results.append((f"camera {index}", "FAIL", "cannot open"))
            continue
        for _ in range(CAMERA_WARMUP_FRAMES):
            cap.read()
        frame = None
        count = 0
        start = time.time()
        for _ in range(CAMERA_MEASURE_FRAMES):
            ok, img = cap.read()
            if ok:
                frame, count = img, count + 1
        elapsed = time.time() - start
        cap.release()
        if frame is None:
            results.append((f"camera {index}", "FAIL", "opened but no frames"))
            continue
        fps = count / elapsed if elapsed > 0 else 0.0
        h, w = frame.shape[:2]
        results.append(
            (
                f"camera {index}",
                camera_status(fps),
                f"{w}x{h} @ {fps:.1f} fps (PASS >= {CAMERA_PASS_FPS:g}, FAIL < {CAMERA_FAIL_FPS:g})",
            )
        )
        if resolution_status((w, h), wanted) != "PASS":
            results.append(
                (
                    f"camera {index} size",
                    "WARN",
                    f"gives {w}x{h}, asked {wanted[0]}x{wanted[1]} (SAR_CAMERA_WIDTH/HEIGHT)",
                )
            )
        if save_path and not saved:
            cv2.imwrite(save_path, frame)
            saved = True
            results.append(("snapshot", "PASS", f"saved {save_path}"))
    if not any(r[0].startswith("camera") and r[1] != "FAIL" for r in results):
        results.append(("camera", "FAIL", "no working camera found in indices 0..3"))
    return results


def mic_status(rms: float, platform: str = sys.platform) -> tuple[str, str]:
    """(status, what to do): PASS with sound, WARN when quiet (nobody talking), FAIL on digital zeros, which is
    what a muted microphone, or a macOS app without microphone access, records."""
    from biobench_sar.core.audio_trigger import DIGITAL_SILENCE_RMS, mic_silence_hint

    if rms < DIGITAL_SILENCE_RMS:
        return "FAIL", mic_silence_hint(platform)
    return ("PASS" if rms > 0.001 else "WARN"), ""


def check_mic() -> list[Result]:
    try:
        import numpy as np
        import pyaudio
    except ImportError as exc:
        return [("microphone", "FAIL", f"missing dependency: {exc.name}")]
    pa = pyaudio.PyAudio()
    stream = None
    try:
        inputs = [
            pa.get_device_info_by_index(i)
            for i in range(pa.get_device_count())
            if pa.get_device_info_by_index(i).get("maxInputChannels", 0) > 0
        ]
        if not inputs:
            return [("microphone", "FAIL", "no input devices")]
        names = ", ".join(str(d["name"]) for d in inputs[:3])
        rate = 16000
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=1024)
        data = b"".join(stream.read(1024, exception_on_overflow=False) for _ in range(int(rate * 2 / 1024)))
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples**2)))
        status, hint = mic_status(rms)
        detail = f"{len(inputs)} input(s) [{names}]; RMS {rms:.4f} over 2 s"
        return [("microphone", status, f"{detail}: only digital silence, so {hint}" if hint else detail)]
    except Exception as exc:
        return [("microphone", "FAIL", str(exc))]
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        pa.terminate()


VOICE_SAMPLES = {"en": "Hardware check.", "es": "Revisión del equipo."}


def check_tts(settings: Settings, skip: bool) -> list[Result]:
    """Make one sample in each kid voice. They come from Microsoft's online voice service: there is no offline voice,
    so without it the answers are shown but not spoken."""
    if skip:
        return [("voice", "WARN", "skipped (--no-voice)")]
    if settings.tts_backend != "edge":
        return [("voice", "WARN", f"SAR_TTS_BACKEND={settings.tts_backend}: answers are shown but not spoken")]
    import tempfile

    from biobench_sar.llm import tts_engine

    with tempfile.TemporaryDirectory() as folder:
        engine = tts_engine.TTSEngine(settings.tts_backend, settings.tts_voice, Path(folder))
        made = {lang: engine.speak(text, lang) for lang, text in VOICE_SAMPLES.items()}
    if all(made.values()):
        return [("voice", "PASS", "kid voices ready: " + ", ".join(engine.edge_voice(lang) for lang in made))]
    if engine.offline:
        return [
            (
                "voice",
                "WARN",
                "can't reach Microsoft's voice service, so answers will be shown but not spoken (there is no offline "
                "voice): check the internet connection",
            )
        ]
    failed = ", ".join(engine.edge_voice(lang) for lang, name in made.items() if not name)
    return [
        (
            "voice",
            "WARN",
            f"the voice service answers, but {failed} failed: check SAR_TTS_VOICE (answers in that language will be "
            "shown but not spoken)",
        )
    ]


def tesseract_install_hint(system: str) -> str:
    if system == "Windows":
        return "install: winget install UB-Mannheim.TesseractOCR"
    if system == "Darwin":
        return "install: brew install tesseract"
    return "install: sudo apt install tesseract-ocr"


def check_tesseract(
    find: Callable[[], str | None] | None = None,
    run: Callable[..., Any] = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
    system: str | None = None,
) -> list[Result]:
    """The same search the app makes before OCR: PATH, then the usual install folders (the Windows installer never
    adds itself to PATH)."""
    if find is None:
        from biobench_sar.rag.ingestor import find_tesseract as find
    system = platform.system() if system is None else system
    exe = find()
    if exe is None:
        return [("tesseract", "FAIL", f"not found on PATH or in the usual folders; {tesseract_install_hint(system)}")]
    try:
        proc = run([exe, "--version"], capture_output=True, text=True, timeout=15)
    except Exception as exc:
        return [("tesseract", "FAIL", f"{exe} does not run ({exc})")]
    lines = (proc.stdout or proc.stderr or "").splitlines()  # tesseract 3 printed its version on stderr
    if proc.returncode != 0:
        return [("tesseract", "FAIL", f"{exe} --version exited {proc.returncode}")]
    version = lines[0].strip() if lines else exe
    where = "" if which("tesseract") else f" at {exe} (not on PATH; the assistant finds it there)"
    return [("tesseract", "PASS", version + where)]


def check_ollama(settings: Settings) -> list[Result]:
    try:
        import requests

        resp = requests.get(f"{settings.vlm_base_url.rstrip('/')}/api/tags", timeout=3)
        resp.raise_for_status()
        names = [m.get("name", "") for m in resp.json().get("models", [])]
    except Exception as exc:
        status = "WARN" if settings.vlm_backend != "ollama" else "FAIL"
        return [("ollama", status, f"unreachable at {settings.vlm_base_url} ({exc.__class__.__name__})")]
    results: list[Result] = [("ollama", "PASS", f"reachable, {len(names)} model(s)")]
    wanted = settings.vlm_model
    have = any(n == wanted or n.split(":")[0] == wanted for n in names)
    if have:
        results.append((f"model {wanted}", "PASS", "pulled"))
    else:
        missing = "WARN" if settings.vlm_backend != "ollama" else "FAIL"
        results.append((f"model {wanted}", missing, f"run: ollama pull {wanted}"))
    return results


def parse_resolutions(system_profiler_output: str) -> list[tuple[int, int]]:
    """'Resolution: 1280 x 800 (WXGA)' / '2560 x 1664 Retina' -> [(1280, 800), (2560, 1664)]."""
    import re

    found = []
    for line in system_profiler_output.splitlines():
        if line.strip().startswith("Resolution:"):
            m = re.search(r"(\d+)\s*x\s*(\d+)", line)
            if m:
                found.append((int(m.group(1)), int(m.group(2))))
    return found


def projector_display_result(resolutions: list[tuple[int, int]], projector: tuple[int, int]) -> Result:
    listed = "; ".join(f"{w}x{h}" for w, h in resolutions) or "none"
    if len(resolutions) < 2:
        return (
            "projector",
            "WARN",
            f"only {len(resolutions)} display(s) ({listed}); connect the projector as an extended display",
        )
    if projector in resolutions:
        return ("projector", "PASS", f"a display runs at {projector[0]}x{projector[1]} ({listed})")
    return (
        "projector",
        "WARN",
        f"no display at {projector[0]}x{projector[1]} ({listed}); set the projector display to its native "
        "resolution or change SAR_PROJECTOR_WIDTH/HEIGHT",
    )


# ---- Windows displays through the Win32 API (ctypes only, no extra package) ------------------------------------
# WCHAR is 2 bytes on Windows but ctypes.c_wchar is 4 elsewhere, so it is c_uint16 here: the layouts (and the tests
# of their sizes) are the same on every OS.
ENUM_CURRENT_SETTINGS = 0xFFFFFFFF  # (DWORD)-1


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_int32),
        ("top", ctypes.c_int32),
        ("right", ctypes.c_int32),
        ("bottom", ctypes.c_int32),
    ]


class MONITORINFOEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_uint32),
        ("rcMonitor", RECT),
        ("rcWork", RECT),
        ("dwFlags", ctypes.c_uint32),
        ("szDevice", ctypes.c_uint16 * 32),
    ]


class DEVMODEW(ctypes.Structure):
    # wingdi.h; the printer/display union after dmFields is spelled out as its display half (16 bytes either way).
    _fields_ = [
        ("dmDeviceName", ctypes.c_uint16 * 32),
        ("dmSpecVersion", ctypes.c_uint16),
        ("dmDriverVersion", ctypes.c_uint16),
        ("dmSize", ctypes.c_uint16),
        ("dmDriverExtra", ctypes.c_uint16),
        ("dmFields", ctypes.c_uint32),
        ("dmPositionX", ctypes.c_int32),
        ("dmPositionY", ctypes.c_int32),
        ("dmDisplayOrientation", ctypes.c_uint32),
        ("dmDisplayFixedOutput", ctypes.c_uint32),
        ("dmColor", ctypes.c_int16),
        ("dmDuplex", ctypes.c_int16),
        ("dmYResolution", ctypes.c_int16),
        ("dmTTOption", ctypes.c_int16),
        ("dmCollate", ctypes.c_int16),
        ("dmFormName", ctypes.c_uint16 * 32),
        ("dmLogPixels", ctypes.c_uint16),
        ("dmBitsPerPel", ctypes.c_uint32),
        ("dmPelsWidth", ctypes.c_uint32),
        ("dmPelsHeight", ctypes.c_uint32),
        ("dmDisplayFlags", ctypes.c_uint32),
        ("dmDisplayFrequency", ctypes.c_uint32),
        ("dmICMMethod", ctypes.c_uint32),
        ("dmICMIntent", ctypes.c_uint32),
        ("dmMediaType", ctypes.c_uint32),
        ("dmDitherType", ctypes.c_uint32),
        ("dmReserved1", ctypes.c_uint32),
        ("dmReserved2", ctypes.c_uint32),
        ("dmPanningWidth", ctypes.c_uint32),
        ("dmPanningHeight", ctypes.c_uint32),
    ]


def wide_string(chars: Any) -> str:
    """A WCHAR[] field as text, up to its NUL."""
    return "".join(map(chr, chars)).split("\0", 1)[0]


def windows_display_sizes(user32: Any = None, functype: Any = None) -> list[tuple[int, int]]:
    """Every display's current mode in physical pixels: EnumDisplayMonitors -> GetMonitorInfoW for its device name
    -> EnumDisplaySettingsW. The monitor rectangle is only the fallback, since Windows scales it for DPI."""
    if user32 is None:
        user32 = ctypes.WinDLL("user32")  # type: ignore[attr-defined]  # Windows only
    if functype is None:
        functype = ctypes.WINFUNCTYPE  # type: ignore[attr-defined]
    # BOOL CALLBACK MonitorEnumProc(HMONITOR, HDC, LPRECT, LPARAM)
    monitor_proc = functype(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ssize_t)
    user32.EnumDisplayMonitors.argtypes = [ctypes.c_void_p, ctypes.c_void_p, monitor_proc, ctypes.c_ssize_t]
    user32.GetMonitorInfoW.argtypes = [ctypes.c_void_p, ctypes.POINTER(MONITORINFOEXW)]
    user32.EnumDisplaySettingsW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.POINTER(DEVMODEW)]
    sizes: list[tuple[int, int]] = []
    errors: list[BaseException] = []

    def on_monitor(hmonitor: int | None, hdc: int | None, rect: int | None, data: int) -> int:
        try:
            info = MONITORINFOEXW()
            info.cbSize = ctypes.sizeof(MONITORINFOEXW)
            if not user32.GetMonitorInfoW(hmonitor, ctypes.byref(info)):
                return 1
            mode = DEVMODEW()
            mode.dmSize = ctypes.sizeof(DEVMODEW)
            if user32.EnumDisplaySettingsW(wide_string(info.szDevice), ENUM_CURRENT_SETTINGS, ctypes.byref(mode)):
                sizes.append((int(mode.dmPelsWidth), int(mode.dmPelsHeight)))
            else:
                r = info.rcMonitor
                sizes.append((r.right - r.left, r.bottom - r.top))
            return 1
        except BaseException as exc:  # an exception can't cross the C callback; stop and re-raise it below
            errors.append(exc)
            return 0

    callback = monitor_proc(on_monitor)
    ok = user32.EnumDisplayMonitors(None, None, callback, 0)
    if errors:
        raise errors[0]
    if not ok:
        raise OSError("EnumDisplayMonitors failed")
    return sizes


def screeninfo_sizes() -> list[tuple[int, int]]:
    """Optional fallback (pip install screeninfo); raises ImportError when it isn't installed."""
    from screeninfo import get_monitors

    return [(int(m.width), int(m.height)) for m in get_monitors()]


def macos_display_sizes() -> list[tuple[int, int]]:
    out = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True, timeout=15).stdout
    return parse_resolutions(out)


def display_sizes(system: str) -> list[tuple[int, int]]:
    if system == "Darwin":
        return macos_display_sizes()
    if system == "Windows":
        try:
            return windows_display_sizes()
        except Exception as exc:
            try:
                return screeninfo_sizes()
            except ImportError:
                raise exc from None
    return screeninfo_sizes()


def check_displays(projector: tuple[int, int] = (1280, 800), system: str | None = None) -> list[Result]:
    system = platform.system() if system is None else system
    try:
        sizes = display_sizes(system)
    except ImportError:
        return [("displays", "WARN", "skipped (pip install screeninfo to list displays on this OS)")]
    except Exception as exc:
        return [("displays", "WARN", f"could not list the displays ({type(exc).__name__}: {exc})")]
    return [projector_display_result(sizes, projector)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SAR whiteboard assistant hardware check")
    parser.add_argument("--camera", type=int, default=None, help="check only this camera index")
    parser.add_argument("--skip-mic", action="store_true")
    parser.add_argument("--skip-camera", action="store_true")
    parser.add_argument("--no-voice", action="store_true", help="skip the kid-voice check (it needs the internet)")
    parser.add_argument("--save-frame", metavar="PATH", help="save a camera snapshot (jpg)")
    args = parser.parse_args(argv)
    applied = load_env_file()
    settings = Settings()

    results: list[Result] = []
    env_path = env_file_path()
    if applied:
        results.append(("settings file", "PASS", f"{env_path.name}: {len(applied)} setting(s)"))
    else:
        results.append(("settings file", "WARN", f"no {env_path.name} (copy sar.env.example to sar.env for your room)"))
    if not args.skip_camera:
        results += check_cameras(
            args.camera if args.camera is not None else None,
            args.save_frame,
            (settings.camera_width, settings.camera_height),
        )
    if not args.skip_mic:
        results += check_mic()
    results += check_tts(settings, args.no_voice)
    results += check_tesseract()
    results += check_ollama(settings)
    results += check_displays(settings.projector_size)

    width = max(len(r[0]) for r in results)
    print(f"{'CHECK'.ljust(width)}  STATUS  DETAIL")
    for name, status, detail in results:
        print(f"{name.ljust(width)}  {status.ljust(6)}  {detail}")
    return 1 if any(r[1] == "FAIL" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
