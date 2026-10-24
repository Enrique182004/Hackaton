#!/usr/bin/env python3
"""Hardware/environment preflight: cameras, mic, TTS, Tesseract, Ollama, displays."""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

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
        status = "PASS" if rms > 0.001 else "WARN"
        return [("microphone", status, f"{len(inputs)} input(s) [{names}]; RMS {rms:.4f} over 2 s")]
    except Exception as exc:
        return [("microphone", "FAIL", str(exc))]
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        pa.terminate()


def check_tts(settings: Settings, no_sound: bool) -> list[Result]:
    if no_sound:
        return [("tts", "WARN", "skipped (--no-sound)")]
    if platform.system() == "Darwin" and shutil.which("say"):
        rc = subprocess.run(["say", "Hardware check"], capture_output=True).returncode
        return [("tts (say)", "PASS" if rc == 0 else "FAIL", "played sample" if rc == 0 else f"exit {rc}")]
    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.say("Hardware check")
        engine.runAndWait()
        return [("tts (pyttsx3)", "PASS", "played sample")]
    except Exception as exc:
        return [("tts", "WARN", f"no local TTS ({exc}); edge-tts needs internet")]


def check_tesseract() -> list[Result]:
    try:
        import pytesseract

        return [("tesseract", "PASS", str(pytesseract.get_tesseract_version()))]
    except ImportError:
        pass
    except Exception as exc:
        return [("tesseract", "FAIL", f"binary not found ({exc})")]
    exe = shutil.which("tesseract")
    if not exe:
        return [("tesseract", "FAIL", "not found on PATH")]
    out = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout.splitlines()
    return [("tesseract", "PASS", out[0] if out else exe)]


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


def check_displays(projector: tuple[int, int] = (1920, 1080)) -> list[Result]:
    system = platform.system()
    if system == "Darwin":
        try:
            out = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True, timeout=15
            ).stdout
        except Exception as exc:
            return [("displays", "WARN", str(exc))]
        return [projector_display_result(parse_resolutions(out), projector)]
    try:
        from screeninfo import get_monitors

        mons = get_monitors()
        desc = "; ".join(f"{m.width}x{m.height}" for m in mons)
        return [("displays", "PASS" if len(mons) >= 2 else "WARN", f"{len(mons)} display(s): {desc}")]
    except ImportError:
        return [("displays", "WARN", "skipped (pip install screeninfo to enable)")]
    except Exception as exc:
        return [("displays", "WARN", str(exc))]


def main() -> int:
    parser = argparse.ArgumentParser(description="SAR whiteboard assistant hardware check")
    parser.add_argument("--camera", type=int, default=None, help="check only this camera index")
    parser.add_argument("--skip-mic", action="store_true")
    parser.add_argument("--skip-camera", action="store_true")
    parser.add_argument("--no-sound", action="store_true", help="skip audible TTS sample")
    parser.add_argument("--save-frame", metavar="PATH", help="save a camera snapshot (jpg)")
    args = parser.parse_args()
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
    results += check_tts(settings, args.no_sound)
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
