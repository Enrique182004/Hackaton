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

from biobench_sar.config import Settings  # noqa: E402

Result = tuple[str, str, str]  # (name, PASS|WARN|FAIL, detail)


def check_cameras(only: int | None, save_path: str | None) -> list[Result]:
    try:
        import cv2
    except ImportError:
        return [("camera", "FAIL", "opencv-python not installed")]
    results: list[Result] = []
    indices = [only] if only is not None else range(4)
    saved = False
    for index in indices:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            if only is not None:
                results.append((f"camera {index}", "FAIL", "cannot open"))
            continue
        frame = None
        count = 0
        start = time.time()
        for _ in range(90):
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
        results.append((f"camera {index}", "PASS" if fps >= 30 else "WARN", f"{w}x{h} @ {fps:.1f} fps (need >= 30)"))
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
        stream.close()
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples**2)))
        status = "PASS" if rms > 0.001 else "WARN"
        return [("microphone", status, f"{len(inputs)} input(s) [{names}]; RMS {rms:.4f} over 2 s")]
    except Exception as exc:
        return [("microphone", "FAIL", str(exc))]
    finally:
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
    results.append((f"model {wanted}", "PASS" if have else "FAIL", "pulled" if have else f"run: ollama pull {wanted}"))
    return results


def check_displays() -> list[Result]:
    system = platform.system()
    if system == "Darwin":
        try:
            out = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True, timeout=15
            ).stdout
        except Exception as exc:
            return [("displays", "WARN", str(exc))]
        res = [line.split(":", 1)[1].strip() for line in out.splitlines() if line.strip().startswith("Resolution:")]
        status = "PASS" if len(res) >= 2 else "WARN"
        return [("displays", status, f"{len(res)} display(s): {'; '.join(res) or 'unknown'} (projector needs a 2nd)")]
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
    settings = Settings()

    results: list[Result] = []
    if not args.skip_camera:
        results += check_cameras(args.camera if args.camera is not None else None, args.save_frame)
    if not args.skip_mic:
        results += check_mic()
    results += check_tts(settings, args.no_sound)
    results += check_tesseract()
    results += check_ollama(settings)
    results += check_displays()

    width = max(len(r[0]) for r in results)
    print(f"{'CHECK'.ljust(width)}  STATUS  DETAIL")
    for name, status, detail in results:
        print(f"{name.ljust(width)}  {status.ljust(6)}  {detail}")
    return 1 if any(r[1] == "FAIL" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
