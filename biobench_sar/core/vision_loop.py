from __future__ import annotations

import threading
import time
from typing import Any, Callable, Protocol

import cv2
import numpy as np

from biobench_sar.core.homography import Calibration
from biobench_sar.schemas import InkRegion

_RECONNECT_DELAY = 0.5
_FPS_ALPHA = 0.1


class CaptureLike(Protocol):
    def read(self) -> tuple[bool, Any]: ...
    def release(self) -> None: ...
    def set(self, prop: int, value: float) -> bool: ...
    def isOpened(self) -> bool: ...


def _default_factory(index: int) -> CaptureLike:
    return cv2.VideoCapture(index)


class VisionLoop:
    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        calibration: Calibration | None = None,
        capture_factory: Callable[[int], CaptureLike] | None = None,
    ) -> None:
        self._index = camera_index
        self._width = width
        self._height = height
        self._calibration = calibration
        self._factory = capture_factory or _default_factory
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._fps = 0.0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="vision-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=3)
        self._thread = None

    def latest_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def latest_warped(self) -> np.ndarray | None:
        with self._lock:
            frame = self._frame
            calibration = self._calibration
        if frame is None:
            return None
        return calibration.warp_frame(frame) if calibration is not None else frame.copy()

    def set_calibration(self, calibration: Calibration | None) -> None:
        with self._lock:
            self._calibration = calibration

    def _open(self) -> CaptureLike | None:
        try:
            cap = self._factory(self._index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            return cap
        except Exception:
            return None

    def _run(self) -> None:
        cap: CaptureLike | None = None
        last = 0.0
        while not self._stop.is_set():
            if cap is None:
                cap = self._open()
                if cap is None or not cap.isOpened():
                    self._release(cap)
                    cap = None
                    self._stop.wait(_RECONNECT_DELAY)
                    continue
            try:
                ok, frame = cap.read()
            except Exception:
                ok, frame = False, None
            if not ok or frame is None:
                self._release(cap)
                cap = None
                self._stop.wait(_RECONNECT_DELAY)
                continue
            now = time.monotonic()
            if last:
                dt = now - last
                if dt > 0:
                    inst = 1.0 / dt
                    self._fps = inst if self._fps == 0.0 else self._fps + _FPS_ALPHA * (inst - self._fps)
            last = now
            with self._lock:
                self._frame = frame
        self._release(cap)

    @staticmethod
    def _release(cap: CaptureLike | None) -> None:
        if cap is None:
            return
        try:
            cap.release()
        except Exception:
            pass


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


def find_ink_regions(frame: np.ndarray, min_area: int = 150, max_regions: int = 10) -> list[InkRegion]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Adaptive threshold tolerates uneven projector/room lighting on the board.
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if cv2.contourArea(contour) >= min_area or w * h >= min_area:
            boxes.append((w * h, x, y, w, h))
    boxes.sort(reverse=True)
    return [
        InkRegion(x=int(x), y=int(y), w=int(w), h=int(h), cx=int(x + w // 2), cy=int(y + h // 2))
        for _, x, y, w, h in boxes[:max_regions]
    ]
