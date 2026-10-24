from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

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
        max_frame_age_s: float | None = 2.0,
    ) -> None:
        self._index = camera_index
        self._width = width
        self._height = height
        self._calibration = calibration
        self._factory = capture_factory or _default_factory
        self._lock = threading.Lock()
        self._max_frame_age_s = max_frame_age_s
        self._frame: np.ndarray | None = None
        self._frame_time = 0.0
        self._fps = 0.0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_age_s(self) -> float | None:
        with self._lock:
            if self._frame is None:
                return None
            return max(time.monotonic() - self._frame_time, 0.0)

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

    def _fresh_frame(self) -> np.ndarray | None:
        # Caller holds the lock.
        if self._frame is None:
            return None
        if self._max_frame_age_s is not None and time.monotonic() - self._frame_time > self._max_frame_age_s:
            return None
        return self._frame

    def latest_frame(self) -> np.ndarray | None:
        with self._lock:
            frame = self._fresh_frame()
            return None if frame is None else frame.copy()

    def latest_warped(self) -> np.ndarray | None:
        with self._lock:
            frame = self._fresh_frame()
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

    def _drop_frame(self) -> None:
        with self._lock:
            self._frame = None
            self._frame_time = 0.0
            self._fps = 0.0

    def _run(self) -> None:
        cap: CaptureLike | None = None
        last = 0.0
        while not self._stop.is_set():
            if cap is None:
                last = 0.0  # a reconnect gap must not enter the fps average
                cap = self._open()
                if cap is None or not cap.isOpened():
                    self._release(cap)
                    cap = None
                    self._drop_frame()
                    self._stop.wait(_RECONNECT_DELAY)
                    continue
            try:
                ok, frame = cap.read()
            except Exception:
                ok, frame = False, None
            if not ok or frame is None:
                self._release(cap)
                cap = None
                last = 0.0
                self._drop_frame()
                self._stop.wait(_RECONNECT_DELAY)
                continue
            now = time.monotonic()
            with self._lock:
                if last:
                    dt = now - last
                    if dt > 0:
                        inst = 1.0 / dt
                        self._fps = inst if self._fps == 0.0 else self._fps + _FPS_ALPHA * (inst - self._fps)
                self._frame = frame
                self._frame_time = now
            last = now
        self._release(cap)
        self._drop_frame()

    @staticmethod
    def _release(cap: CaptureLike | None) -> None:
        if cap is None:
            return
        with contextlib.suppress(Exception):
            cap.release()


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


_MIN_BOX_AREA = 600
_BOX_PAD = 3
_ERASE_PAD = 6
_RING = 15
_MARK_CHUNK = 1024


def _box_rects(mask: np.ndarray, min_area: int) -> list[tuple[int, int, int, int]]:
    """Closed, roughly rectangular outlines: their enclosed hole is a rect-like child contour."""
    sealed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, hierarchy = cv2.findContours(sealed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    rects: list[tuple[int, int, int, int]] = []
    for contour, node in zip(contours, hierarchy[0], strict=False):
        if node[3] < 0:  # outer contour, not a hole
            continue
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if area < max(min_area, _MIN_BOX_AREA) or area < 0.75 * w * h:
            continue
        if len(cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)) > 8:
            continue
        rects.append((x, y, w, h))

    # An enclosing frame around other boxes is not a step; keep the innermost boxes.
    def contains(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
        return a != b and a[0] <= b[0] and a[1] <= b[1] and a[0] + a[2] >= b[0] + b[2] and a[1] + a[3] >= b[1] + b[3]

    return [r for r in rects if not any(contains(r, o) for o in rects)]


def _absorb_marks(
    symbols: list[tuple[int, int, int, int]], marks: list[tuple[int, int, int, int]], reach: int
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """Grow each symbol (x0, y0, x1, y1) box over the marks within `reach` px; return (symbols, unclaimed marks).

    A mark joins only its nearest symbol, so "3.5" keeps its dot and "-4" its minus sign.
    """
    if not symbols or not marks or reach <= 0:
        return symbols, marks
    big, small = np.array(symbols), np.array(marks)
    nearest = np.empty(len(small), np.intp)
    near = np.empty(len(small), bool)
    # Chunked so a noisy frame with thousands of specks never builds one huge marks x symbols matrix.
    for i in range(0, len(small), _MARK_CHUNK):
        part = small[i : i + _MARK_CHUNK]
        dx = np.maximum(np.maximum(big[None, :, 0] - part[:, None, 2], part[:, None, 0] - big[None, :, 2]), 0)
        dy = np.maximum(np.maximum(big[None, :, 1] - part[:, None, 3], part[:, None, 1] - big[None, :, 3]), 0)
        gap2 = dx * dx + dy * dy
        best = gap2.argmin(axis=1)
        nearest[i : i + len(part)] = best
        near[i : i + len(part)] = gap2[np.arange(len(part)), best] <= reach * reach
    for col, grow in ((0, np.minimum), (1, np.minimum), (2, np.maximum), (3, np.maximum)):
        grow.at(big[:, col], nearest[near], small[near, col])
    return [tuple(int(v) for v in b) for b in big], [tuple(int(v) for v in m) for m in small[~near]]


def find_ink_regions(
    frame: np.ndarray,
    min_area: int = 150,
    max_regions: int = 10,
    block_size: int = 31,
    c: int = 15,
    small_merge_px: int = 24,
) -> list[InkRegion]:
    """Bounding boxes of marker ink in `frame`'s own coordinates, largest first.

    Blobs smaller than `small_merge_px`² (or `min_area`) within `small_merge_px` of a larger free-ink
    blob extend that blob's box instead of standing alone (decimal points, minus signs, superscripts);
    the rest must still reach `min_area`. Flowchart boxes never absorb marks. 0 disables merging.
    """
    block_size = max(int(block_size), 3)
    if block_size % 2 == 0:
        block_size += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Adaptive threshold tolerates uneven projector/room lighting on the board.
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c)
    height, width = mask.shape[:2]

    found: list[tuple[int, int, int, int, int]] = []  # (area, x, y, w, h)
    rest = mask
    ring: np.ndarray | None = None
    rects = _box_rects(mask, min_area)
    if rects:
        rest = mask.copy()
        zone = np.zeros_like(mask)
        for x, y, w, h in rects:
            x0, y0 = max(x - _BOX_PAD, 0), max(y - _BOX_PAD, 0)
            x1, y1 = min(x + w + _BOX_PAD, width), min(y + h + _BOX_PAD, height)
            found.append(((x1 - x0) * (y1 - y0), x0, y0, x1 - x0, y1 - y0))
            cv2.rectangle(zone, (x0 - _ERASE_PAD, y0 - _ERASE_PAD), (x1 + _ERASE_PAD, y1 + _ERASE_PAD), 255, -1)
        rest[zone > 0] = 0
        ring = cv2.dilate(zone, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * _RING + 1, 2 * _RING + 1))) > 0
        ring &= zone == 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    rest = cv2.morphologyEx(rest, cv2.MORPH_CLOSE, kernel)
    rest = cv2.dilate(rest, kernel, iterations=1)
    if ring is not None:
        # Connector strokes that touched a box leave stubs; drop any ink blob adjacent to a box.
        _, labels = cv2.connectedComponents(rest, connectivity=8)
        touching = np.unique(labels[ring & (rest > 0)])
        rest[np.isin(labels, touching[touching > 0])] = 0
    contours, _ = cv2.findContours(rest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    reach = max(int(small_merge_px), 0)
    # Areas are taken after the dilation above: a decimal point measures ~170-290 px², a minus sign
    # ~260-390, a superscript ~420. They clear min_area, so alone they become the smallest regions and the
    # first cut by max_regions; anything under reach² is instead a mark of the symbol beside it.
    mark_area = max(min_area, reach * reach)
    symbols: list[tuple[int, int, int, int]] = []  # (x0, y0, x1, y1)
    marks: list[tuple[int, int, int, int]] = []
    for contour in contours:
        # A bbox is never smaller than its contour, so the bbox alone decides "contour or bbox >= area".
        x, y, w, h = cv2.boundingRect(contour)
        (symbols if w * h >= mark_area else marks).append((x, y, x + w, y + h))
    symbols, marks = _absorb_marks(symbols, marks, reach)
    for x0, y0, x1, y1 in symbols + marks:
        if (x1 - x0) * (y1 - y0) >= min_area:
            found.append(((x1 - x0) * (y1 - y0), x0, y0, x1 - x0, y1 - y0))
    found.sort(reverse=True)
    return [
        InkRegion(x=int(x), y=int(y), w=int(w), h=int(h), cx=int(x + w // 2), cy=int(y + h // 2))
        for _, x, y, w, h in found[:max_regions]
    ]
