from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

_WORK_SIDE = 320  # longest side of the analysis copy; keeps 1080p far under the 10 ms budget
_MAX_CHROMA = 40  # the bulb's reflection is near-white; bright projected colour (the avatar) is not glare
_FLAT_STD = 6.0
_MIN_FLAT_SHARE = 0.5
_MAX_HOTSPOTS = 5
_OPEN = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


@dataclass(frozen=True)
class GlareReport:
    fraction: float
    hotspots: list[tuple[int, int, int, int]]
    severe: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "fraction": round(self.fraction, 4),
            "hotspots": [list(box) for box in self.hotspots],
            "severe": self.severe,
        }


def _fill_small_holes(mask: np.ndarray) -> np.ndarray:
    """Fill writing enclosed by a hotspot; a hole bigger than its ring is enclosed board, not lost ink."""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return mask
    links = hierarchy[0]
    filled = mask.copy()
    for i in range(len(contours)):
        if links[i][3] < 0:
            cv2.drawContours(filled, contours, i, 1, cv2.FILLED)
    for i, contour in enumerate(contours):
        parent = links[i][3]
        if parent >= 0 and cv2.contourArea(contour) > cv2.contourArea(contours[parent]) / 2:
            cv2.drawContours(filled, contours, i, 0, cv2.FILLED)
    return filled | mask


def measure_glare(
    frame: np.ndarray,
    *,
    clip_level: int = 250,
    min_blob_frac: float = 0.002,
    severe_frac: float = 0.01,
) -> GlareReport:
    """Find specular hotspots (the projector bulb's mirror image) that wash the board out.

    The frame is taken to be the board, so pass the projector-space (warped) frame. ``fraction`` is the
    share of it inside a hotspot, ink included; ``hotspots`` are (x, y, w, h) in ``frame`` coordinates.
    """
    if frame.size == 0:
        return GlareReport(0.0, [], False)
    height, width = frame.shape[:2]
    small = frame
    scale = _WORK_SIDE / max(height, width)
    if scale < 1.0:
        size = (max(round(width * scale), 1), max(round(height * scale), 1))
        # Area averaging also cancels sensor noise, so an evenly lit board never reaches the clip level.
        small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    if small.ndim == 3:
        channels = small[..., :3]
        value = channels.max(axis=2)
        chroma = value - channels.min(axis=2)
    else:
        value, chroma = small, np.zeros_like(small)

    bright = (value >= clip_level) & (chroma <= _MAX_CHROMA)
    mask = cv2.morphologyEx(bright.astype(np.uint8), cv2.MORPH_OPEN, _OPEN)
    # Marker strokes inside a hotspot are washed out too; bridging them counts them towards the area.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _CLOSE)
    mask = _fill_small_holes(mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return GlareReport(0.0, [], False)

    lit = value.astype(np.float32)
    mean = cv2.blur(lit, (3, 3))
    flat = cv2.blur(lit * lit, (3, 3)) - mean * mean <= _FLAT_STD**2
    bright_px = np.bincount(labels[bright], minlength=count)
    flat_px = np.bincount(labels[bright & flat], minlength=count)
    areas = stats[:, cv2.CC_STAT_AREA]
    total = value.size
    # Clipping leaves a flat plateau; a bright fine texture (print, a pattern) is not washed out.
    ok = (areas >= max(min_blob_frac * total, 1.0)) & (flat_px >= _MIN_FLAT_SHARE * bright_px)
    ok[0] = False
    kept = np.flatnonzero(ok)
    kept = kept[np.argsort(-areas[kept], kind="stable")]
    fraction = float(areas[kept].sum()) / total

    sx, sy = width / value.shape[1], height / value.shape[0]
    hotspots: list[tuple[int, int, int, int]] = []
    for i in kept[:_MAX_HOTSPOTS]:
        x, y, w, h = (int(v) for v in stats[i, :4])
        x0, y0 = int(x * sx), int(y * sy)
        x1, y1 = min(math.ceil((x + w) * sx), width), min(math.ceil((y + h) * sy), height)
        hotspots.append((x0, y0, x1 - x0, y1 - y0))
    return GlareReport(fraction=fraction, hotspots=hotspots, severe=bool(kept.size) and fraction >= severe_frac)
