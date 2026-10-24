"""High-resolution crops of the inked part of the board for the VLM.

The full projector-space frame gets downscaled to ~1024 px before it reaches the VLM, which blurs
small math (decimal points, exponents, minus signs) away. Instead we crop the union of the ink
regions and render it straight from the raw camera frame at the camera's native pixel density.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from biobench_sar.core.homography import Calibration
from biobench_sar.core.vision_loop import encode_jpeg
from biobench_sar.schemas import InkRegion

Roi = tuple[int, int, int, int]


@dataclass(frozen=True)
class InkCrop:
    jpeg: bytes
    roi: Roi  # x, y, w, h in projector coordinates
    size: tuple[int, int]  # width, height of the crop image
    scale: float  # crop pixels per projector pixel

    def to_projector(self, x: float, y: float) -> tuple[float, float]:
        """Map a pixel of the crop image back to projector coordinates."""
        return self.roi[0] + x / self.scale, self.roi[1] + y / self.scale


def _fit_span(lo: int, hi: int, min_len: int, limit: int) -> tuple[int, int] | None:
    lo, hi = max(lo, 0), min(hi, limit)
    if hi <= lo:
        return None
    min_len = min(min_len, limit)
    if hi - lo < min_len:
        lo = min(max((lo + hi - min_len) // 2, 0), limit - min_len)
        hi = lo + min_len
    return lo, hi


def union_roi(
    regions: Sequence[InkRegion],
    projector_size: tuple[int, int],
    pad: int = 48,
    min_size: tuple[int, int] = (320, 180),
) -> Roi | None:
    """Padded union box of the ink regions, grown to min_size and clamped inside the frame."""
    if not regions:
        return None
    width, height = projector_size
    xs = _fit_span(min(r["x"] for r in regions) - pad, max(r["x"] + r["w"] for r in regions) + pad, min_size[0], width)
    ys = _fit_span(min(r["y"] for r in regions) - pad, max(r["y"] + r["h"] for r in regions) + pad, min_size[1], height)
    if xs is None or ys is None:
        return None
    return xs[0], ys[0], xs[1] - xs[0], ys[1] - ys[0]


def _camera_density(matrix: np.ndarray, roi: Roi) -> float | None:
    """Camera pixels per projector pixel over the ROI (geometric mean, from the area of its camera quad)."""
    x, y, w, h = roi
    corners = np.array([[x, y, 1], [x + w, y, 1], [x + w, y + h, 1], [x, y + h, 1]], dtype=np.float64)
    try:
        cam = corners @ np.linalg.inv(matrix).T
    except np.linalg.LinAlgError:
        return None
    hw = cam[:, 2]
    if not (np.all(hw > 0) or np.all(hw < 0)):  # the ROI crosses the camera's horizon line
        return None
    px, py = cam[:, 0] / hw, cam[:, 1] / hw
    area = 0.5 * abs(float(np.dot(px, np.roll(py, -1)) - np.dot(py, np.roll(px, -1))))
    density = math.sqrt(area / (w * h))
    return density if math.isfinite(density) and density > 0 else None


def _crop_matrix(matrix: np.ndarray, x0: int, y0: int, scale: float) -> np.ndarray:
    s = np.diag([scale, scale, 1.0])
    t = np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]])
    return s @ t @ matrix


def _render(frame: np.ndarray, matrix: np.ndarray, roi: Roi, scale: float, density: float) -> np.ndarray:
    x0, y0, w, h = roi
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    if scale >= density:
        return cv2.warpPerspective(frame, _crop_matrix(matrix, x0, y0, scale), size, flags=cv2.INTER_CUBIC)
    # warpPerspective silently treats INTER_AREA as INTER_LINEAR (no box filter, so thin strokes alias).
    # Warp once at up to 2x the target (native density bounds detail, 2x bounds cost on 4K cameras),
    # then box-filter down with resize, which does honour INTER_AREA.
    inner = min(density, 2.0 * scale)
    inner_size = (max(1, round(w * inner)), max(1, round(h * inner)))
    big = cv2.warpPerspective(frame, _crop_matrix(matrix, x0, y0, inner), inner_size, flags=cv2.INTER_LINEAR)
    return cv2.resize(big, size, interpolation=cv2.INTER_AREA)


def crop_ink_view(
    raw_frame: np.ndarray | None,
    calibration: Calibration,
    regions: Sequence[InkRegion],
    *,
    pad: int = 48,
    max_side: int = 1600,
    min_side: int = 320,
    quality: int = 90,
) -> InkCrop | None:
    """Perspective-corrected crop of the inked ROI, rendered from the raw camera frame at native detail.

    Regions are in projector coordinates (find_ink_regions on the warped frame). Returns None when
    there is nothing to crop or the calibration cannot be inverted over the ROI.
    """
    if raw_frame is None or not regions:
        return None
    roi = union_roi(regions, calibration.projector_size, pad=pad)
    if roi is None:
        return None
    matrix = np.asarray(calibration.matrix, dtype=np.float64)
    density = _camera_density(matrix, roi)
    if density is None:
        return None
    longest = max(roi[2], roi[3])
    scale = min(max(density, min_side / longest), max_side / longest)
    image = _render(raw_frame, matrix, roi, scale, density)
    return InkCrop(encode_jpeg(image, quality), roi, (image.shape[1], image.shape[0]), scale)


def crop_from_warped(
    warped_frame: np.ndarray | None,
    regions: Sequence[InkRegion],
    *,
    pad: int = 48,
    quality: int = 90,
) -> InkCrop | None:
    """Plain crop of the projector-space frame at 1:1, for when there is no meaningful calibration."""
    if warped_frame is None or not regions:
        return None
    height, width = warped_frame.shape[:2]
    roi = union_roi(regions, (width, height), pad=pad)
    if roi is None:
        return None
    x, y, w, h = roi
    return InkCrop(encode_jpeg(warped_frame[y : y + h, x : x + w], quality), roi, (w, h), 1.0)
