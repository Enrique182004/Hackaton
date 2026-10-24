from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_MARGIN_FRAC = 0.06
log = logging.getLogger(__name__)


def _board_geometry(width: int, height: int, cols: int, rows: int) -> tuple[float, float, float]:
    """Return (square size, origin x, origin y) of the centered board."""
    margin = min(width, height) * _MARGIN_FRAC
    square = min((width - 2 * margin) / (cols + 1), (height - 2 * margin) / (rows + 1))
    ox = (width - square * (cols + 1)) / 2
    oy = (height - square * (rows + 1)) / 2
    return square, ox, oy


def generate_checkerboard(width: int, height: int, cols: int = 9, rows: int = 6) -> np.ndarray:
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    square, ox, oy = _board_geometry(width, height, cols, rows)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                x0, y0 = round(ox + c * square), round(oy + r * square)
                x1, y1 = round(ox + (c + 1) * square), round(oy + (r + 1) * square)
                img[y0:y1, x0:x1] = 0
    return img


def projector_corner_points(width: int, height: int, cols: int = 9, rows: int = 6) -> np.ndarray:
    square, ox, oy = _board_geometry(width, height, cols, rows)
    # Row-major with x fastest: the order cv2.findChessboardCorners uses.
    pts = [(ox + (c + 1) * square, oy + (r + 1) * square) for r in range(rows) for c in range(cols)]
    return np.array(pts, dtype=np.float32)


def detect_corners(frame: np.ndarray, cols: int = 9, rows: int = 6) -> np.ndarray | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners.reshape(-1, 2).astype(np.float32)


def compute_homography(camera_pts: np.ndarray, projector_pts: np.ndarray) -> np.ndarray:
    cam = np.asarray(camera_pts, dtype=np.float32).reshape(-1, 2)
    proj = np.asarray(projector_pts, dtype=np.float32).reshape(-1, 2)
    if len(cam) < 4 or len(cam) != len(proj):
        raise ValueError("need at least 4 matching point pairs")
    try:
        matrix, _ = cv2.findHomography(cam, proj, cv2.RANSAC, 3.0)
    except cv2.error as exc:
        raise ValueError(f"homography estimation failed: {exc}") from exc
    if matrix is None:
        raise ValueError("homography estimation failed")
    return matrix


@dataclass
class Calibration:
    matrix: np.ndarray
    projector_size: tuple[int, int]

    def camera_to_projector(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        if pts.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return cv2.perspectiveTransform(pts, self.matrix.astype(np.float64)).reshape(-1, 2)

    def warp_frame(self, frame: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(frame, self.matrix, self.projector_size)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"matrix": self.matrix.tolist(), "projector_size": list(self.projector_size)}
        tmp = path.with_name(path.name + ".tmp")
        try:
            tmp.write_text(json.dumps(payload))
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: Path, expected_projector_size: tuple[int, int] | None = None) -> Calibration | None:
        try:
            data = json.loads(Path(path).read_text())
            matrix = np.array(data["matrix"], dtype=np.float64)
            width, height = (int(v) for v in data["projector_size"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if matrix.shape != (3, 3):
            return None
        if expected_projector_size is not None and (width, height) != tuple(expected_projector_size):
            log.warning(
                "ignoring calibration %s: saved projector size %s != expected %s",
                path,
                (width, height),
                tuple(expected_projector_size),
            )
            return None
        return cls(matrix, (width, height))

    @classmethod
    def identity(cls, camera_size: tuple[int, int], projector_size: tuple[int, int]) -> Calibration:
        matrix = np.diag([projector_size[0] / camera_size[0], projector_size[1] / camera_size[1], 1.0])
        return cls(matrix, projector_size)


def calibrate_from_frame(
    frame: np.ndarray, projector_size: tuple[int, int], cols: int = 9, rows: int = 6
) -> Calibration | None:
    corners = detect_corners(frame, cols, rows)
    if corners is None:
        return None
    targets = projector_corner_points(projector_size[0], projector_size[1], cols, rows)
    try:
        matrix = compute_homography(corners, targets)
    except ValueError:
        return None
    return Calibration(matrix, projector_size)
