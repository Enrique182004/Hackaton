from __future__ import annotations

import time
from collections.abc import Callable
from typing import NamedTuple

import cv2
import numpy as np
import pytest

from biobench_sar.core.homography import Calibration
from biobench_sar.core.ink_crop import InkCrop, crop_from_warped, crop_ink_view, union_roi
from biobench_sar.core.vision_loop import encode_jpeg, find_ink_regions
from biobench_sar.schemas import InkRegion

CAM = (2560, 1440)
PROJ = (1920, 1080)
# Keystoned view of the projected area inside the camera frame.
QUAD = np.float32([[180, 120], [2380, 170], [2330, 1330], [230, 1290]])
RECT = np.float32([[0, 0], [PROJ[0], 0], [PROJ[0], PROJ[1]], [0, PROJ[1]]])
INK = (20, 20, 20)
FONT = cv2.FONT_HERSHEY_SIMPLEX
# What the pipeline sends today: the whole warped frame, downscaled by the VLM client to this longest side.
BASELINE_SIDE = 1024


class Scene(NamedTuple):
    camera: np.ndarray
    calibration: Calibration
    warped: np.ndarray
    regions: list[InkRegion]
    dot_camera: tuple[int, int]
    dot_projector: tuple[float, float]


def _region(x: int, y: int, w: int, h: int) -> InkRegion:
    return InkRegion(x=x, y=y, w=w, h=h, cx=x + w // 2, cy=y + h // 2)


def _decode(jpeg: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)


def _blob(gray: np.ndarray, center: tuple[float, float], radius: int) -> tuple[int, tuple[float, float]]:
    """Area and centroid of the dark blob nearest to center (0 area if none)."""
    cx, cy = round(center[0]), round(center[1])
    x0, y0 = cx - radius, cy - radius
    window = (gray[y0 : cy + radius + 1, x0 : cx + radius + 1] < 128).astype(np.uint8)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(window, connectivity=8)
    if n <= 1:
        return 0, center
    best = 1 + int(np.argmin([np.hypot(*(centroids[i] - radius)) for i in range(1, n)]))
    return int(stats[best, cv2.CC_STAT_AREA]), (x0 + centroids[best][0], y0 + centroids[best][1])


def _median_ms(fn: Callable[[], object], runs: int = 7) -> float:
    fn()  # warm-up
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


@pytest.fixture(scope="module")
def scene() -> Scene:
    """2560x1440 camera frame of '3.5 - x^2' in small type; the decimal point is a ~7 px dot."""
    cam = np.full((CAM[1], CAM[0], 3), 80, np.uint8)
    cv2.fillConvexPoly(cam, QUAD.astype(np.int32), (235, 235, 235))
    x, y = 1100, 760
    (digit_w, _), _ = cv2.getTextSize("3", FONT, 0.9, 2)
    cv2.putText(cam, "3", (x, y), FONT, 0.9, INK, 2, cv2.LINE_AA)
    dot = (x + digit_w + 12, y - 3)
    cv2.circle(cam, dot, 3, INK, -1, cv2.LINE_AA)
    cv2.putText(cam, "5 - x^2", (dot[0] + 15, y), FONT, 0.9, INK, 2, cv2.LINE_AA)
    calibration = Calibration(cv2.getPerspectiveTransform(QUAD, RECT), PROJ)
    warped = calibration.warp_frame(cam)
    regions = find_ink_regions(warped)
    dot_proj = calibration.camera_to_projector(np.float32([dot]))[0]
    return Scene(cam, calibration, warped, regions, dot, (float(dot_proj[0]), float(dot_proj[1])))


def test_union_roi_pads_and_grows_to_min_size() -> None:
    regions = [_region(100, 200, 50, 20), _region(400, 210, 30, 30)]
    # x: 100..430 padded to 52..478 (already wider than 320); y: 200..240 padded to 152..288, grown to 180.
    assert union_roi(regions, PROJ) == (52, 130, 426, 180)
    assert union_roi(regions, PROJ, pad=0, min_size=(0, 0)) == (100, 200, 330, 40)


def test_union_roi_covers_all_ink(scene: Scene) -> None:
    assert scene.regions
    roi = union_roi(scene.regions, PROJ)
    assert roi is not None
    x, y, w, h = roi
    ys, xs = np.nonzero(cv2.cvtColor(scene.camera, cv2.COLOR_BGR2GRAY) < 60)
    ink = scene.calibration.camera_to_projector(np.stack([xs, ys], axis=1).astype(np.float32))
    (ix0, iy0), (ix1, iy1) = ink.min(axis=0), ink.max(axis=0)
    assert x <= ix0 - 48 and y <= iy0 - 48
    assert x + w >= ix1 + 48 and y + h >= iy1 + 48
    assert w >= 320 and h >= 180
    assert x >= 0 and y >= 0 and x + w <= PROJ[0] and y + h <= PROJ[1]


def test_union_roi_clamps_at_frame_edges() -> None:
    assert union_roi([_region(0, 0, 10, 10)], PROJ) == (0, 0, 320, 180)
    assert union_roi([_region(1910, 1070, 10, 10)], PROJ) == (1600, 900, 320, 180)
    # Padded box runs past the right edge: clipped there, not shifted left.
    assert union_roi([_region(1500, 500, 415, 40)], PROJ) == (1452, 430, 468, 180)
    assert union_roi([_region(10, 10, 5, 5)], (200, 100)) == (0, 0, 200, 100)
    assert union_roi([_region(2500, 10, 20, 20)], PROJ, pad=10) is None


def test_none_when_nothing_to_crop(scene: Scene) -> None:
    assert union_roi([], PROJ) is None
    assert crop_ink_view(scene.camera, scene.calibration, []) is None
    assert crop_ink_view(None, scene.calibration, scene.regions) is None
    assert crop_from_warped(scene.warped, []) is None
    assert crop_from_warped(None, scene.regions) is None


def test_crop_keeps_native_detail(scene: Scene) -> None:
    crop = crop_ink_view(scene.camera, scene.calibration, scene.regions)
    assert crop is not None
    image = _decode(crop.jpeg)
    assert (image.shape[1], image.shape[0]) == crop.size
    assert crop.roi == union_roi(scene.regions, PROJ)
    assert crop.size == (round(crop.roi[2] * crop.scale), round(crop.roi[3] * crop.scale))

    shrink = BASELINE_SIDE / max(PROJ)
    full = _decode(encode_jpeg(scene.warped))
    base = cv2.resize(full, (round(PROJ[0] * shrink), round(PROJ[1] * shrink)), interpolation=cv2.INTER_AREA)
    base = _decode(encode_jpeg(base))
    px, py = scene.dot_projector
    base_area, _ = _blob(base, (px * shrink, py * shrink), radius=4)
    crop_area, _ = _blob(image, ((px - crop.roi[0]) * crop.scale, (py - crop.roi[1]) * crop.scale), radius=8)
    camera_area, _ = _blob(cv2.cvtColor(scene.camera, cv2.COLOR_BGR2GRAY), scene.dot_camera, radius=8)

    assert crop.scale > 2 * shrink
    assert crop_area >= 2.5 * max(base_area, 1)  # measured: ~40 px vs ~9 px
    assert crop_area >= 0.7 * camera_area  # the camera's own detail survives


def test_crop_is_perspective_corrected(scene: Scene) -> None:
    crop = crop_ink_view(scene.camera, scene.calibration, scene.regions)
    assert crop is not None
    px, py = scene.dot_projector
    area, centroid = _blob(
        _decode(crop.jpeg), ((px - crop.roi[0]) * crop.scale, (py - crop.roi[1]) * crop.scale), radius=8
    )
    assert area > 0
    mapped = crop.to_projector(*centroid)
    assert mapped == pytest.approx((px, py), abs=1.5)


def test_to_projector_inverts_roi_and_scale() -> None:
    crop = InkCrop(b"", (100, 50, 320, 180), (640, 360), 2.0)
    assert crop.to_projector(64, 36) == (132.0, 68.0)
    assert crop.to_projector(0, 0) == (100.0, 50.0)


def test_identity_calibration_matches_plain_crop() -> None:
    frame = np.full((PROJ[1], PROJ[0], 3), 235, np.uint8)
    cv2.putText(frame, "3.5 - x^2", (900, 600), FONT, 0.8, INK, 2, cv2.LINE_AA)
    regions = find_ink_regions(frame)
    calibration = Calibration.identity(PROJ, PROJ)
    view = crop_ink_view(frame, calibration, regions)
    plain = crop_from_warped(frame, regions)
    assert view is not None and plain is not None
    assert (view.roi, view.size, view.scale) == (plain.roi, plain.size, plain.scale)
    x, y, w, h = plain.roi
    assert plain.size == (w, h) and plain.scale == 1.0
    assert plain.jpeg == encode_jpeg(frame[y : y + h, x : x + w], 90)
    assert np.array_equal(_decode(view.jpeg), _decode(plain.jpeg))


@pytest.mark.parametrize(
    ("camera_size", "region", "size", "scale"),
    [
        # 4K camera: keep its 2 px per projector px instead of capping at projector resolution.
        ((3840, 2160), _region(800, 500, 100, 40), (640, 360), 2.0),
        # Low-res camera: never upsample past its native 0.5 px per projector px.
        ((960, 540), _region(500, 300, 904, 404), (500, 250), 0.5),
        # Very low-res camera: still at least min_side on the longest side.
        ((640, 360), _region(800, 500, 100, 40), (320, 180), 1.0),
        # Whole board on a 4K camera: capped at max_side.
        ((3840, 2160), _region(0, 0, *PROJ), (1600, 900), 1600 / 1920),
    ],
)
def test_output_scale_rules(
    camera_size: tuple[int, int], region: InkRegion, size: tuple[int, int], scale: float
) -> None:
    frame = np.full((camera_size[1], camera_size[0], 3), 235, np.uint8)
    crop = crop_ink_view(frame, Calibration.identity(camera_size, PROJ), [region])
    assert crop is not None
    assert crop.size == size
    assert crop.scale == pytest.approx(scale)
    assert _decode(crop.jpeg).shape == (size[1], size[0])


def test_shrinking_is_antialiased() -> None:
    stripes = np.zeros((2160, 3840), np.uint8)
    stripes[:, ::2] = 255
    crop = crop_ink_view(stripes, Calibration.identity((3840, 2160), PROJ), [_region(0, 0, *PROJ)])
    assert crop is not None and crop.size == (1600, 900)
    # A single warpPerspective treats INTER_AREA as bilinear, which turns 1 px stripes into moire.
    m = np.diag([crop.scale / 2, crop.scale / 2, 1.0])
    aliased = cv2.warpPerspective(stripes, m, crop.size, flags=cv2.INTER_AREA)
    assert _decode(crop.jpeg).std() < 0.5 * aliased.std()  # measured: ~25 vs ~76


def test_crop_at_projector_corner(scene: Scene) -> None:
    crop = crop_ink_view(scene.camera, scene.calibration, [_region(1900, 1060, 20, 20)])
    assert crop is not None
    assert crop.roi == (1600, 900, 320, 180)
    image = _decode(crop.jpeg)
    assert (image.shape[1], image.shape[0]) == crop.size
    assert image.mean() > 200  # board pixels, not the dark area outside the projected quad


def test_degenerate_calibration_returns_none(scene: Scene) -> None:
    regions = [_region(0, 0, *PROJ)]
    assert crop_ink_view(scene.camera, Calibration(np.zeros((3, 3)), PROJ), regions) is None
    # Projector x=1000 maps to the camera's line at infinity: the ROI cannot be pulled back.
    beyond_horizon = np.linalg.inv(np.array([[1.0, 0, 0], [0, 1.0, 0], [0.001, 0, -1.0]]))
    assert crop_ink_view(scene.camera, Calibration(beyond_horizon, PROJ), regions) is None


def test_crop_is_fast_on_2560x1440(scene: Scene) -> None:
    math_ms = _median_ms(lambda: crop_ink_view(scene.camera, scene.calibration, scene.regions))
    board_ms = _median_ms(lambda: crop_ink_view(scene.camera, scene.calibration, [_region(0, 0, *PROJ)]))
    # measured on an M4 Air: ~0.2 ms for the math ROI, ~12 ms for the whole board
    assert math_ms < 30
    assert board_ms < 30
