from __future__ import annotations

import dataclasses
import json
import math
import time

import cv2
import numpy as np
import pytest

from biobench_sar.core.glare import GlareReport, measure_glare

W, H = 1920, 1080
WHITE = (255, 255, 255)


def _board(level: float = 225, noise: float = 6.0, size: tuple[int, int] = (W, H), seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    width, height = size
    img = rng.standard_normal((height, width, 3), dtype=np.float32) * noise + level
    return np.clip(img, 0, 255).astype(np.uint8)


def _lit(lighting: np.ndarray, noise: float = 6.0) -> np.ndarray:
    return np.clip(_board(0, noise).astype(np.float32) + lighting[..., None], 0, 255).astype(np.uint8)


def _horizontal_gradient(lo: float = 150, hi: float = 245) -> np.ndarray:
    return np.broadcast_to(np.linspace(lo, hi, W, dtype=np.float32), (H, W))


def _vignette(centre: float = 242, falloff: float = 70) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return centre - falloff * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)


def _ellipse_frac(axes: tuple[int, int]) -> float:
    return math.pi * axes[0] * axes[1] / (W * H)


def _no_glare(report: GlareReport) -> None:
    assert report.fraction == 0.0
    assert report.hotspots == []
    assert report.severe is False


@pytest.mark.parametrize("level", [200, 225, 240])
@pytest.mark.parametrize("size", [(W, H), (640, 480)])
@pytest.mark.parametrize("noise", [6.0, 10.0])
def test_evenly_lit_white_board_is_not_glare(level: int, size: tuple[int, int], noise: float) -> None:
    _no_glare(measure_glare(_board(level, noise, size)))


def test_dark_room_is_not_glare() -> None:
    _no_glare(measure_glare(_board(18, 5)))


@pytest.mark.parametrize("lighting", [_horizontal_gradient(), _vignette()], ids=["gradient", "vignette"])
def test_uneven_lighting_is_not_glare(lighting: np.ndarray) -> None:
    _no_glare(measure_glare(_lit(lighting)))


def test_ink_on_a_normal_board_is_not_glare() -> None:
    frame = _board()
    cv2.putText(frame, "y = -0.25 x^2", (300, 500), cv2.FONT_HERSHEY_SIMPLEX, 3, (30, 30, 30), 8)
    cv2.rectangle(frame, (200, 650), (900, 900), (40, 40, 40), 6)
    _no_glare(measure_glare(frame))


@pytest.mark.parametrize(
    ("axes", "found", "severe"),
    [((25, 20), False, False), ((70, 50), True, False), ((200, 120), True, True), ((420, 300), True, True)],
    ids=["speck", "medium", "large", "huge"],
)
def test_clipped_hotspot_by_size(axes: tuple[int, int], found: bool, severe: bool) -> None:
    frame = _board()
    cv2.ellipse(frame, (960, 540), axes, 0, 0, 360, WHITE, -1)
    report = measure_glare(frame)
    assert report.severe is severe
    if not found:
        _no_glare(report)
        return
    assert report.fraction == pytest.approx(_ellipse_frac(axes), rel=0.1)
    assert len(report.hotspots) == 1
    x, y, w, h = report.hotspots[0]
    assert (x, y, w, h) == pytest.approx((960 - axes[0], 540 - axes[1], 2 * axes[0], 2 * axes[1]), abs=12)


def test_hotspot_with_soft_halo_on_gradient_lighting() -> None:
    frame = _lit(_horizontal_gradient())
    spot = np.zeros((H, W), np.float32)
    cv2.ellipse(spot, (1500, 400), (150, 110), 0, 0, 360, 1.0, -1)
    spot = np.clip(cv2.GaussianBlur(spot, (0, 0), 25) * 1.6, 0, 1)[..., None]
    frame = (frame * (1 - spot) + 255 * spot).astype(np.uint8)
    report = measure_glare(frame)
    assert report.severe
    assert len(report.hotspots) == 1
    x, y, w, h = report.hotspots[0]
    assert x < 1500 < x + w and y < 400 < y + h
    assert 0.5 * _ellipse_frac((150, 110)) < report.fraction < 1.5 * _ellipse_frac((150, 110))


def test_hotspot_in_a_dark_room() -> None:
    frame = _board(18, 5)
    cv2.ellipse(frame, (500, 300), (90, 60), 0, 0, 360, WHITE, -1)
    report = measure_glare(frame)
    assert report.fraction == pytest.approx(_ellipse_frac((90, 60)), rel=0.15)
    assert len(report.hotspots) == 1


@pytest.mark.parametrize("ink", [30, 120])
def test_ink_inside_a_hotspot_counts_as_washed_out(ink: int) -> None:
    frame = _board()
    cv2.ellipse(frame, (960, 540), (200, 120), 0, 0, 360, WHITE, -1)
    clean = measure_glare(frame)
    cv2.putText(frame, "-0.25", (870, 520), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (ink,) * 3, 5)
    cv2.putText(frame, "x^2", (900, 600), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (ink,) * 3, 5)
    cv2.circle(frame, (1060, 560), 25, (ink,) * 3, -1)
    inked = measure_glare(frame)
    assert inked.fraction == pytest.approx(clean.fraction, rel=0.02)
    assert inked.hotspots == clean.hotspots


def test_bright_ring_does_not_claim_the_board_it_encloses() -> None:
    frame = _board()
    cv2.rectangle(frame, (0, 0), (W - 1, H - 1), WHITE, 40)  # e.g. a shiny board frame
    report = measure_glare(frame)
    ring = 2 * (W + H) * 20 / (W * H)
    assert report.fraction == pytest.approx(ring, rel=0.25)


def test_bright_projected_colour_is_not_glare() -> None:
    frame = _board()
    cv2.ellipse(frame, (960, 540), (200, 120), 0, 0, 360, (40, 140, 255), -1)  # the orange avatar
    _no_glare(measure_glare(frame))


def test_bright_fine_texture_is_not_glare() -> None:
    frame = _board()
    for y in range(300, 780, 24):
        for x in range(600, 1080, 24):
            frame[y : y + 24, x : x + 24] = 255 if ((x + y) // 24) % 2 == 0 else 180
    _no_glare(measure_glare(frame))


def test_hotspots_are_largest_first_and_capped_at_five() -> None:
    frame = _board()
    sizes = [50, 70, 55, 90, 52, 80, 60]
    for i, r in enumerate(sizes):
        cv2.circle(frame, (150 + i * 260, 300 + (i % 2) * 450), r, WHITE, -1)
    report = measure_glare(frame)
    assert len(report.hotspots) == 5
    widths = [w for _, _, w, _ in report.hotspots]
    assert widths == sorted(widths, reverse=True)
    assert widths[0] == pytest.approx(2 * max(sizes), abs=12)
    assert report.fraction == pytest.approx(sum(math.pi * r * r for r in sizes) / (W * H), rel=0.1)


def test_thresholds_are_respected() -> None:
    frame = _board()
    cv2.ellipse(frame, (960, 540), (70, 50), 0, 0, 360, (245, 245, 245), -1)
    assert measure_glare(frame).hotspots == []
    assert len(measure_glare(frame, clip_level=240).hotspots) == 1

    speck = _board()
    cv2.circle(speck, (960, 540), 30, WHITE, -1)
    assert measure_glare(speck).hotspots == []
    assert len(measure_glare(speck, min_blob_frac=0.0005).hotspots) == 1

    medium = _board()
    cv2.ellipse(medium, (960, 540), (70, 50), 0, 0, 360, WHITE, -1)
    assert measure_glare(medium).severe is False
    assert measure_glare(medium, severe_frac=0.004).severe is True


def test_grayscale_and_bgra_frames() -> None:
    frame = _board()
    cv2.ellipse(frame, (960, 540), (200, 120), 0, 0, 360, WHITE, -1)
    expected = measure_glare(frame)
    gray = measure_glare(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    bgra = measure_glare(cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA))
    assert gray.fraction == pytest.approx(expected.fraction, rel=0.05)
    assert bgra == expected


def test_small_frame_is_used_as_is() -> None:
    frame = np.full((100, 160, 3), 220, np.uint8)
    cv2.circle(frame, (60, 50), 20, WHITE, -1)
    report = measure_glare(frame)
    assert report.fraction == pytest.approx(math.pi * 400 / (100 * 160), rel=0.1)
    assert report.hotspots == [pytest.approx((40, 30, 41, 41), abs=2)]


def test_empty_frame() -> None:
    _no_glare(measure_glare(np.zeros((0, 0, 3), np.uint8)))


def test_report_is_frozen_and_json_ready() -> None:
    frame = _board()
    cv2.ellipse(frame, (960, 540), (200, 120), 0, 0, 360, WHITE, -1)
    report = measure_glare(frame)
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.severe = False  # type: ignore[misc]
    body = json.loads(json.dumps(report.to_dict()))
    assert body == {
        "fraction": round(report.fraction, 4),
        "hotspots": [list(report.hotspots[0])],
        "severe": True,
    }


def test_fast_at_1080p() -> None:
    frame = _board()
    cv2.ellipse(frame, (960, 540), (200, 120), 0, 0, 360, WHITE, -1)
    measure_glare(frame)
    times = []
    for _ in range(15):
        start = time.perf_counter()
        measure_glare(frame)
        times.append(time.perf_counter() - start)
    assert sorted(times)[len(times) // 2] < 0.010
