from __future__ import annotations

import sys
import threading
import time
from typing import Any

import cv2
import numpy as np

from biobench_sar.core.homography import Calibration
from biobench_sar.core.vision_loop import VisionLoop, encode_jpeg, find_ink_regions


class FakeCapture:
    def __init__(self, fail_after: int | None = None) -> None:
        self.reads = 0
        self.released = False
        self.props: dict[int, float] = {}
        self.fail_after = fail_after

    def read(self) -> tuple[bool, Any]:
        self.reads += 1
        time.sleep(0.005)
        if self.fail_after is not None and self.reads > self.fail_after:
            return False, None
        return True, np.full((48, 64, 3), self.reads % 255, np.uint8)

    def release(self) -> None:
        self.released = True

    def set(self, prop: int, value: float) -> bool:
        self.props[prop] = value
        return True

    def isOpened(self) -> bool:
        return True


def _warped_shape(loop: VisionLoop) -> tuple[int, ...] | None:
    w = loop.latest_warped()
    return None if w is None else w.shape


def _wait(cond, timeout: float = 3.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_loop_captures_and_sets_props() -> None:
    caps: list[FakeCapture] = []

    def factory(i: int) -> FakeCapture:
        caps.append(FakeCapture())
        return caps[-1]

    loop = VisionLoop(width=320, height=240, capture_factory=factory)
    assert loop.latest_frame() is None and not loop.is_running
    loop.start()
    try:
        assert _wait(lambda: loop.latest_frame() is not None and loop.fps > 0)
        assert loop.is_running
        assert caps[0].props[cv2.CAP_PROP_FRAME_WIDTH] == 320
        assert caps[0].props[cv2.CAP_PROP_FRAME_HEIGHT] == 240
        frame = loop.latest_frame()
        assert frame is not None
        frame[:] = 7  # copy: mutating must not affect the loop
        again = loop.latest_frame()
        assert again is not None and not (again == 7).all()
    finally:
        loop.stop()
    assert not loop.is_running and caps[0].released


def test_warped_uses_calibration() -> None:
    loop = VisionLoop(capture_factory=lambda i: FakeCapture())
    assert loop.latest_warped() is None
    loop.start()
    try:
        assert _wait(lambda: loop.latest_warped() is not None)
        raw = loop.latest_warped()
        assert raw is not None and raw.shape == (48, 64, 3)
        loop.set_calibration(Calibration.identity((64, 48), (128, 96)))
        assert _wait(lambda: _warped_shape(loop) == (96, 128, 3))
        loop.set_calibration(None)
        assert _wait(lambda: _warped_shape(loop) == (48, 64, 3))
    finally:
        loop.stop()


def test_latest_frame_and_calibration_is_one_consistent_read() -> None:
    calibration = Calibration.identity((64, 48), (128, 96))
    loop = VisionLoop(capture_factory=lambda i: FakeCapture(), calibration=calibration)
    assert loop.latest_frame_and_calibration() == (None, calibration), "no frame yet, the calibration still known"
    loop.start()
    try:
        assert _wait(lambda: loop.latest_frame_and_calibration()[0] is not None)
        frame, got = loop.latest_frame_and_calibration()
        assert frame is not None and frame.shape == (48, 64, 3), "the raw camera frame, not the warped one"
        assert got is calibration
        frame[:] = 7
        again, _ = loop.latest_frame_and_calibration()
        assert again is not None and again is not frame, "a copy: the caller may draw on it"
        loop.set_calibration(None)
        assert loop.latest_frame_and_calibration()[1] is None
    finally:
        loop.stop()


def test_latest_frame_and_calibration_holds_the_lock() -> None:
    """Read under the same lock set_calibration takes: a recalibration can't land between frame and matrix."""
    loop = VisionLoop(capture_factory=lambda i: FakeCapture())
    loop._frame, loop._frame_time = np.zeros((4, 4, 3), np.uint8), time.monotonic()
    done = threading.Event()
    with loop._lock:
        reader = threading.Thread(target=lambda: (loop.latest_frame_and_calibration(), done.set()))
        reader.start()
        assert not done.wait(0.1), "blocked while the lock is held"
    assert done.wait(2)
    reader.join(2)


def test_reconnects_after_read_failure() -> None:
    caps: list[FakeCapture] = []

    def factory(i: int) -> FakeCapture:
        caps.append(FakeCapture(fail_after=3 if not caps else None))
        return caps[-1]

    loop = VisionLoop(capture_factory=factory)
    loop.start()
    try:
        assert _wait(lambda: len(caps) >= 2, timeout=5)
        assert caps[0].released
        n = caps[1].reads
        assert _wait(lambda: caps[1].reads > n + 2)
    finally:
        loop.stop()


def test_factory_exception_does_not_kill_thread() -> None:
    calls = {"n": 0}

    def factory(i: int) -> FakeCapture:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("no camera")
        return FakeCapture()

    loop = VisionLoop(capture_factory=factory)
    loop.start()
    try:
        assert _wait(lambda: loop.latest_frame() is not None, timeout=5)
    finally:
        loop.stop()


def test_encode_jpeg_roundtrip() -> None:
    img = np.full((40, 60, 3), 200, np.uint8)
    data = encode_jpeg(img, quality=90)
    assert data[:2] == b"\xff\xd8"
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == img.shape


def test_find_ink_regions_sorted_and_filtered() -> None:
    img = np.full((400, 600, 3), 255, np.uint8)
    cv2.rectangle(img, (50, 50), (250, 200), (0, 0, 0), 3)  # large
    cv2.rectangle(img, (400, 300), (460, 340), (0, 0, 0), 3)  # small
    cv2.line(img, (10, 390), (12, 392), (0, 0, 0), 1)  # speck
    regions = find_ink_regions(img, min_area=500)
    assert len(regions) == 2
    big, small = regions
    assert big["w"] * big["h"] > small["w"] * small["h"]
    assert abs(big["cx"] - 150) <= 10 and abs(big["cy"] - 125) <= 10
    assert abs(small["cx"] - 430) <= 10 and abs(small["cy"] - 320) <= 10
    assert all(isinstance(v, int) for r in regions for v in r.values())


def test_find_ink_regions_merges_strokes_and_caps() -> None:
    img = np.full((300, 300, 3), 255, np.uint8)
    for x in (100, 108, 116):
        cv2.line(img, (x, 100), (x, 160), (0, 0, 0), 2)
    assert len(find_ink_regions(img, min_area=100)) == 1
    many = np.full((300, 600, 3), 255, np.uint8)
    for i in range(8):
        cv2.rectangle(many, (10 + i * 70, 100), (50 + i * 70, 160), (0, 0, 0), -1)
    assert len(find_ink_regions(many, max_regions=3)) == 3


def test_find_ink_regions_blank() -> None:
    assert find_ink_regions(np.full((200, 200, 3), 255, np.uint8)) == []


def _flowchart(size: tuple[int, int] = (1080, 1920), wobble: bool = False):
    h, w = size
    img = np.full((h, w, 3), 255, np.uint8)
    centers = [(200 + i * 340, 540) for i in range(5)]
    bw, bh = 220, 120
    rng = np.random.default_rng(0)
    for i, (cx, cy) in enumerate(centers):
        x0, y0, x1, y1 = cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2
        if wobble:
            pts = np.array([(x0, y0), (cx, y0), (x1, y0), (x1, cy), (x1, y1), (cx, y1), (x0, y1), (x0, cy)])
            pts = pts + rng.integers(-4, 5, pts.shape)
            cv2.polylines(img, [pts.astype(np.int32)], True, (20, 20, 20), 4)
        else:
            cv2.rectangle(img, (x0, y0), (x1, y1), (20, 20, 20), 4)
        cv2.putText(img, f"Step {i}", (x0 + 30, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 3)
        if i < 4:
            cv2.arrowedLine(img, (x1, cy), (centers[i + 1][0] - bw // 2, cy), (20, 20, 20), 4, tipLength=0.25)
    return img, centers


def test_flowchart_boxes_split_into_regions() -> None:
    img, centers = _flowchart()
    regions = find_ink_regions(img)
    assert len(regions) == 5
    for cx, cy in centers:
        assert any(abs(r["cx"] - cx) <= 15 and abs(r["cy"] - cy) <= 15 for r in regions)


def test_wobbly_boxes_still_detected() -> None:
    img, centers = _flowchart(wobble=True)
    regions = find_ink_regions(img)
    assert len(regions) == 5
    for cx, cy in centers:
        assert any(abs(r["cx"] - cx) <= 15 and abs(r["cy"] - cy) <= 15 for r in regions)


def test_flowchart_keeps_standalone_text_blob() -> None:
    img, _ = _flowchart()
    cv2.putText(img, "note", (100, 900), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 4)
    regions = find_ink_regions(img)
    assert len(regions) == 6
    assert any(abs(r["cy"] - 880) <= 30 and r["cx"] < 300 for r in regions)


def test_find_ink_regions_fast_on_1080p() -> None:
    img, _ = _flowchart()
    find_ink_regions(img)
    start = time.perf_counter()
    for _ in range(5):
        find_ink_regions(img)
    per_call_ms = (time.perf_counter() - start) / 5 * 1000
    print(f"find_ink_regions 1080p: {per_call_ms:.1f} ms")
    assert per_call_ms < 150  # target <50 ms; loose bound for loaded CI machines


# --- stale frames / fps after reconnect -------------------------------------------------------


class BlockingCapture(FakeCapture):
    """Delivers `frames` frames, then hangs in read() until released (a frozen camera)."""

    def __init__(self, frames: int = 3) -> None:
        super().__init__()
        self.frames = frames
        self.gate = threading.Event()

    def read(self) -> tuple[bool, Any]:
        self.reads += 1
        if self.reads > self.frames:
            self.gate.wait(10)
            return False, None
        time.sleep(0.005)
        return True, np.full((48, 64, 3), 9, np.uint8)

    def release(self) -> None:
        super().release()
        self.gate.set()


def test_frame_cleared_when_camera_disconnects() -> None:
    caps: list[FakeCapture] = []

    def factory(i: int) -> FakeCapture:
        # First capture dies after 3 frames; every reopen fails to produce frames.
        caps.append(FakeCapture(fail_after=3 if not caps else 0))
        return caps[-1]

    loop = VisionLoop(capture_factory=factory)
    assert loop.frame_age_s is None
    loop.start()
    try:
        assert _wait(lambda: len(caps) >= 2, timeout=5)
        assert loop.latest_frame() is None
        assert loop.latest_warped() is None
        assert loop.frame_age_s is None
        assert loop.fps == 0.0
    finally:
        loop.stop()


def test_frame_age_and_staleness_cutoff() -> None:
    cap = BlockingCapture(frames=3)
    loop = VisionLoop(capture_factory=lambda i: cap, max_frame_age_s=0.15)
    loop.start()
    try:
        assert _wait(lambda: loop.latest_frame() is not None)
        age = loop.frame_age_s
        assert age is not None and age < 0.15
        # Camera freezes without erroring: the last frame ages out.
        assert _wait(lambda: loop.latest_frame() is None, timeout=2)
        assert loop.latest_warped() is None
        stale_age = loop.frame_age_s
        assert stale_age is not None and stale_age > 0.15
    finally:
        loop.stop()


def test_max_frame_age_can_be_disabled() -> None:
    cap = BlockingCapture(frames=2)
    loop = VisionLoop(capture_factory=lambda i: cap, max_frame_age_s=None)
    loop.start()
    try:
        assert _wait(lambda: cap.reads > 2)
        time.sleep(0.1)
        assert loop.latest_frame() is not None
    finally:
        loop.stop()


def test_fps_not_dragged_down_by_reconnect_gap() -> None:
    caps: list[FakeCapture] = []

    def factory(i: int) -> FakeCapture:
        caps.append(FakeCapture(fail_after=20 if not caps else None))
        return caps[-1]

    loop = VisionLoop(capture_factory=factory)
    loop.start()
    try:
        assert _wait(lambda: loop.fps > 0)
        # First fps sample after the reconnect must reflect read cadence, not the ~0.5 s outage.
        assert _wait(lambda: len(caps) >= 2 and caps[1].reads >= 2 and loop.fps > 0, timeout=5)
        assert loop.fps > 20
    finally:
        loop.stop()


# --- ink detection under sensor noise / uneven lighting ---------------------------------------


def _tune_module():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "tune_ink.py"
    spec = importlib.util.spec_from_file_location("tune_ink", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def test_blank_noisy_unevenly_lit_board_has_no_regions() -> None:
    tune = _tune_module()
    for seed in range(3):
        assert find_ink_regions(tune.blank_board(seed)) == []


def test_noisy_flowchart_finds_all_five_boxes() -> None:
    tune = _tune_module()
    for seed, stroke in ((0, 5), (1, 7), (2, 9)):
        regions = find_ink_regions(tune.flowchart_board(seed, stroke=stroke))
        hit, extra = tune.score(regions)
        assert hit == 5, (seed, stroke, regions)
        assert extra == 0, (seed, stroke, regions)  # the free-text note is allowed, nothing else
        assert len(regions) <= 6


def test_ink_thresholds_are_tunable() -> None:
    tune = _tune_module()
    img = tune.flowchart_board(0, stroke=7)
    assert len(find_ink_regions(img)) == 6
    # Large contrast offset: nothing clears the threshold.
    assert find_ink_regions(img, c=120) == []
    # Even block sizes are coerced to the next odd value rather than raising.
    assert len(find_ink_regions(img, block_size=30)) == len(find_ink_regions(img, block_size=31))
    assert len(find_ink_regions(img, max_regions=2)) == 2


# --- small maths marks: decimal points, minus signs, superscripts -----------------------------


def _number_with_dot(tune) -> tuple[np.ndarray, list]:
    """Clean '3.5' with a 6 px dot 8 px after the 3 and 14 px before the 5: [box 3, box dot, box 5]."""
    img = np.full((300, 400, 3), 235, np.uint8)
    three = tune.glyph(img, "3", 100, 150, 1.8, 5)
    x = three[0] + three[2] + 8
    point = tune.dot(img, x + 3, 147, 6.0)
    five = tune.glyph(img, "5", x + 6 + 14, 150, 1.8, 5)
    return img, [three, point, five]


def test_mark_joins_only_its_nearest_symbol() -> None:
    tune = _tune_module()
    img, (three, point, five) = _number_with_dot(tune)
    regions = find_ink_regions(img, **tune.MATH_PROFILE)
    assert len(regions) == 2
    assert tune.region_at(regions, point) is tune.region_at(regions, three)
    assert tune.region_at(regions, five) is not tune.region_at(regions, three)
    # Merging off: the dot clears min_area on its own and becomes a third, dot-sized region.
    alone = find_ink_regions(img, **{**tune.MATH_PROFILE, "small_merge_px": 0})
    assert len(alone) == 3
    dot_region = tune.region_at(alone, point)
    assert dot_region is not None and tune.covered([dot_region], [three, five]) == 0


def test_blob_below_min_area_needs_a_neighbour() -> None:
    tune = _tune_module()
    img, (three, point, _) = _number_with_dot(tune)
    tune.dot(img, 330, 60, 6.0)  # the same dot with nothing within reach
    regions = find_ink_regions(img, min_area=400)
    assert len(regions) == 2  # '3.' and '5'; the lone dot is below min_area
    assert tune.region_at(regions, point) is tune.region_at(regions, three)
    assert not any(r["x"] > 300 for r in regions)
    # Disabled, the old rule applies: the dot beside the 3 is dropped too.
    assert tune.region_at(find_ink_regions(img, min_area=400, small_merge_px=0), point) is None


def test_decimal_points_and_minus_signs_stay_inside_their_numbers() -> None:
    tune = _tune_module()
    for name, seed in (("spec", 0), ("spec", 1), ("stress", 2)):
        board = tune.math_board(seed, specks=tune.SPECKS, glare=tune.GLARE_GAIN, **tune.CONDITIONS[name])
        regions = find_ink_regions(board.image, **tune.MATH_PROFILE)
        for tokens in board.equations:
            for i, (ch, box) in enumerate(tokens):
                if ch not in (".", "-"):
                    continue
                region = tune.region_at(regions, box)
                neighbours = [b for c, b in tokens[max(i - 1, 0) : i + 2] if c not in tune.MARKS]
                assert region is not None, (name, seed, ch, box)
                assert tune.covered([region], neighbours) >= 1, (name, seed, ch, box, region)
        assert tune.covered(regions, board.marks) == len(board.marks)
        assert tune.covered(regions, board.glyphs) == len(board.glyphs)
        # 5-8 px marks far from any ink still clear min_area on their own.
        assert tune.covered(regions, board.lone) == len(board.lone)


def test_merging_keeps_marks_under_the_default_region_cap() -> None:
    tune = _tune_module()
    board = tune.math_board(0, glare=tune.GLARE_GAIN, **tune.CONDITIONS["spec"])
    assert tune.covered(find_ink_regions(board.image), board.marks) == len(board.marks)
    # Unmerged, every mark is a tiny region and the largest-first cap of 10 cuts them all.
    assert tune.covered(find_ink_regions(board.image, small_merge_px=0), board.marks) == 0


def test_isolated_speck_is_ignored() -> None:
    tune = _tune_module()
    for name, seed in (("spec", 0), ("stress", 1)):
        img = np.full((tune.H, tune.W, 3), 235, np.uint8)
        tune.dot(img, 960.4, 700.3, 2.0, 120)
        board = tune.degrade(img, seed, glare=tune.GLARE_GAIN, **tune.CONDITIONS[name])
        assert find_ink_regions(board, **tune.MATH_PROFILE) == []
        assert find_ink_regions(board) == []


def test_flowchart_unchanged_at_math_profile() -> None:
    tune = _tune_module()
    for seed, stroke in ((0, 5), (1, 7), (2, 9)):
        board = tune.flowchart_board(seed, stroke=stroke, glare=tune.GLARE_GAIN)
        regions = find_ink_regions(board, **tune.MATH_PROFILE)
        assert tune.score(regions) == (5, 0), (seed, stroke, regions)
        assert len(regions) == 6


def test_find_ink_regions_under_30ms_at_1080p() -> None:
    tune = _tune_module()
    board = tune.math_board(0, specks=tune.SPECKS, glare=tune.GLARE_GAIN, **tune.CONDITIONS["spec"]).image
    chart = tune.flowchart_board(0, stroke=7)
    assert board.shape == chart.shape == (1080, 1920, 3)
    # Best of 15: the floor is what the code costs; a mean also measures whatever else shares the CPU.
    math_ms = tune.best_ms(board, runs=15, **tune.MATH_PROFILE)
    chart_ms = tune.best_ms(chart, runs=15)
    print(f"find_ink_regions 1080p best-of-15: maths {math_ms:.1f} ms, flowchart {chart_ms:.1f} ms")
    assert math_ms < 30 and chart_ms < 30
