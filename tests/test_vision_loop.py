from __future__ import annotations

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
