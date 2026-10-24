"""Sweep ink-detection thresholds on synthetic noisy boards.

Usage: python scripts/tune_ink.py [--quick]
The scene generators are reused by tests/test_vision_loop.py.
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biobench_sar.core.vision_loop import find_ink_regions

H, W = 1080, 1920
BOX_W, BOX_H = 220, 120
CENTERS = [(200 + i * 340, 540) for i in range(5)]
NOTE_CENTER = (150, 885)
# "spec" is the target environment; "stress" is well beyond it.
CONDITIONS: dict[str, dict[str, float]] = {
    "spec": dict(sigma=6.0, band=35, blur=1.2),
    "stress": dict(sigma=10.0, band=45, blur=2.0),
}
STROKES = (5, 7, 9)


def degrade(img: np.ndarray, seed: int, sigma: float = 6.0, band: int = 35, blur: float = 1.2) -> np.ndarray:
    """Sensor noise, a darker lighting band, and slight blur."""
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32)
    if blur > 0:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    h, w = out.shape[:2]
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    ramp = np.exp(-(((xs * 0.8 + ys * 0.4) - 0.6) ** 2) / (2 * 0.12**2))
    out -= band * ramp[..., None]
    out -= 8 * ((xs - 0.5) ** 2 + (ys - 0.5) ** 2)[..., None]
    out += rng.normal(0, sigma, out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def blank_board(seed: int = 0, **kw: float) -> np.ndarray:
    return degrade(np.full((H, W, 3), 235, np.uint8), seed, **kw)


def flowchart_board(seed: int = 0, stroke: int = 6, note: bool = True, **kw: float) -> np.ndarray:
    img = np.full((H, W, 3), 235, np.uint8)
    ink = (30, 30, 30)
    for i, (cx, cy) in enumerate(CENTERS):
        x0, y0, x1, y1 = cx - BOX_W // 2, cy - BOX_H // 2, cx + BOX_W // 2, cy + BOX_H // 2
        cv2.rectangle(img, (x0, y0), (x1, y1), ink, stroke)
        cv2.putText(img, f"Step {i}", (x0 + 30, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, ink, max(stroke // 2, 2))
        if i < 4:
            cv2.arrowedLine(img, (x1, cy), (CENTERS[i + 1][0] - BOX_W // 2, cy), ink, stroke, tipLength=0.25)
    if note:
        cv2.putText(img, "note", (100, 900), cv2.FONT_HERSHEY_SIMPLEX, 1.5, ink, stroke)
    return degrade(img, seed, **kw)


def score(regions: list, note: bool = True) -> tuple[int, int]:
    """(boxes matched, spurious regions). The free-text note is not spurious."""
    hit = sum(any(abs(r["cx"] - cx) <= 20 and abs(r["cy"] - cy) <= 20 for r in regions) for cx, cy in CENTERS)
    extra = len(regions) - hit
    if note and any(abs(r["cx"] - NOTE_CENTER[0]) <= 60 and abs(r["cy"] - NOTE_CENTER[1]) <= 40 for r in regions):
        extra -= 1
    return hit, max(extra, 0)


def evaluate(block_size: int, c: int, min_area: int, cond: dict[str, float], seeds: range) -> tuple[int, int, int, int]:
    """(blank spurious, boxes missed, chart spurious, #charts) over all seeds/strokes."""
    kw = dict(min_area=min_area, block_size=block_size, c=c)
    blank = sum(len(find_ink_regions(blank_board(s, **cond), **kw)) for s in seeds)
    miss = spur = n = 0
    for s in seeds:
        for stroke in STROKES:
            hit, extra = score(find_ink_regions(flowchart_board(s, stroke=stroke, **cond), **kw))
            miss += 5 - hit
            spur += extra
            n += 1
    return blank, miss, spur, n


def main() -> None:
    quick = "--quick" in sys.argv
    seeds = range(2 if quick else 4)
    blocks, cs, areas = (15, 21, 31, 41, 51), (5, 8, 10, 12, 15, 20, 25), (150,)
    probe = flowchart_board(0, **CONDITIONS["spec"])
    print("block   C min_area | " + " | ".join(f"{n}: blank-spur boxes-miss chart-spur" for n in CONDITIONS) + " | ms")
    for block, c, area in itertools.product(blocks, cs, areas):
        cells = []
        for cond in CONDITIONS.values():
            b, m, s, n = evaluate(block, c, area, cond, seeds)
            cells.append(f"{b:>3}/{len(seeds)} {m:>4}/{5 * n} {s:>3}/{n}")
        find_ink_regions(probe, block_size=block, c=c)
        t = time.perf_counter()
        for _ in range(5):
            find_ink_regions(probe, block_size=block, c=c)
        ms = (time.perf_counter() - t) / 5 * 1000
        print(f"{block:>5} {c:>3} {area:>8} | " + " | ".join(cells) + f" | {ms:.1f}")


if __name__ == "__main__":
    main()
