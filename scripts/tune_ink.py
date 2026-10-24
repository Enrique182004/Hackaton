"""Sweep ink-detection thresholds on synthetic noisy boards.

Usage: python scripts/tune_ink.py [--quick] [--flowchart]
Default: math-symbol sweep (small marks kept, glare spurious, flowchart regression, ms per call).
--flowchart: the original block/C sweep on flowchart boards only.
The scene generators are reused by tests/test_vision_loop.py.
"""

from __future__ import annotations

import itertools
import statistics
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
INK = (30, 30, 30)
FONT = cv2.FONT_HERSHEY_SIMPLEX
# Specular hotspot in the top-right, clear of the flowchart row; it clips to 255 at its core.
GLARE_CENTER, GLARE_AXES, GLARE_SOFT = (1500, 240), (280, 150), 10.0
GLARE_GAIN = 90.0
EQUATIONS = (
    ("3.5 - x^2 = 0.25", (120, 240)),
    ("-4x + 7 = 2/3", (120, 560)),
    ("y = -0.5x", (1300, 290)),  # inside the glare
)
LONE_Y = 960
# Recommended SAR_INK_* values for maths boards (see the sweep); tests pin behaviour at these.
MATH_PROFILE = dict(min_area=60, block_size=31, c=15, max_regions=20, small_merge_px=24)
Box = tuple[int, int, int, int]  # x, y, w, h


def degrade(
    img: np.ndarray, seed: int, sigma: float = 6.0, band: int = 35, blur: float = 1.2, glare: float = 0.0
) -> np.ndarray:
    """Sensor noise, a darker lighting band, slight blur and an optional clipped glare hotspot."""
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
    if glare > 0:
        spot = np.zeros((h, w), np.float32)
        cv2.ellipse(spot, GLARE_CENTER, GLARE_AXES, 0, 0, 360, 1.0, -1)
        out += glare * cv2.GaussianBlur(spot, (0, 0), GLARE_SOFT)[..., None]
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


def dot(img: np.ndarray, cx: float, cy: float, diameter: float) -> Box:
    r = diameter / 2
    cv2.circle(img, (round(cx * 16), round(cy * 16)), round(r * 16), INK, -1, cv2.LINE_AA, shift=4)
    return round(cx - r), round(cy - r), round(diameter), round(diameter)


def minus(img: np.ndarray, x: int, cy: int, length: int, thick: int) -> Box:
    y = cy - thick // 2
    cv2.rectangle(img, (x, y), (x + length - 1, y + thick - 1), INK, -1)
    return x, y, length, thick


def superscript(img: np.ndarray, ch: str, x: int, base: int, scale: float, stroke: int) -> Box:
    (w, h), _ = cv2.getTextSize(ch, FONT, scale, stroke)
    cv2.putText(img, ch, (x, base), FONT, scale, INK, stroke, cv2.LINE_AA)
    return x, base - h, w, h


def write_math(
    img: np.ndarray, text: str, origin: tuple[int, int], rng: np.random.Generator, scale: float = 1.8, stroke: int = 5
) -> list[Box]:
    """Marker-style equation; '.', '-' and '^2' are drawn as separate small marks whose boxes are returned.

    Each mark sits 8-12 px from its neighbours, just past the 7x7 dilation, so it is its own component.
    """
    x, base = origin
    cap = cv2.getTextSize("0", FONT, scale, stroke)[0][1] - stroke
    marks: list[Box] = []
    raise_next = False
    for ch in text:
        gap = int(rng.integers(8, 13))
        if ch == "^":
            raise_next = True
        elif ch == " ":
            x += int(0.45 * cap)
        elif ch == ".":
            d = float(rng.uniform(5.0, 8.0))
            marks.append(dot(img, x + gap + d / 2, base - d / 2, d))
            x += gap + round(d) + int(rng.integers(8, 13))
        elif ch == "-":
            length, thick = int(rng.integers(16, 21)), int(rng.integers(3, 6))
            marks.append(minus(img, x + gap, base - cap // 2, length, thick))
            x += gap + length + int(rng.integers(8, 13))
        elif raise_next:
            box = superscript(img, ch, x + gap, base - int(0.55 * cap), scale * 0.42, max(stroke // 2, 2))
            marks.append(box)
            x += gap + box[2] + int(rng.integers(8, 13))
            raise_next = False
        else:
            cv2.putText(img, ch, (x, base), FONT, scale, INK, stroke, cv2.LINE_AA)
            x += cv2.getTextSize(ch, FONT, scale, stroke)[0][0]
    return marks


def math_board(seed: int = 0, **kw: float) -> tuple[np.ndarray, list[Box], list[Box]]:
    """(image, small marks inside equations, lone small marks far from any other ink)."""
    rng = np.random.default_rng(1000 + seed)
    img = np.full((H, W, 3), 235, np.uint8)
    marks = [m for text, origin in EQUATIONS for m in write_math(img, text, origin, rng)]
    lone = [
        dot(img, 300, LONE_Y, float(rng.uniform(5.0, 8.0))),
        minus(img, 700, LONE_Y, 18, 4),
        superscript(img, "2", 1100, LONE_Y + 9, 1.8 * 0.42, 2),
    ]
    return degrade(img, seed, **kw), marks, lone


def covered(regions: list, marks: list[Box]) -> int:
    """Marks whose centre lies inside some region: kept on their own or merged into a neighbour."""
    n = 0
    for x, y, w, h in marks:
        mx, my = x + w / 2, y + h / 2
        n += any(r["x"] <= mx <= r["x"] + r["w"] and r["y"] <= my <= r["y"] + r["h"] for r in regions)
    return n


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


def median_ms(img: np.ndarray, runs: int = 7, **kw: int) -> float:
    find_ink_regions(img, **kw)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        find_ink_regions(img, **kw)
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


def flowchart_sweep(seeds: range) -> None:
    blocks, cs, areas = (15, 21, 31, 41, 51), (5, 8, 10, 12, 15, 20, 25), (150,)
    probe = flowchart_board(0, **CONDITIONS["spec"])
    print("block   C min_area | " + " | ".join(f"{n}: blank-spur boxes-miss chart-spur" for n in CONDITIONS) + " | ms")
    for block, c, area in itertools.product(blocks, cs, areas):
        cells = []
        for cond in CONDITIONS.values():
            b, m, s, n = evaluate(block, c, area, cond, seeds)
            cells.append(f"{b:>3}/{len(seeds)} {m:>4}/{5 * n} {s:>3}/{n}")
        print(
            f"{block:>5} {c:>3} {area:>8} | " + " | ".join(cells) + f" | {median_ms(probe, block_size=block, c=c):.1f}"
        )


def _gray_boards(seeds: range, cond: dict[str, float]) -> tuple[list, list, list]:
    """(math boards with their marks, blank boards, flowchart boards), all grey and all with glare."""
    cond = {**cond, "glare": GLARE_GAIN}
    maths = [math_board(s, **cond) for s in seeds]
    gray = [(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), marks, lone) for img, marks, lone in maths]
    blanks = [cv2.cvtColor(blank_board(s, **cond), cv2.COLOR_BGR2GRAY) for s in seeds]
    charts = [cv2.cvtColor(flowchart_board(s, stroke=k, **cond), cv2.COLOR_BGR2GRAY) for s in seeds for k in STROKES]
    return gray, blanks, charts


def math_sweep(seeds: range, cap: int = 10) -> None:
    areas, blocks, cs = (30, 40, 50, 60, 75, 100, 150), (21, 31, 41), (10, 15, 20)
    probe = math_board(0, **CONDITIONS["spec"], glare=GLARE_GAIN)[0]
    for name, cond in CONDITIONS.items():
        maths, blanks, charts = _gray_boards(seeds, cond)
        n_marks, n_lone = sum(len(m) for _, m, _ in maths), sum(len(lo) for _, _, lo in maths)
        print(f"\n## {name} {cond} + glare {GLARE_GAIN:g}")
        print(f"marks: '.', '-', '^2' inside equations ({n_marks}); lone: same marks far from ink ({n_lone});")
        print(f"merge = small_merge_px 24, off = 0; @{cap} = max_regions {cap}, all = uncapped;")
        print(
            f"blank: regions on {len(blanks)} blank boards; chart: boxes missed / spurious over {len(charts)} boards."
        )
        print("area blk  C | marks@all merge/off | marks@cap merge/off | lone | regions merge/off | blank | chart | ms")
        for area, block, c in itertools.product(areas, blocks, cs):
            kw = dict(min_area=area, block_size=block, c=c, max_regions=1000)
            all_on = all_off = cap_on = cap_off = lone_kept = 0
            n_on, n_off = [], []
            for img, marks, lone in maths:
                on, off = find_ink_regions(img, **kw), find_ink_regions(img, small_merge_px=0, **kw)
                n_on.append(len(on))
                n_off.append(len(off))
                all_on, all_off = all_on + covered(on, marks), all_off + covered(off, marks)
                # Regions come back largest first, so the production cap is a prefix.
                cap_on, cap_off = cap_on + covered(on[:cap], marks), cap_off + covered(off[:cap], marks)
                lone_kept += covered(on, lone)
            spur = sum(len(find_ink_regions(b, **kw)) for b in blanks)
            miss = extra = 0
            for chart in charts:
                hit, more = score(find_ink_regions(chart, **kw))
                miss, extra = miss + 5 - hit, extra + more
            ms = median_ms(probe, **kw) if name == "spec" else float("nan")
            print(
                f"{area:>4} {block:>3} {c:>2} | {all_on:>8}/{all_off:<2}/{n_marks:<7} | {cap_on:>8}/{cap_off:<2}/{n_marks:<7}"
                f" | {lone_kept:>2}/{n_lone:<2}| {max(n_on):>8}/{max(n_off):<8} | {spur:>5} | {miss:>2}/{extra:<2} | {ms:.1f}"
            )


def cap_table(seeds: range, profile: dict[str, int]) -> None:
    """Marks kept vs max_regions at one setting: merging is what keeps marks under the cap."""
    print(f"\n## max_regions at {profile}")
    print("cond   | " + " | ".join(f"cap {cap:>2} merge/off" for cap in (10, 15, 20, 30)))
    for name, cond in CONDITIONS.items():
        maths, _, _ = _gray_boards(seeds, cond)
        total = sum(len(m) for _, m, _ in maths)
        cells = []
        for cap in (10, 15, 20, 30):
            on = sum(covered(find_ink_regions(img, **{**profile, "max_regions": cap}), m) for img, m, _ in maths)
            off = sum(
                covered(find_ink_regions(img, small_merge_px=0, **{**profile, "max_regions": cap}), m)
                for img, m, _ in maths
            )
            cells.append(f"{on:>6}/{off:<2}/{total}")
        print(f"{name:<6} | " + " | ".join(f"{cell:<18}" for cell in cells))


def main() -> None:
    seeds = range(2 if "--quick" in sys.argv else 4)
    if "--flowchart" in sys.argv:
        flowchart_sweep(seeds)
        return
    math_sweep(seeds)
    cap_table(seeds, MATH_PROFILE)


if __name__ == "__main__":
    main()
