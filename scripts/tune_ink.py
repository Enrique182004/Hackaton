"""Sweep ink-detection thresholds on synthetic noisy boards.

Usage: python scripts/tune_ink.py [--quick] [--flowchart]
Default: maths sweep (small marks kept, dust/glare spurious, flowchart regression, ms per call) and a
max_regions table at MATH_PROFILE. --flowchart: the original block/C sweep on flowchart boards only.
The scene generators are reused by tests/test_vision_loop.py.
"""

from __future__ import annotations

import itertools
import sys
import time
from dataclasses import dataclass
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
SPECKS = 60  # dust specks per board in the maths sweep
EQUATIONS = (
    ("3.5 - x^2 = 0.25", (120, 240)),
    ("-4x + 7 = 2/3", (120, 560)),
    ("y = -0.5x", (1300, 290)),  # inside the glare
)
MARKS = (".", "-", "^2")
LONE_Y = 960
# Recommended SAR_INK_* values for maths boards (see the PR's sweep); tests pin behaviour at these.
MATH_PROFILE = dict(min_area=150, block_size=31, c=15, max_regions=20, small_merge_px=24)
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


def dot(img: np.ndarray, cx: float, cy: float, diameter: float, ink: int = INK[0]) -> Box:
    r = diameter / 2
    cv2.circle(img, (round(cx * 16), round(cy * 16)), round(r * 16), (ink,) * 3, -1, cv2.LINE_AA, shift=4)
    return round(cx - r), round(cy - r), round(diameter), round(diameter)


def minus(img: np.ndarray, x: int, cy: int, length: int, thick: int, ink: int = INK[0]) -> Box:
    y = cy - thick // 2
    cv2.rectangle(img, (x, y), (x + length - 1, y + thick - 1), (ink,) * 3, -1)
    return x, y, length, thick


def glyph(img: np.ndarray, ch: str, x: int, base: int, scale: float, stroke: int, ink: int = INK[0]) -> Box:
    """putText one character; returns its tight ink box (getTextSize is the font's, not the glyph's)."""
    cv2.putText(img, ch, (x, base), FONT, scale, (ink,) * 3, stroke, cv2.LINE_AA)
    (w, h), below = cv2.getTextSize(ch, FONT, scale, stroke)
    pad = stroke
    mask = np.zeros((h + below + 2 * pad, w + 2 * pad), np.uint8)
    cv2.putText(mask, ch, (pad, h + pad), FONT, scale, 255, stroke, cv2.LINE_AA)
    ys, xs = np.nonzero(mask)
    x0, y0 = x + int(xs.min()) - pad, base + int(ys.min()) - h - pad
    return x0, y0, int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1


def dust(img: np.ndarray, rng: np.random.Generator, n: int) -> None:
    """Dust and eraser residue: 1.5-3.5 px specks, grey 80-170, anywhere on the board."""
    for _ in range(n):
        x, y = float(rng.uniform(10, W - 10)), float(rng.uniform(10, H - 10))
        dot(img, x, y, float(rng.uniform(1.5, 3.5)), int(rng.integers(80, 171)))


def write_math(
    img: np.ndarray, text: str, origin: tuple[int, int], rng: np.random.Generator, scale: float = 1.8, stroke: int = 5
) -> list[tuple[str, Box]]:
    """Marker-style equation; returns every drawn token ('^2' for the superscript) with its ink box.

    '.', '-' and '^2' sit 8-12 px from their neighbours, just past the 7x7 dilation, so each is its own
    component, and they are tapped on with lighter pressure (grey 30-110) than the digits.
    """
    x, base = origin
    cap = cv2.getTextSize("0", FONT, scale, stroke)[0][1] - stroke
    tokens: list[tuple[str, Box]] = []
    raise_next = False
    for ch in text:
        gap, ink = int(rng.integers(8, 13)), int(rng.integers(30, 111))
        if ch == "^":
            raise_next = True
        elif ch == " ":
            x += int(0.45 * cap)
        elif ch == ".":
            d = float(rng.uniform(5.0, 8.0))
            tokens.append((ch, dot(img, x + gap + d / 2, base - d / 2, d, ink)))
            x += gap + round(d) + int(rng.integers(8, 13))
        elif ch == "-":
            length, thick = int(rng.integers(16, 21)), int(rng.integers(3, 6))
            tokens.append((ch, minus(img, x + gap, base - cap // 2, length, thick, ink)))
            x += gap + length + int(rng.integers(8, 13))
        elif raise_next:
            box = glyph(img, ch, x + gap, base - int(0.55 * cap), scale * 0.42, max(stroke // 2, 2), ink)
            tokens.append(("^" + ch, box))
            x = box[0] + box[2] + int(rng.integers(8, 13))
            raise_next = False
        else:
            tokens.append((ch, glyph(img, ch, x, base, scale, stroke)))
            x += cv2.getTextSize(ch, FONT, scale, stroke)[0][0]
    return tokens


@dataclass(frozen=True)
class MathBoard:
    image: np.ndarray
    equations: list[list[tuple[str, Box]]]  # drawn tokens per equation, in writing order
    lone: list[Box]  # a dot, a minus and a superscript-sized 2 far from any other ink

    @property
    def marks(self) -> list[Box]:
        return [box for eq in self.equations for ch, box in eq if ch in MARKS]

    @property
    def glyphs(self) -> list[Box]:
        return [box for eq in self.equations for ch, box in eq if ch not in MARKS]


def blank_board(seed: int = 0, specks: int = 0, **kw: float) -> np.ndarray:
    img = np.full((H, W, 3), 235, np.uint8)
    dust(img, np.random.default_rng(2000 + seed), specks)
    return degrade(img, seed, **kw)


def flowchart_board(seed: int = 0, stroke: int = 6, note: bool = True, specks: int = 0, **kw: float) -> np.ndarray:
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
    dust(img, np.random.default_rng(2000 + seed), specks)
    return degrade(img, seed, **kw)


def math_board(seed: int = 0, specks: int = 0, **kw: float) -> MathBoard:
    rng = np.random.default_rng(1000 + seed)
    img = np.full((H, W, 3), 235, np.uint8)
    equations = [write_math(img, text, origin, rng) for text, origin in EQUATIONS]
    lone = [
        dot(img, 300, LONE_Y, float(rng.uniform(5.0, 8.0)), int(rng.integers(30, 111))),
        minus(img, 700, LONE_Y, 18, 4, int(rng.integers(30, 111))),
        glyph(img, "2", 1100, LONE_Y + 9, 1.8 * 0.42, 2, int(rng.integers(30, 111))),
    ]
    dust(img, np.random.default_rng(2000 + seed), specks)
    return MathBoard(degrade(img, seed, **kw), equations, lone)


def region_at(regions: list, box: Box) -> dict | None:
    """The first region containing the centre of `box`."""
    x, y, w, h = box
    mx, my = x + w / 2, y + h / 2
    return next((r for r in regions if r["x"] <= mx <= r["x"] + r["w"] and r["y"] <= my <= r["y"] + r["h"]), None)


def covered(regions: list, boxes: list[Box]) -> int:
    """Boxes whose centre lies inside some region: kept on their own or merged into a neighbour."""
    return sum(region_at(regions, box) is not None for box in boxes)


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


def best_ms(img: np.ndarray, runs: int = 7, **kw: int) -> float:
    """Best of `runs` calls: the floor is stable even when other processes share the CPU."""
    find_ink_regions(img, **kw)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        find_ink_regions(img, **kw)
        times.append((time.perf_counter() - t) * 1000)
    return min(times)


def flowchart_sweep(seeds: range) -> None:
    blocks, cs, areas = (15, 21, 31, 41, 51), (5, 8, 10, 12, 15, 20, 25), (150,)
    probe = flowchart_board(0, **CONDITIONS["spec"])
    print("block   C min_area | " + " | ".join(f"{n}: blank-spur boxes-miss chart-spur" for n in CONDITIONS) + " | ms")
    for block, c, area in itertools.product(blocks, cs, areas):
        cells = []
        for cond in CONDITIONS.values():
            b, m, s, n = evaluate(block, c, area, cond, seeds)
            cells.append(f"{b:>3}/{len(seeds)} {m:>4}/{5 * n} {s:>3}/{n}")
        print(f"{block:>5} {c:>3} {area:>8} | " + " | ".join(cells) + f" | {best_ms(probe, block_size=block, c=c):.1f}")


def _gray_boards(seeds: range, cond: dict[str, float]) -> tuple[list[MathBoard], list, list]:
    """(maths boards, blank boards, flowchart boards), grey, all with glare and dust."""
    cond = {**cond, "glare": GLARE_GAIN}
    gray = [
        MathBoard(cv2.cvtColor(b.image, cv2.COLOR_BGR2GRAY), b.equations, b.lone)
        for b in (math_board(s, specks=SPECKS, **cond) for s in seeds)
    ]
    blanks = [cv2.cvtColor(blank_board(s, specks=SPECKS, **cond), cv2.COLOR_BGR2GRAY) for s in seeds]
    charts = [
        cv2.cvtColor(flowchart_board(s, stroke=k, specks=SPECKS, **cond), cv2.COLOR_BGR2GRAY)
        for s in seeds
        for k in STROKES
    ]
    return gray, blanks, charts


def math_sweep(seeds: range, cap: int = 10) -> None:
    areas, blocks, cs = (30, 40, 50, 60, 75, 100, 150), (21, 31, 41), (10, 15, 20)
    probe = math_board(0, specks=SPECKS, **CONDITIONS["spec"], glare=GLARE_GAIN).image
    print("ms* = best of 7 calls on a 1080p BGR maths board (spec only)")
    for name, cond in CONDITIONS.items():
        maths, blanks, charts = _gray_boards(seeds, cond)
        n_marks, n_lone = sum(len(b.marks) for b in maths), sum(len(b.lone) for b in maths)
        print(f"\n## {name} {cond} + glare {GLARE_GAIN:g} + {SPECKS} dust specks per board")
        print(f"marks: '.', '-', '^2' inside equations ({n_marks}); lone: same marks far from ink ({n_lone});")
        print(f"merge = small_merge_px 24, off = 0; @cap = max_regions {cap}, @all = uncapped;")
        print(f"regions: most on one maths board; blank: regions on {len(blanks)} dusty blank boards; ", end="")
        print(f"chart: boxes missed / spurious over {len(charts)} flowcharts.")
        print(
            "area blk  C | marks@all merge/off | marks@cap merge/off | lone | regions merge/off | blank | chart | ms*"
        )
        for area, block, c in itertools.product(areas, blocks, cs):
            kw = dict(min_area=area, block_size=block, c=c, max_regions=1000)
            all_on = all_off = cap_on = cap_off = lone_kept = 0
            n_on, n_off = [], []
            for board in maths:
                on = find_ink_regions(board.image, **kw)
                off = find_ink_regions(board.image, **{**kw, "small_merge_px": 0})
                n_on.append(len(on))
                n_off.append(len(off))
                all_on, all_off = all_on + covered(on, board.marks), all_off + covered(off, board.marks)
                # Regions come back largest first, so the production cap is a prefix.
                cap_on += covered(on[:cap], board.marks)
                cap_off += covered(off[:cap], board.marks)
                lone_kept += covered(on, board.lone)
            spur = sum(len(find_ink_regions(b, **kw)) for b in blanks)
            miss = extra = 0
            for chart in charts:
                hit, more = score(find_ink_regions(chart, **kw))
                miss, extra = miss + 5 - hit, extra + more
            ms = f"{best_ms(probe, **kw):.1f}" if name == "spec" else "-"
            print(
                f"{area:>4} {block:>3} {c:>2} | {all_on:>8}/{all_off:<2}/{n_marks:<7} | {cap_on:>8}/{cap_off:<2}/{n_marks:<7}"
                f" | {lone_kept:>2}/{n_lone:<2}| {max(n_on):>8}/{max(n_off):<8} | {spur:>5} | {miss:>2}/{extra:<4} | {ms}"
            )


def cap_table(seeds: range, profile: dict[str, int]) -> None:
    """Marks and glyphs kept vs max_regions at one setting; merging is what keeps marks under the cap."""
    caps = (10, 15, 20, 30)
    print(f"\n## max_regions at {profile}: kept merge/off/total")
    print("cond   kind   | " + " | ".join(f"cap {cap:<12}" for cap in caps))
    for name, cond in CONDITIONS.items():
        maths, _, _ = _gray_boards(seeds, cond)
        for kind in ("marks", "glyphs"):
            total = sum(len(getattr(b, kind)) for b in maths)
            cells = []
            for cap in caps:
                on_kw = {**profile, "max_regions": cap}
                off_kw = {**on_kw, "small_merge_px": 0}
                on = sum(covered(find_ink_regions(b.image, **on_kw), getattr(b, kind)) for b in maths)
                off = sum(covered(find_ink_regions(b.image, **off_kw), getattr(b, kind)) for b in maths)
                cells.append(f"{on:>3}/{off:<3}/{total:<3}")
            print(f"{name:<6} {kind:<6} | " + " | ".join(f"{cell:<16}" for cell in cells))


def main() -> None:
    seeds = range(2 if "--quick" in sys.argv else 4)
    if "--flowchart" in sys.argv:
        flowchart_sweep(seeds)
        return
    math_sweep(seeds)
    cap_table(seeds, MATH_PROFILE)


if __name__ == "__main__":
    main()
