"""Draw the app icon (a notebook page with the assistant's face) as SVG and PNGs from one set of shapes.

    .venv/bin/python scripts/make_icons.py

Writes biobench_sar/static/icons/: icon.svg, icon-192.png, icon-512.png (rounded tile), icon-maskable-512.png
(full bleed, art inside the maskable safe zone) and apple-touch-icon.png (180 px, opaque: iOS fills
transparency with black). The PNGs are drawn with Pillow at 4x and scaled down, so no SVG renderer is needed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

ICON_DIR = Path(__file__).resolve().parents[1] / "biobench_sar" / "static" / "icons"
VIEW = 512
TILE_RADIUS = 112
SUPERSAMPLE = 4

NAVY = "#23497d"
PAPER = "#f8f4ea"
INK = "#1d2b45"
PENCIL = "#b4442e"
MARGIN = "#d98b7f"
RULE = "#b9c7d9"
RIM = "#4f76ab"

Point = tuple[float, float]


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float
    r: float
    fill: str


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    r: float
    fill: str | None = None
    stroke: str | None = None
    width: float = 0


@dataclass(frozen=True)
class Stroke:
    """A round-capped polyline; with `quads`, a chain of quadratic curves (control, end) from `points[0]`."""

    points: tuple[Point, ...]
    color: str
    width: float
    quads: tuple[tuple[Point, Point], ...] = ()


def _wave(x0: float, y: float, segments: int, step: float = 28, lift: float = 20) -> Stroke:
    quads = tuple(
        ((x0 + step * i + step / 2, y - lift if i % 2 == 0 else y + lift), (x0 + step * (i + 1), y))
        for i in range(segments)
    )
    return Stroke(((x0, y),), NAVY, 10, quads)


ART: tuple[Rect | Circle | Stroke, ...] = (
    Rect(120, 92, 272, 328, 30, PAPER),
    Stroke(((123, 340), (389, 340)), RULE, 6),
    Stroke(((123, 380), (389, 380)), RULE, 6),
    Stroke(((172, 95), (172, 417)), MARGIN, 6),
    _wave(200, 322, 5),
    _wave(200, 362, 3),
    Stroke(((282, 150), (282, 124)), INK, 12),
    Circle(282, 116, 12, fill=PENCIL),
    Circle(282, 212, 64, stroke=INK, width=16),
    Circle(260, 206, 10, fill=INK),
    Circle(304, 206, 10, fill=INK),
    Stroke(((258, 234),), PENCIL, 12, (((282, 256), (306, 234)),)),
)


def _fmt(value: float) -> str:
    return f"{value:g}"


def _svg_shape(shape: Rect | Circle | Stroke) -> str:
    if isinstance(shape, Rect):
        return (
            f'<rect x="{_fmt(shape.x)}" y="{_fmt(shape.y)}" width="{_fmt(shape.w)}" height="{_fmt(shape.h)}" '
            f'rx="{_fmt(shape.r)}" fill="{shape.fill}"/>'
        )
    if isinstance(shape, Circle):
        paint = (
            f'fill="{shape.fill}"'
            if shape.fill
            else f'fill="none" stroke="{shape.stroke}" stroke-width="{_fmt(shape.width)}"'
        )
        return f'<circle cx="{_fmt(shape.cx)}" cy="{_fmt(shape.cy)}" r="{_fmt(shape.r)}" {paint}/>'
    (x, y), *rest = shape.points
    d = f"M{_fmt(x)} {_fmt(y)}"
    d += "".join(f" L{_fmt(px)} {_fmt(py)}" for px, py in rest)
    d += "".join(f" Q{_fmt(c[0])} {_fmt(c[1])} {_fmt(e[0])} {_fmt(e[1])}" for c, e in shape.quads)
    return f'<path d="{d}" fill="none" stroke="{shape.color}" stroke-width="{_fmt(shape.width)}" stroke-linecap="round" stroke-linejoin="round"/>'


def render_svg() -> str:
    body = "\n  ".join(_svg_shape(s) for s in ART)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {VIEW} {VIEW}" role="img" aria-label="Whiteboard Assistant">\n'
        "  <title>Whiteboard Assistant</title>\n"
        f'  <rect width="{VIEW}" height="{VIEW}" rx="{TILE_RADIUS}" fill="{NAVY}"/>\n'
        f'  <rect x="5" y="5" width="{VIEW - 10}" height="{VIEW - 10}" rx="{TILE_RADIUS - 5}" fill="none" stroke="{RIM}" stroke-width="10"/>\n'
        f"  {body}\n"
        "</svg>\n"
    )


def _sample_quads(start: Point, quads: tuple[tuple[Point, Point], ...], steps: int = 24) -> list[Point]:
    points = [start]
    p0 = start
    for (cx, cy), (ex, ey) in quads:
        for i in range(1, steps + 1):
            t = i / steps
            u = 1 - t
            points.append((u * u * p0[0] + 2 * u * t * cx + t * t * ex, u * u * p0[1] + 2 * u * t * cy + t * t * ey))
        p0 = (ex, ey)
    return points


def render_png(size: int, *, full_bleed: bool, art_scale: float = 1.0) -> Image.Image:
    """The icon at size px. full_bleed: square opaque tile (maskable / iOS); else the rounded tile with its rim."""
    big = size * SUPERSAMPLE
    k = big / VIEW
    image = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    def at(x: float, y: float) -> Point:
        c = VIEW / 2
        return ((c + (x - c) * art_scale) * k, (c + (y - c) * art_scale) * k)

    def length(v: float) -> float:
        return v * art_scale * k

    if full_bleed:
        draw.rectangle((0, 0, big, big), fill=NAVY)
    else:
        draw.rounded_rectangle((0, 0, big - 1, big - 1), radius=TILE_RADIUS * k, fill=RIM)
        inset = 10 * k
        draw.rounded_rectangle(
            (inset, inset, big - 1 - inset, big - 1 - inset), radius=(TILE_RADIUS - 10) * k, fill=NAVY
        )
    for shape in ART:
        if isinstance(shape, Rect):
            x0, y0 = at(shape.x, shape.y)
            x1, y1 = at(shape.x + shape.w, shape.y + shape.h)
            draw.rounded_rectangle((x0, y0, x1, y1), radius=length(shape.r), fill=shape.fill)
        elif isinstance(shape, Circle):
            cx, cy = at(shape.cx, shape.cy)
            if shape.fill:
                r = length(shape.r)
                draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=shape.fill)
            else:
                r = length(shape.r + shape.width / 2)
                draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=shape.stroke, width=round(length(shape.width)))
        else:
            raw = _sample_quads(shape.points[0], shape.quads) if shape.quads else list(shape.points)
            points = [at(x, y) for x, y in raw]
            width = length(shape.width)
            draw.line(points, fill=shape.color, width=round(width), joint="curve")
            for x, y in (points[0], points[-1]):
                draw.ellipse((x - width / 2, y - width / 2, x + width / 2, y + width / 2), fill=shape.color)
    small = image.resize((size, size), Image.Resampling.LANCZOS)
    return small.convert("RGB") if full_bleed else small


OUTPUTS: tuple[tuple[str, int, bool, float], ...] = (
    ("icon-192.png", 192, False, 1.0),
    ("icon-512.png", 512, False, 1.0),
    # Maskable safe zone: a centred circle of 40% radius; at 0.78 the page's corners stay inside it.
    ("icon-maskable-512.png", 512, True, 0.78),
    ("apple-touch-icon.png", 180, True, 0.92),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ICON_DIR)
    out: Path = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    (out / "icon.svg").write_text(render_svg(), encoding="utf-8")
    for name, size, full_bleed, scale in OUTPUTS:
        render_png(size, full_bleed=full_bleed, art_scale=scale).save(out / name, optimize=True)
        print(out / name)


if __name__ == "__main__":
    main()
