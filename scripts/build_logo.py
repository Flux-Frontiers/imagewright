#!/usr/bin/env python3
"""Draw the imagewright logo in Art Techno and write every size the repo uses.

The mark is the contract: prompt lines run into the die, and one gold line
carries the result out to a framed picture, a lighthouse whose lamp room is
the die. The palette, drawing helpers and outlined lettering come from the
Art Techno brand build, so the SVGs render the same everywhere.

Usage::

    python scripts/build_logo.py

Needs ``fonttools`` and Pillow, Google Chrome for the PNG renders, and a clone
of https://github.com/Flux-Frontiers/art_techno beside this repo. Set ``CHROME``
to the browser binary if it is not at the default macOS path.

Writes:

- ``assets/imagewright_logo.svg`` and ``assets/imagewright_logo.png``: the square
  mark (1024 px PNG)
- ``assets/imagewright_icon.svg``: the mark reduced for 64 px and below
- ``assets/imagewright_banner.svg`` and ``assets/imagewright_banner.png``
- ``docs/assets/logo.png`` and ``docs/assets/favicon.png``: the docs site's
  header logo and favicon, from the icon
"""

import importlib.util
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
AT_BUILD = ROOT.parent / "art_techno" / "docs" / "brand" / "build.py"

_spec = importlib.util.spec_from_file_location("at_build", AT_BUILD)
at = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(at)

f, line, pad, via, diamond, clipped_rect = (
    at.f,
    at.line,
    at.pad,
    at.via,
    at.diamond,
    at.clipped_rect,
)
OBSIDIAN, PANEL, COPPER, GOLD, GREEN, MINT, SILK = (
    at.OBSIDIAN,
    at.PANEL,
    at.COPPER,
    at.GOLD,
    at.GREEN,
    at.MINT,
    at.SILK,
)

NAME = "IMAGEWRIGHT"
PART = "PROMPT IN, PICTURE OUT"


def rect(
    x: float, y: float, w: float, h: float, fill: str, stroke: str = "", width: float = 0
) -> str:
    s = f' stroke="{stroke}" stroke-width="{f(width)}"' if stroke else ""
    return f'<rect x="{f(x)}" y="{f(y)}" width="{f(w)}" height="{f(h)}" fill="{fill}"{s}/>'


def die(x: float, y: float, s: float, core: str) -> str:
    """A square green die in a gold line, centered on ``x, y``.

    :param x: Center x.
    :param y: Center y.
    :param s: Side length.
    :param core: Color of the square at its center.
    """
    return rect(x - s / 2, y - s / 2, s, s, GREEN, GOLD, s / 9) + rect(
        x - s / 6, y - s / 6, s / 3, s / 3, core
    )


def node(x: float, y: float, r: float) -> str:
    """A prompt's start: a copper via."""
    return via(x, y, r, COPPER)


def poly(pts, fill: str, opacity: float = 1.0) -> str:
    """A closed, filled polygon.

    :param pts: Corner points.
    :param fill: Fill color.
    :param opacity: Fill opacity.
    """
    d = "L".join(f"{f(x)} {f(y)}" for x, y in pts)
    op = f' fill-opacity="{opacity}"' if opacity < 1 else ""
    return f'<path d="M{d}Z" fill="{fill}"{op}/>'


def picture(x0: float, y0: float, w: float, h: float, width: float, small: bool = False) -> str:
    """The image: a lighthouse at night in a gold frame, top-left corner at ``x0, y0``.

    The lamp room is the fleet's die, so the picture is lit by the same part
    that made it.

    :param x0: Left edge.
    :param y0: Top edge.
    :param w: Width.
    :param h: Height.
    :param width: Frame stroke width.
    :param small: Draw the icon's reduced picture: one beam, no stripes, waves or star.
    """
    out = [
        f'<path d="{clipped_rect(x0, y0, w, h, w * 0.1)}" fill="{PANEL}" '
        f'stroke="{GOLD}" stroke-width="{f(width)}"/>'
    ]
    sea = y0 + 0.74 * h
    tx = x0 + 0.40 * w
    top = y0 + 0.36 * h
    lamp = (tx, top - 0.07 * w)
    # beams from the lamp, a faint wedge each way
    out.append(
        poly(
            [lamp, (x0 + w - width, lamp[1] - 0.16 * w), (x0 + w - width, lamp[1] + 0.06 * w)],
            GOLD,
            opacity=0.35,
        )
    )
    if not small:
        out.append(
            poly(
                [lamp, (x0 + width, lamp[1] - 0.10 * w), (x0 + width, lamp[1] + 0.03 * w)],
                GOLD,
                opacity=0.2,
            )
        )
    # the tower, its stripes, the lamp room (a die) and the cap
    out.append(
        poly(
            [
                (tx - 0.12 * w, sea),
                (tx - 0.065 * w, top),
                (tx + 0.065 * w, top),
                (tx + 0.12 * w, sea),
            ],
            GOLD,
        )
    )
    if not small:
        for v in (0.48, 0.60):
            yy = y0 + v * h
            half = 0.065 * w + 0.055 * w * (yy - top) / (sea - top)
            out.append(line([(tx - half, yy), (tx + half, yy)], PANEL, 5))
    out.append(die(*lamp, 0.14 * w, SILK))
    cap = [(tx - 0.10 * w, top - 0.14 * w), (tx, top - 0.24 * w), (tx + 0.10 * w, top - 0.14 * w)]
    out.append(poly(cap, GOLD))
    # the sea, its waves and a star
    out.append(
        line([(x0 + width / 2, sea), (x0 + w - width / 2, sea)], MINT, max(width * 0.6, 1.5))
    )
    if not small:
        for u, v, run in ((0.18, 0.82, 0.16), (0.56, 0.86, 0.22), (0.30, 0.91, 0.12)):
            out.append(line([(x0 + u * w, y0 + v * h), (x0 + (u + run) * w, y0 + v * h)], MINT, 3))
        out.append(diamond(x0 + 0.78 * w, y0 + 0.16 * h, 0.04 * w, SILK))
    return "".join(out)


def scan() -> str:
    """Prompt lines in, the die, a picture out, on a 512 field."""
    out = []
    dx, dy = 222, 212
    # the prompt: three lines converging on the die
    for (x0, y), yin in (((70, 156), 198), ((110, 212), 212), ((86, 268), 226)):
        out.append(
            line([(x0, y), (dx - 70 - abs(y - yin), y), (dx - 70, yin), (dx - 28, yin)], COPPER, 4)
        )
        out.append(node(x0, y, 8))
    out.append(line([(dx + 28, dy), (300, dy)], GOLD, 6))
    out.append(die(dx, dy, 56, SILK))
    out.append(picture(300, 118, 150, 188, 6))
    return "".join(out)


def scan_small() -> str:
    """The mark reduced for 64 px and below: the die and a simpler picture."""
    out = [line([(20.5, 32), (28, 32)], GOLD, 3), die(14, 32, 13, SILK)]
    out.append(picture(28, 14, 28, 36, 2.5, small=True))
    return "".join(out)


def border(w: float, h: float, inset: float, c: float, heavy: float) -> str:
    """Double frame with 45-degree corners and a green diamond at each corner.

    :param w: Canvas width.
    :param h: Canvas height.
    :param inset: Distance of the heavy line from the canvas edge.
    :param c: Corner cut of the heavy line.
    :param heavy: Stroke width of the heavy line.
    """
    out = [
        f'<path d="{clipped_rect(inset, inset, w - 2 * inset, h - 2 * inset, c)}" fill="none" stroke="{GOLD}" stroke-width="{f(heavy)}"/>',
        f'<path d="{clipped_rect(inset * 1.9, inset * 1.9, w - 3.8 * inset, h - 3.8 * inset, c * 0.75)}" fill="none" stroke="{GOLD}" stroke-width="{f(heavy * 0.4)}" stroke-opacity=".6"/>',
    ]
    d = inset * 1.45 + c * 0.4375
    for cx, cy in ((d, d), (w - d, d), (d, h - d), (w - d, h - d)):
        out.append(diamond(cx, cy, inset * 0.5, GREEN))
    return "".join(out)


def svg(w: int, h: int, title: str, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
        f"<title>{title}</title>"
        f'<rect width="{w}" height="{h}" fill="{OBSIDIAN}"/>{body}</svg>\n'
    )


def mark(display, mono) -> str:
    """512 square mark: the motif, the name and the part line beneath."""
    body = border(512, 512, 10, 26, 5) + scan()
    body += display.path(NAME, 256, 430, 40, 0.2, "middle", GOLD)
    body += mono.path(PART, 256, 462, 13.5, 0.3, "middle", MINT)
    return svg(512, 512, "imagewright mark, Art Techno", body)


def icon() -> str:
    """64 icon: the reduced motif in one heavy frame, no lettering."""
    body = f'<path d="{clipped_rect(2.5, 2.5, 59, 59, 7.5)}" fill="none" stroke="{GOLD}" stroke-width="3"/>'
    return svg(64, 64, "imagewright icon, Art Techno", body + scan_small())


def banner(display, mono) -> str:
    """1280 x 320 banner: the motif left, the name and part line right.

    The name is set as large as the room right of the frame allows, and the
    pair is centered on the banner's height.
    """
    s = 0.78
    body = border(1280, 320, 10, 18, 3)
    body += (
        f'<g transform="translate({f(230 - 256 * s)} {f(160 - 212 * s)}) scale({s})">{scan()}</g>'
    )
    cx, room, gap = 820, 760, 34
    size = min(120, 120 * room / display.width(NAME, 120, 0.2))
    part = min(26, 26 * room / mono.width(PART, 26, 0.3))
    cap = display.cap * size / display.upm
    pcap = mono.cap * part / mono.upm
    base = (320 - (cap + gap + pcap)) / 2 + cap
    body += display.path(NAME, cx, base, size, 0.2, "middle", GOLD)
    body += mono.path(PART, cx, base + gap + pcap, part, 0.3, "middle", MINT)
    return svg(1280, 320, "imagewright banner, Art Techno", body)


def render(src: Path, out: Path, w: int, h: int) -> None:
    """Render the SVG ``src`` to ``out`` at ``w`` x ``h`` px with headless Chrome.

    :param src: SVG file; its ``width`` attribute sets the CSS width.
    :param out: PNG to write.
    :param w: Output width in px.
    :param h: Output height in px.
    """
    svg_w = int(src.read_text().split('width="', 1)[1].split('"', 1)[0])
    subprocess.run(
        [
            at.CHROME,
            "--headless",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--window-size={svg_w},{svg_w * h // w}",
            f"--force-device-scale-factor={w / svg_w}",
            f"--screenshot={out}",
            src.as_uri(),
        ],
        check=True,
        capture_output=True,
    )


def resized(master: Path, dest: Path, px: int) -> None:
    Image.open(master).convert("RGB").resize((px, px), Image.LANCZOS).save(dest, optimize=True)


def main() -> None:
    at.fetch_fonts()
    display = at.Face(at.CACHE / "JosefinSans.ttf", weight=600)
    mono = at.Face(at.CACHE / "IBMPlexMono-Medium.ttf")

    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    mark_svg = assets / "imagewright_logo.svg"
    icon_svg = assets / "imagewright_icon.svg"
    banner_svg = assets / "imagewright_banner.svg"
    mark_svg.write_text(mark(display, mono))
    icon_svg.write_text(icon())
    banner_svg.write_text(banner(display, mono))

    mark_png = mark_svg.with_suffix(".png")
    render(mark_svg, mark_png, 1024, 1024)
    render(banner_svg, banner_svg.with_suffix(".png"), 2560, 640)
    docs = ROOT / "docs" / "assets"
    docs.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        icon_png = Path(tmp) / "icon.png"
        render(icon_svg, icon_png, 1024, 1024)
        resized(icon_png, docs / "logo.png", 256)
        resized(icon_png, docs / "favicon.png", 32)


if __name__ == "__main__":
    main()
