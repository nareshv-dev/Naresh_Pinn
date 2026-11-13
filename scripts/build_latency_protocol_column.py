"""Build a single-column IEEE figure for the inference latency protocol."""
from __future__ import annotations

from html import escape
from pathlib import Path


W, H = 1200, 1060
OUT = Path("artifacts/architecture")
SVG_PATH = OUT / "inference_efficiency_protocol_column.svg"

INK = "#17324D"
MUTED = "#5F7285"
BLUE, BLUE_BG = "#2474C6", "#EAF4FF"
GREEN, GREEN_BG = "#18845E", "#E8F7F0"
ORANGE, ORANGE_BG = "#C96A05", "#FFF2E2"
PURPLE, PURPLE_BG = "#7A3FC1", "#F4EBFF"


def rect(x, y, w, h, fill="#FFFFFF", stroke=INK, sw=4, radius=26):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def text(x, y, value, size=44, weight=400, fill=INK, anchor="start"):
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, Segoe UI, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(value)}</text>'
    )


def line(x1, y1, x2, y2, stroke=INK, sw=7, arrow=False):
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round"{marker}/>'


def path(points, stroke=INK, sw=7, arrow=False):
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    d = " ".join(("M" if i == 0 else "L") + f" {x} {y}" for i, (x, y) in enumerate(points))
    return f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"{marker}/>'


def build():
    s = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<defs><marker id="arrow" markerWidth="13" markerHeight="13" refX="10" refY="5" orient="auto" markerUnits="strokeWidth"><path d="M 0 0 L 10 5 L 0 10 z" fill="#17324D"/></marker></defs>',
        rect(0, 0, W, H, "#FFFFFF", "#FFFFFF", 0, 0),
    ]

    # Common held-out query batch
    s.append(rect(145, 30, 910, 185, BLUE_BG, BLUE, 5, 28))
    s.append(text(600, 86, "1  HELD-OUT QUERY BATCH", 39, 850, BLUE, "middle"))
    s.append(text(600, 145, "Same C-rate, ambient temperature,", 36, 650, INK, "middle"))
    s.append(text(600, 190, "time grid, batch size, and outputs", 36, 650, INK, "middle"))

    # Split to two timing paths
    s.append(line(600, 215, 600, 255, INK, 7))
    s.append(line(300, 255, 900, 255, INK, 7))
    s.append(line(300, 255, 300, 292, INK, 7, True))
    s.append(line(900, 255, 900, 292, INK, 7, True))

    s.append(rect(55, 315, 520, 255, GREEN_BG, GREEN, 5, 26))
    s.append(text(315, 370, "2A  PINN TIMING", 39, 850, GREEN, "middle"))
    s.append(text(315, 430, "100 warm-up calls", 36, 700, INK, "middle"))
    s.append(text(315, 476, "synchronized execution", 34, 650, INK, "middle"))
    s.append(text(315, 520, "median and P95 latency", 34, 650, INK, "middle"))
    s.append(text(315, 555, "median t(PINN)", 31, 800, GREEN, "middle"))

    s.append(rect(625, 315, 520, 255, ORANGE_BG, ORANGE, 5, 26))
    s.append(text(885, 370, "2B  SPMe TIMING", 39, 850, ORANGE, "middle"))
    s.append(text(885, 430, "identical hardware", 36, 700, INK, "middle"))
    s.append(text(885, 476, "identical held-out queries", 34, 650, INK, "middle"))
    s.append(text(885, 520, "solver wall-clock latency", 34, 650, INK, "middle"))
    s.append(text(885, 555, "median t(SPMe)", 31, 800, ORANGE, "middle"))

    # Merge into the reported metric
    s.append(path([(315, 570), (315, 625), (570, 625), (570, 657)], INK, 7, True))
    s.append(path([(885, 570), (885, 625), (630, 625), (630, 657)], INK, 7, True))
    s.append(rect(145, 680, 910, 285, PURPLE_BG, PURPLE, 5, 28))
    s.append(text(600, 735, "3  EFFICIENCY METRIC", 40, 850, PURPLE, "middle"))
    s.append(text(600, 810, "Speedup = median t(SPMe) / median t(PINN)", 39, 800, INK, "middle"))
    s.append(text(600, 870, "Report median, P95, and speedup", 37, 700, INK, "middle"))
    s.append(text(600, 923, "Training time is reported separately", 33, 650, MUTED, "middle"))
    s.append(rect(260, 990, 680, 48, "#FFFFFF", PURPLE, 3, 24))
    s.append(text(600, 1024, "same queries  ·  same hardware  ·  same precision", 29, 800, PURPLE, "middle"))
    s.append("</svg>")
    return "".join(s)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SVG_PATH.write_text(build(), encoding="utf-8")
    print(SVG_PATH.resolve())


if __name__ == "__main__":
    main()
