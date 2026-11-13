"""Rasterize the project architecture SVG without external dependencies.

This tiny renderer supports the SVG primitives emitted by
``build_pinn_architecture_candidate.py`` and creates the PNG used for review.
"""
from __future__ import annotations

import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SVG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/architecture/pc_mp_pinn_architecture_candidate.svg")
PNG = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("artifacts/architecture/pc_mp_pinn_architecture_candidate.png")
FONT_ROOT = Path(r"C:\Windows\Fonts")
NS = "{http://www.w3.org/2000/svg}"


def number(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    match = re.match(r"[-+0-9.]+", value)
    return float(match.group(0)) if match else default


def font(size: int, weight: int) -> ImageFont.FreeTypeFont:
    candidates = (
        ["segoeuib.ttf", "arialbd.ttf"]
        if weight >= 700
        else ["seguisb.ttf", "segoeui.ttf", "arial.ttf"]
    )
    for name in candidates:
        path = FONT_ROOT / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def arrow(draw: ImageDraw.ImageDraw, p1: tuple[float, float], p2: tuple[float, float], width: int, color: str) -> None:
    angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    length = max(18, width * 4.0)
    half = max(9, width * 2.0)
    base_x = p2[0] - length * math.cos(angle)
    base_y = p2[1] - length * math.sin(angle)
    left = (base_x + half * math.sin(angle), base_y - half * math.cos(angle))
    right = (base_x - half * math.sin(angle), base_y + half * math.cos(angle))
    draw.polygon([p2, left, right], fill=color)


def dash_line(draw: ImageDraw.ImageDraw, points, fill, width, dash: tuple[int, int] | None) -> None:
    if not dash:
        draw.line(points, fill=fill, width=width, joint="curve")
        return
    on, off = dash
    for a, b in zip(points, points[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        dist = math.hypot(dx, dy)
        if dist == 0:
            continue
        ux, uy = dx / dist, dy / dist
        pos = 0.0
        while pos < dist:
            end = min(dist, pos + on)
            draw.line(
                [(a[0] + ux * pos, a[1] + uy * pos), (a[0] + ux * end, a[1] + uy * end)],
                fill=fill,
                width=width,
            )
            pos += on + off


def main() -> None:
    root = ET.fromstring(SVG.read_text(encoding="utf-8"))
    width, height = int(number(root.get("width"))), int(number(root.get("height")))
    image = Image.new("RGB", (width, height), "#F7FAFC")
    draw = ImageDraw.Draw(image)

    def render(element: ET.Element, in_defs: bool = False) -> None:
        tag = element.tag.removeprefix(NS)
        if tag == "defs":
            return
        if tag in {"svg", "g"}:
            for child in element:
                render(child, in_defs)
            return
        if tag == "rect":
            x, y = number(element.get("x")), number(element.get("y"))
            w, h = number(element.get("width")), number(element.get("height"))
            radius = number(element.get("rx"))
            fill = element.get("fill", "#FFFFFF")
            if fill.startswith("url("):
                fill = "#EAF4FF"
            stroke = element.get("stroke", "#000000")
            sw = int(number(element.get("stroke-width"), 1))
            draw.rounded_rectangle((x, y, x + w, y + h), radius=radius, fill=fill, outline=stroke, width=sw)
            return
        if tag == "circle":
            cx, cy, r = number(element.get("cx")), number(element.get("cy")), number(element.get("r"))
            draw.ellipse(
                (cx - r, cy - r, cx + r, cy + r),
                fill=element.get("fill", "#FFFFFF"),
                outline=element.get("stroke", "#000000"),
                width=int(number(element.get("stroke-width"), 1)),
            )
            return
        if tag == "line":
            points = [
                (number(element.get("x1")), number(element.get("y1"))),
                (number(element.get("x2")), number(element.get("y2"))),
            ]
            color = element.get("stroke", "#000000")
            sw = int(number(element.get("stroke-width"), 1))
            dash_attr = element.get("stroke-dasharray")
            dash = tuple(map(int, dash_attr.split())) if dash_attr else None
            dash_line(draw, points, color, sw, dash)
            if element.get("marker-end"):
                arrow(draw, points[-2], points[-1], sw, "#17324D")
            return
        if tag == "path":
            if element.get("fill") != "none":
                return
            values = re.findall(r"[ML]\s*([-+0-9.]+)\s+([-+0-9.]+)", element.get("d", ""))
            points = [(float(x), float(y)) for x, y in values]
            if len(points) < 2:
                return
            color = element.get("stroke", "#000000")
            sw = int(number(element.get("stroke-width"), 1))
            dash_attr = element.get("stroke-dasharray")
            dash = tuple(map(int, dash_attr.split())) if dash_attr else None
            dash_line(draw, points, color, sw, dash)
            if element.get("marker-end"):
                arrow(draw, points[-2], points[-1], sw, "#17324D")
            return
        if tag == "text":
            x, y = number(element.get("x")), number(element.get("y"))
            size = int(number(element.get("font-size"), 32))
            weight = int(number(element.get("font-weight"), 400))
            fill = element.get("fill", "#17324D")
            anchor = element.get("text-anchor", "start")
            pil_anchor = {"start": "ls", "middle": "ms", "end": "rs"}.get(anchor, "ls")
            face = font(size, weight)
            tspans = list(element)
            if not tspans:
                draw.text((x, y), element.text or "", font=face, fill=fill, anchor=pil_anchor)
                return
            current_y = y
            for child in tspans:
                current_y += number(child.get("dy"), 0)
                child_x = number(child.get("x"), x)
                draw.text((child_x, current_y), child.text or "", font=face, fill=fill, anchor=pil_anchor)

    render(root)
    PNG.parent.mkdir(parents=True, exist_ok=True)
    image.save(PNG, dpi=(300, 300), optimize=True)
    print(PNG.resolve())


if __name__ == "__main__":
    main()
