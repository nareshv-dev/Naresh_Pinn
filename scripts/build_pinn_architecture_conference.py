"""Build a clean IEEE full-width block architecture for the PC-MP-PINN."""
from __future__ import annotations

from html import escape
from pathlib import Path


W, H = 2400, 1100
OUT = Path("artifacts/architecture")
SVG_PATH = OUT / "pc_mp_pinn_architecture_conference.svg"

INK = "#17324D"
MUTED = "#5D7184"
LINE = "#A9BAC8"
BLUE, BLUE_BG = "#226FC1", "#EAF4FF"
GREEN, GREEN_BG = "#17835D", "#E9F7F0"
ORANGE, ORANGE_BG = "#C96704", "#FFF2E3"
PURPLE, PURPLE_BG = "#7440BC", "#F3EBFF"
RED, RED_BG = "#C83F59", "#FFF0F3"
TEAL, TEAL_BG = "#087C88", "#E8F7F8"
GOLD, GOLD_BG = "#B97700", "#FFF8DF"


def rect(x, y, w, h, fill="#FFFFFF", stroke=LINE, sw=3, radius=18, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash_attr}/>'


def text(x, y, value, size=32, weight=400, fill=INK, anchor="start"):
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, Segoe UI, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(value)}</text>'
    )


def line(x1, y1, x2, y2, stroke=INK, sw=4, arrow=False, dash=None):
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round"{marker}{dash_attr}/>'


def path(points, stroke=INK, sw=4, arrow=False, dash=None):
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    d = " ".join(("M" if i == 0 else "L") + f" {x} {y}" for i, (x, y) in enumerate(points))
    return f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"{marker}{dash_attr}/>'


def pill(x, y, w, h, label, fill, stroke, size=27, weight=700, text_fill=None):
    return rect(x, y, w, h, fill, stroke, 2, h / 2) + text(x + w / 2, y + h * 0.68, label, size, weight, text_fill or stroke, "middle")


def stage_label(x, y, number, label, color):
    return (
        pill(x, y, 56, 52, str(number), color, color, 32, 850, "#FFFFFF")
        + text(x + 76, y + 39, label, 34, 850, INK)
    )


def mlp_box(x, y, color, bg, title):
    items = [rect(x, y, 350, 126, bg, color, 3, 16)]
    items.append(text(x + 175, y + 36, title, 28, 800, color, "middle"))
    # Four layer bars communicate architecture without pretending to draw 96 nodes.
    heights = [30, 40, 50, 60]
    for i, bar_h in enumerate(heights):
        bx = x + 58 + i * 58
        by = y + 58 + (60 - bar_h) / 2
        items.append(rect(bx, by, 26, bar_h, "#FFFFFF", color, 2, 6))
    items.append(text(x + 300, y + 82, "×96", 28, 800, color, "middle"))
    items.append(text(x + 300, y + 110, "tanh", 23, 700, color, "middle"))
    return "".join(items)


def build():
    s = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<defs><marker id="arrow" markerWidth="13" markerHeight="13" refX="10" refY="5" orient="auto" markerUnits="strokeWidth"><path d="M 0 0 L 10 5 L 0 10 z" fill="#17324D"/></marker></defs>',
        rect(0, 0, W, H, "#FFFFFF", "#FFFFFF", 0, 0),
    ]

    # Stage 1: inputs and reference sampling
    s.append(rect(55, 55, 2290, 225, "#F8FBFE", BLUE, 3, 24, "10 8"))
    s.append(stage_label(78, 73, 1, "Inputs and reference sampling", BLUE))
    input_specs = [
        (90, "Radius", "ρ = r/Rₚ"),
        (370, "Time", "τ = t/7200 s"),
        (650, "Load", "C-rate"),
        (930, "Environment", "Ambient T∞"),
    ]
    for x, title, value in input_specs:
        s.append(rect(x, 140, 245, 105, BLUE_BG, BLUE, 2, 14))
        s.append(text(x + 122, 178, title, 23, 750, MUTED, "middle"))
        s.append(text(x + 122, 224, value, 30, 800, BLUE, "middle"))
    s.append(rect(1245, 140, 440, 105, GOLD_BG, GOLD, 2, 14))
    s.append(text(1465, 178, "Sparse reference labels", 24, 800, GOLD, "middle"))
    s.append(text(1465, 224, "PyBaMM SPMe:  cₛ , T", 28, 750, INK, "middle"))
    s.append(rect(1715, 140, 590, 105, PURPLE_BG, PURPLE, 2, 14))
    s.append(text(2010, 178, "Physics collocation points", 24, 800, PURPLE, "middle"))
    s.append(text(2010, 224, "interior  ·  center  ·  particle surface", 27, 700, INK, "middle"))

    # Main model container
    s.append(rect(55, 325, 2290, 555, "#F5FCF8", GREEN, 4, 26))
    s.append(stage_label(78, 343, 2, "Physics-Consistent Multi-Physics PINN", GREEN))
    s.append(pill(1880, 344, 425, 48, "TWO TRAINABLE FIELD NETWORKS", "#FFFFFF", GREEN, 21, 800))

    # Concentration row
    s.append(pill(90, 430, 220, 50, "CONCENTRATION", GREEN, GREEN, 22, 800, "#FFFFFF"))
    s.append(rect(90, 500, 245, 126, "#FFFFFF", BLUE, 3, 16))
    s.append(text(212, 548, "[ρ, τ, C, T∞]", 30, 800, BLUE, "middle"))
    s.append(text(212, 592, "4 inputs", 24, 700, MUTED, "middle"))
    s.append(mlp_box(390, 500, GREEN, GREEN_BG, "Fully connected MLP"))
    s.append(rect(795, 500, 320, 126, "#FFFFFF", GREEN, 3, 16))
    s.append(text(955, 542, "Hard initial condition", 25, 800, GREEN, "middle"))
    s.append(text(955, 589, "ĉₛ = cₛ,0* + τNᶜ", 29, 750, INK, "middle"))
    s.append(rect(1170, 500, 245, 126, GREEN_BG, GREEN, 3, 16))
    s.append(text(1292, 548, "ĉₛ(ρ,τ)", 34, 850, GREEN, "middle"))
    s.append(text(1292, 592, "concentration field", 23, 650, MUTED, "middle"))
    s.append(rect(1850, 500, 455, 126, "#FFFFFF", GREEN, 3, 16))
    s.append(text(2077, 540, "Diffusion residual", 27, 800, GREEN, "middle"))
    s.append(text(2077, 580, "Rᶜ + center/surface BCs", 25, 700, INK, "middle"))
    s.append(text(2077, 608, "parameters: Dₛ, Rₚ", 21, 650, MUTED, "middle"))

    # Temperature row
    s.append(pill(90, 665, 220, 50, "TEMPERATURE", ORANGE, ORANGE, 22, 800, "#FFFFFF"))
    s.append(rect(90, 730, 245, 126, "#FFFFFF", BLUE, 3, 16))
    s.append(text(212, 778, "[τ, C, T∞]", 30, 800, BLUE, "middle"))
    s.append(text(212, 822, "3 inputs · no radius", 22, 700, MUTED, "middle"))
    s.append(mlp_box(390, 730, ORANGE, ORANGE_BG, "Fully connected MLP"))
    s.append(rect(795, 730, 320, 126, "#FFFFFF", ORANGE, 3, 16))
    s.append(text(955, 772, "Hard initial condition", 25, 800, ORANGE, "middle"))
    s.append(text(955, 819, "T̂* = T∞* + τNᵀ", 29, 750, INK, "middle"))
    s.append(rect(1170, 730, 245, 126, ORANGE_BG, ORANGE, 3, 16))
    s.append(text(1292, 778, "T̂(τ)", 34, 850, ORANGE, "middle"))
    s.append(text(1292, 822, "lumped temperature", 23, 650, MUTED, "middle"))
    s.append(rect(1850, 730, 455, 126, "#FFFFFF", ORANGE, 3, 16))
    s.append(text(2077, 770, "Thermal residual", 27, 800, ORANGE, "middle"))
    s.append(text(2077, 810, "Rᵀ = ∂T̂*/∂τ − q* + αΔT*", 24, 700, INK, "middle"))
    s.append(text(2077, 838, "parameters: q*, α", 21, 650, MUTED, "middle"))

    # Project core: analytical mechanical coupling
    s.append(rect(1470, 455, 340, 270, PURPLE_BG, PURPLE, 5, 22))
    s.append(pill(1495, 477, 200, 46, "PROJECT CORE", PURPLE, PURPLE, 21, 850, "#FFFFFF"))
    s.append(text(1640, 575, "Analytical stress", 30, 850, PURPLE, "middle"))
    s.append(text(1640, 613, "from concentration", 27, 800, PURPLE, "middle"))
    s.append(text(1640, 663, "σ̂ᵣ, σ̂θ = S[ĉₛ; E,ν,Ω]", 25, 800, INK, "middle"))
    s.append(pill(1495, 684, 290, 30, "NO LEARNED STRESS HEAD", "#FFFFFF", PURPLE, 17, 850))

    # Forward arrows: all terminate before the next card border.
    row_y = 563
    for x1, x2 in [(335, 378), (740, 783), (1115, 1158)]:
        s.append(line(x1, row_y, x2, row_y, INK, 5, True))
    row_y = 793
    for x1, x2 in [(335, 378), (740, 783), (1115, 1158)]:
        s.append(line(x1, row_y, x2, row_y, INK, 5, True))
    # The predicted fields connect directly to their physical operators.
    s.append(line(1415, 563, 1458, 563, PURPLE, 5, True))
    s.append(path([(1415, 545), (1435, 545), (1435, 430), (2077, 430), (2077, 488)], GREEN, 5, True))
    s.append(line(1415, 793, 1838, 793, ORANGE, 5, True))

    # Stage 3: compact objective and returned states
    s.append(rect(55, 920, 2290, 135, "#F8FAFC", RED, 3, 22))
    s.append(stage_label(78, 939, 3, "Training objective", RED))
    s.append(text(445, 980, "L = Ldata(cₛ,T) + 0.05Ldiff + 0.05Lthermal + 0.10Lcenter + 0.10Lsurface", 29, 800, INK))
    s.append(text(445, 1024, "Outputs: concentration profile  ·  lumped temperature  ·  concentration-derived radial and tangential stress", 25, 700, MUTED))
    s.append(pill(1970, 950, 335, 62, "PHYSICS + SPARSE DATA", RED_BG, RED, 22, 850))

    # Residuals feed the objective; the diffusion path travels around the
    # thermal card, while mechanics remains a deterministic output map.
    s.append(path([(2270, 626), (2320, 626), (2320, 908)], GREEN, 4, True, "10 8"))
    s.append(path([(2077, 856), (2077, 908)], ORANGE, 4, True, "10 8"))

    s.append("</svg>")
    return "".join(s)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SVG_PATH.write_text(build(), encoding="utf-8")
    print(SVG_PATH.resolve())


if __name__ == "__main__":
    main()
