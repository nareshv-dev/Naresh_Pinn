"""Create the compact, paper-ready PC-MP-PINN architecture figure."""
from __future__ import annotations

from html import escape
from pathlib import Path


W, H = 2200, 950
OUT = Path("artifacts/architecture")
SVG_PATH = OUT / "pc_mp_pinn_architecture_paper.svg"

INK = "#17324D"
MUTED = "#5F7285"
LINE = "#AABBC8"
BLUE, BLUE_BG = "#2474C6", "#EAF4FF"
GREEN, GREEN_BG = "#18845E", "#E8F7F0"
ORANGE, ORANGE_BG = "#C96A05", "#FFF2E2"
PURPLE, PURPLE_BG = "#7A3FC1", "#F4EBFF"
RED, RED_BG = "#C93F5A", "#FFF0F3"
TEAL, TEAL_BG = "#087E8B", "#E8F7F8"


def rect(x, y, w, h, fill="#FFFFFF", stroke=LINE, sw=3, radius=20):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def text(x, y, value, size=32, weight=400, fill=INK, anchor="start"):
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, Segoe UI, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(value)}</text>'
    )


def line(x1, y1, x2, y2, stroke=INK, sw=4, arrow=False, dash=None, opacity=1.0):
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}" '
        f'stroke-linecap="round" opacity="{opacity}"{marker}{dash_attr}/>'
    )


def path(points, stroke=INK, sw=4, arrow=False, dash=None):
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    d = " ".join(("M" if i == 0 else "L") + f" {x} {y}" for i, (x, y) in enumerate(points))
    return f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round" stroke-linecap="round"{marker}{dash_attr}/>'


def circle(cx, cy, r, fill="#FFFFFF", stroke=INK, sw=3):
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def pill(x, y, w, h, label, fill, stroke, size=28, weight=700, text_fill=None):
    return rect(x, y, w, h, fill, stroke, 2, h / 2) + text(
        x + w / 2, y + h * 0.68, label, size, weight, text_fill or stroke, "middle"
    )


def stage(x, width, number, label, color):
    return (
        pill(x, 25, 48, 46, str(number), color, color, 25, 800, "#FFFFFF")
        + text(x + 64, 60, label, 30, 800)
        + line(x, 82, x + width, 82, color, 5)
    )


def network(x0, y0, input_count, accent, output_label):
    items: list[str] = []
    xs = [x0, x0 + 88, x0 + 176, x0 + 264, x0 + 352, x0 + 462]
    input_ys = [y0 + i * 38 for i in range(input_count)]
    # Four visible nodes represent each 96-neuron hidden layer.  Their vertical
    # range deliberately stops above the hard-constraint formula band.
    hidden_ys = [y0 - 10 + i * 38 for i in range(4)]
    out_ys = [y0 + 50]
    layers = [input_ys, hidden_ys, hidden_ys, hidden_ys, hidden_ys, out_ys]
    for li in range(len(xs) - 1):
        for i, left_y in enumerate(layers[li]):
            next_count = len(layers[li + 1])
            targets = layers[li + 1] if next_count == 1 else [layers[li + 1][i % next_count], layers[li + 1][(i + 2) % next_count]]
            for right_y in targets:
                items.append(line(xs[li] + 12, left_y, xs[li + 1] - 12, right_y, accent, 2, opacity=0.45))
    for li, (xx, ys) in enumerate(zip(xs, layers)):
        for yy in ys:
            items.append(circle(xx, yy, 12 if li < 5 else 17, accent if li == 5 else "#FFFFFF", accent, 3))
    items.append(text(xs[-1], out_ys[0] + 46, output_label, 27, 800, accent, "middle"))
    return "".join(items)


def build():
    s: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<defs><marker id="arrow" markerWidth="13" markerHeight="13" refX="10" refY="5" orient="auto" markerUnits="strokeWidth"><path d="M 0 0 L 10 5 L 0 10 z" fill="#17324D"/></marker></defs>',
        rect(0, 0, W, H, "#FFFFFF", "#FFFFFF", 0, 0),
    ]

    s += [
        stage(40, 270, 1, "Inputs", BLUE),
        stage(350, 600, 2, "Field networks", GREEN),
        stage(1000, 590, 3, "Physics coupling", PURPLE),
        stage(1640, 520, 4, "Predicted states", TEAL),
    ]

    # Input card
    s.append(rect(40, 115, 270, 635, "#FFFFFF", BLUE, 4, 26))
    s.append(text(175, 165, "Model inputs", 35, 800, BLUE, "middle"))
    for y, label in [(205, "ρ = r / Rₚ"), (282, "τ = t / 7200 s"), (359, "C-rate"), (436, "Ambient T∞")]:
        s.append(pill(68, y, 214, 58, label, BLUE_BG, BLUE, 29, 700))
    s.append(line(68, 527, 282, 527, LINE, 2))
    s.append(text(175, 573, "Reference points", 30, 800, INK, "middle"))
    s.append(text(175, 618, "SPMe labels: cₛ, T", 27, 650, MUTED, "middle"))
    s.append(text(175, 658, "interior + boundary", 27, 650, MUTED, "middle"))
    s.append(pill(68, 690, 214, 44, "PyBaMM · Chen2020", "#F1F5F9", MUTED, 21, 700))

    # Neural branches
    s.append(rect(350, 115, 600, 295, "#FFFFFF", GREEN, 4, 24))
    s.append(text(378, 160, "Concentration MLP", 34, 800, GREEN))
    s.append(pill(722, 128, 200, 46, "4 → 4×96 tanh → 1", GREEN_BG, GREEN, 21, 750))
    s.append(network(390, 210, 4, GREEN, "Nᶜ"))
    s.append(rect(378, 350, 544, 43, GREEN_BG, GREEN, 2, 12))
    s.append(text(650, 380, "ĉₛ = cₛ,0* + τNᶜ(ρ,τ,C,T∞)", 27, 700, GREEN, "middle"))

    s.append(rect(350, 455, 600, 295, "#FFFFFF", ORANGE, 4, 24))
    s.append(text(378, 500, "Temperature MLP", 34, 800, ORANGE))
    s.append(pill(722, 468, 200, 46, "3 → 4×96 tanh → 1", ORANGE_BG, ORANGE, 21, 750))
    s.append(network(390, 550, 3, ORANGE, "Nᵀ"))
    s.append(rect(378, 690, 544, 43, ORANGE_BG, ORANGE, 2, 12))
    s.append(text(650, 720, "T̂* = T∞* + τNᵀ(τ,C,T∞)", 27, 700, ORANGE, "middle"))

    # One clean input bus feeds both subnetworks.  Arrow tips stop before the
    # card border so they do not cover the outline.
    s.append(line(310, 430, 320, 430, INK, 5))
    s.append(line(320, 260, 320, 600, INK, 5))
    s.append(line(320, 260, 338, 260, INK, 5, True))
    s.append(line(320, 600, 338, 600, INK, 5, True))

    # Physics cards
    s.append(rect(1000, 115, 590, 195, "#FFFFFF", GREEN, 4, 24))
    s.append(pill(1027, 136, 255, 48, "SOLID DIFFUSION", GREEN_BG, GREEN, 23, 800))
    s.append(text(1028, 226, "Rᶜ = ∂ĉₛ/∂τ − κ[∂²ĉₛ/∂ρ²", 29, 700, INK))
    s.append(text(1110, 261, "+ (2/ρ) ∂ĉₛ/∂ρ]", 29, 700, INK))
    s.append(text(1028, 295, "center symmetry  ·  surface-flux boundary", 24, 650, MUTED))

    s.append(rect(1000, 335, 590, 175, "#FFFFFF", ORANGE, 4, 24))
    s.append(pill(1027, 356, 270, 48, "LUMPED THERMAL", ORANGE_BG, ORANGE, 23, 800))
    s.append(text(1028, 452, "Rᵀ = ∂T̂*/∂τ − q* + α(T̂* − T∞*)", 29, 700, INK))
    s.append(text(1528, 488, "parameters: q*, α", 23, 650, MUTED, "end"))

    s.append(rect(1000, 535, 590, 215, PURPLE_BG, PURPLE, 5, 24))
    s.append(pill(1027, 556, 220, 48, "PROJECT CORE", PURPLE, PURPLE, 23, 800, "#FFFFFF"))
    s.append(text(1028, 651, "σ̂ᵣ, σ̂θ = S[ĉₛ ; E, ν, Ω]", 34, 800, PURPLE))
    s.append(text(1028, 700, "analytical stress from concentration", 27, 700, INK))
    s.append(pill(1028, 714, 390, 30, "NO INDEPENDENT STRESS HEAD", "#FFFFFF", PURPLE, 19, 800))
    s.append(text(1558, 734, "E, ν, Ω", 23, 750, PURPLE, "end"))

    # The concentration output splits to diffusion and analytical mechanics.
    s.append(line(950, 260, 972, 260, INK, 5))
    s.append(path([(972, 260), (972, 212), (988, 212)], INK, 5, True))
    s.append(path([(972, 260), (972, 640), (988, 640)], PURPLE, 5, True))
    s.append(path([(950, 600), (972, 600), (972, 422), (988, 422)], INK, 5, True))

    # Output card and aligned arrows
    s.append(rect(1640, 115, 520, 635, "#FFFFFF", TEAL, 4, 26))
    s.append(text(1900, 166, "Model outputs", 35, 800, TEAL, "middle"))
    outputs = [
        (205, GREEN_BG, GREEN, "ĉₛ(ρ,τ)", "concentration profile"),
        (370, ORANGE_BG, ORANGE, "T̂(τ)", "lumped temperature"),
        (555, PURPLE_BG, PURPLE, "σ̂ᵣ(ρ,τ), σ̂θ(ρ,τ)", "derived particle stress"),
    ]
    for y, bg, color, symbol, description in outputs:
        s.append(rect(1680, y, 440, 120, bg, color, 3, 20))
        s.append(text(1900, y + 48, symbol, 35, 800, color, "middle"))
        s.append(text(1900, y + 91, description, 27, 650, MUTED, "middle"))
    s.append(pill(1690, 695, 420, 38, "stress is concentration-consistent by construction", "#FFFFFF", PURPLE, 20, 750))

    s.append(path([(1590, 212), (1615, 212), (1615, 265), (1628, 265)], GREEN, 5, True))
    s.append(path([(1590, 422), (1615, 422), (1615, 430), (1628, 430)], ORANGE, 5, True))
    s.append(path([(1590, 640), (1615, 640), (1615, 615), (1628, 615)], PURPLE, 5, True))

    # Compact training objective: enough to identify a PINN without shrinking the architecture.
    s.append(rect(350, 790, 1810, 125, RED_BG, RED, 4, 22))
    s.append(pill(378, 810, 260, 50, "PINN OBJECTIVE", RED, RED, 24, 800, "#FFFFFF"))
    s.append(text(675, 855, "L = Ldata(cₛ,T) + 0.05Ldiff + 0.05Lthermal + 0.10Lcenter + 0.10Lsurface", 31, 750, INK))
    s.append(text(675, 892, "Automatic differentiation + labeled samples + physics collocation points", 25, 650, MUTED))

    s.append("</svg>")
    return "".join(s)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SVG_PATH.write_text(build(), encoding="utf-8")
    print(SVG_PATH.resolve())


if __name__ == "__main__":
    main()
