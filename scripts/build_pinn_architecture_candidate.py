"""Build the review-only PC-MP-PINN architecture diagram.

The script writes a vector SVG.  The image is deliberately kept separate from
the manuscript figure so it can be reviewed before replacing the paper asset.
"""
from __future__ import annotations

from html import escape
from pathlib import Path


W, H = 3200, 1800
OUT = Path("artifacts/architecture")
SVG_PATH = OUT / "pc_mp_pinn_architecture_candidate.svg"

INK = "#17324D"
MUTED = "#5F7285"
LINE = "#A8BAC9"
BLUE = "#2474C6"
BLUE_BG = "#EAF4FF"
GREEN = "#18845E"
GREEN_BG = "#E8F7F0"
ORANGE = "#C96A05"
ORANGE_BG = "#FFF2E2"
PURPLE = "#7A3FC1"
PURPLE_BG = "#F4EBFF"
RED = "#C93F5A"
RED_BG = "#FFF0F3"
TEAL = "#087E8B"
TEAL_BG = "#E8F7F8"


def rect(x, y, w, h, *, fill="#FFFFFF", stroke=LINE, sw=3, r=24, dash=None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash_attr}/>'
    )


def line(x1, y1, x2, y2, *, stroke=INK, sw=4, arrow=False, dash=None, opacity=1.0) -> str:
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
        f'stroke-width="{sw}" stroke-linecap="round" opacity="{opacity}"{marker}{dash_attr}/>'
    )


def path(points, *, stroke=INK, sw=4, arrow=False, dash=None, opacity=1.0) -> str:
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    d = " ".join(("M" if i == 0 else "L") + f" {x} {y}" for i, (x, y) in enumerate(points))
    return (
        f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}" '
        f'stroke-linejoin="round" stroke-linecap="round" opacity="{opacity}"{marker}{dash_attr}/>'
    )


def circle(cx, cy, r, *, fill="#FFFFFF", stroke=INK, sw=3) -> str:
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def txt(x, y, value, *, size=34, weight=400, fill=INK, anchor="start", family="Arial, Segoe UI, sans-serif", italic=False) -> str:
    style = "font-style:italic;" if italic else ""
    return (
        f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" style="{style}">'
        f'{escape(value)}</text>'
    )


def multiline(x, y, lines, *, size=31, leading=42, weight=400, fill=INK, anchor="start") -> str:
    spans = []
    for i, item in enumerate(lines):
        dy = 0 if i == 0 else leading
        spans.append(f'<tspan x="{x}" dy="{dy}">{escape(item)}</tspan>')
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, Segoe UI, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">' + "".join(spans) + "</text>"
    )


def pill(x, y, w, h, label, *, fill, stroke, size=27, weight=600, text_fill=None) -> str:
    if text_fill is None:
        text_fill = stroke
    return rect(x, y, w, h, fill=fill, stroke=stroke, sw=2, r=h / 2) + txt(
        x + w / 2, y + h * 0.66, label, size=size, weight=weight, fill=text_fill, anchor="middle"
    )


def stage(x, width, number, title, color) -> str:
    return (
        pill(x, 184, 62, 54, str(number), fill=color, stroke=color, size=28, weight=700, text_fill="#FFFFFF")
        + txt(x + 82, 222, title, size=31, weight=700)
        + line(x, 252, x + width, 252, stroke=color, sw=6)
    )


def neural_network(x0, y0, input_count, accent, out_label) -> str:
    """Draw a readable schematic of the fully connected layers."""
    chunks: list[str] = []
    xs = [x0, x0 + 120, x0 + 235, x0 + 350, x0 + 465, x0 + 595]
    hidden_count = 5
    input_ys = [y0 + i * 48 for i in range(input_count)]
    hidden_ys = [y0 - 22 + i * 48 for i in range(hidden_count)]
    output_ys = [y0 + 74]

    layers = [input_ys, hidden_ys, hidden_ys, hidden_ys, hidden_ys, output_ys]
    # Sparse representative connections keep the network legible at paper size.
    for li in range(len(xs) - 1):
        left, right = layers[li], layers[li + 1]
        for i, yl in enumerate(left):
            targets = right if len(right) <= 2 else [right[(i + j) % len(right)] for j in (0, 2)]
            for yr in targets:
                chunks.append(line(xs[li] + 15, yl, xs[li + 1] - 15, yr, stroke=accent, sw=2, opacity=0.34))
    for li, (xx, ys) in enumerate(zip(xs, layers)):
        for yy in ys:
            fill = accent if li == len(xs) - 1 else "#FFFFFF"
            chunks.append(circle(xx, yy, 15 if li < len(xs) - 1 else 19, fill=fill, stroke=accent, sw=3))
    chunks.append(txt(xs[-1], output_ys[0] + 53, out_label, size=27, weight=700, fill=accent, anchor="middle"))
    return "".join(chunks)


def build_svg() -> str:
    s: list[str] = []
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    s.append("""
    <defs>
      <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
        <feDropShadow dx="0" dy="8" stdDeviation="10" flood-color="#17324D" flood-opacity="0.10"/>
      </filter>
      <marker id="arrow" markerWidth="14" markerHeight="14" refX="10" refY="5" orient="auto" markerUnits="strokeWidth">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#17324D"/>
      </marker>
      <linearGradient id="headerGrad" x1="0" x2="1">
        <stop offset="0" stop-color="#EAF4FF"/><stop offset="0.52" stop-color="#E8F7F0"/><stop offset="1" stop-color="#F4EBFF"/>
      </linearGradient>
    </defs>
    """)
    s.append(rect(0, 0, W, H, fill="#F7FAFC", stroke="#F7FAFC", sw=0, r=0))
    s.append(f'<rect x="0" y="0" width="{W}" height="154" fill="url(#headerGrad)"/>')
    s.append(txt(72, 76, "Physics-Consistent Multi-Physics PINN Architecture", size=58, weight=800))
    s.append(txt(74, 124, "Sparse-label EV battery surrogate: concentration + lumped temperature + analytically coupled particle stress", size=30, weight=500, fill=MUTED))

    # Stage labels
    s.append(stage(80, 335, 1, "Normalized inputs", BLUE))
    s.append(stage(500, 820, 2, "Two field networks", GREEN))
    s.append(stage(1390, 740, 3, "Physics + coupling", PURPLE))
    s.append(stage(2200, 400, 4, "Joint training", RED))
    s.append(stage(2680, 440, 5, "Outputs + evidence", TEAL))

    # Inputs
    s.append(f'<g filter="url(#shadow)">{rect(80, 360, 335, 900, fill="#FFFFFF", stroke=BLUE, sw=4, r=30)}</g>')
    s.append(txt(247, 424, "Operating point", size=35, weight=750, fill=BLUE, anchor="middle"))
    s.append(txt(247, 465, "and coordinates", size=30, weight=600, fill=MUTED, anchor="middle"))
    s.append(pill(112, 510, 270, 70, "ρ = r / Rₚ", fill=BLUE_BG, stroke=BLUE, size=31))
    s.append(pill(112, 605, 270, 70, "τ = t / 7200 s", fill=BLUE_BG, stroke=BLUE, size=29))
    s.append(pill(112, 700, 270, 70, "C* = normalized C-rate", fill=BLUE_BG, stroke=BLUE, size=25))
    s.append(pill(112, 795, 270, 70, "T*ₐₘᵦ = normalized Tₐₘᵦ", fill=BLUE_BG, stroke=BLUE, size=23))
    s.append(line(112, 910, 382, 910, stroke=LINE, sw=2))
    s.append(txt(247, 962, "Training samples", size=31, weight=750, fill=INK, anchor="middle"))
    s.append(multiline(247, 1012, ["Sparse SPMe labels: c*, T*", "Interior collocation points", "Center + surface points"], size=25, leading=39, fill=MUTED, anchor="middle"))
    s.append(pill(112, 1162, 270, 60, "PyBaMM · SPMe · Chen2020", fill="#F1F5F9", stroke=MUTED, size=22, weight=650))

    # Network cards
    s.append(f'<g filter="url(#shadow)">{rect(500, 300, 820, 500, fill="#FFFFFF", stroke=GREEN, sw=4, r=30)}</g>')
    s.append(pill(530, 326, 360, 58, "CONCENTRATION BRANCH", fill=GREEN_BG, stroke=GREEN, size=25, weight=750))
    s.append(txt(1282, 366, "spatial field", size=24, weight=650, fill=MUTED, anchor="end"))
    s.append(neural_network(555, 460, 4, GREEN, "Nᶜ"))
    s.append(txt(848, 430, "4 hidden layers × 96 neurons · tanh", size=27, weight=650, fill=GREEN, anchor="middle"))
    s.append(txt(555, 675, "4 inputs", size=24, weight=650, fill=MUTED, anchor="middle"))
    s.append(rect(535, 704, 750, 68, fill=GREEN_BG, stroke=GREEN, sw=2, r=16))
    s.append(txt(910, 749, "ĉₛ(ρ,τ) = cₛ,0* + τ Nᶜ(ρ,τ,C*,T*ₐₘᵦ)   ·   hard initial condition", size=26, weight=650, fill=GREEN, anchor="middle"))

    s.append(f'<g filter="url(#shadow)">{rect(500, 870, 820, 500, fill="#FFFFFF", stroke=ORANGE, sw=4, r=30)}</g>')
    s.append(pill(530, 896, 340, 58, "TEMPERATURE BRANCH", fill=ORANGE_BG, stroke=ORANGE, size=25, weight=750))
    s.append(txt(1282, 936, "lumped state", size=24, weight=650, fill=MUTED, anchor="end"))
    s.append(neural_network(555, 1030, 3, ORANGE, "Nᵀ"))
    s.append(txt(848, 1000, "4 hidden layers × 96 neurons · tanh", size=27, weight=650, fill=ORANGE, anchor="middle"))
    s.append(txt(555, 1245, "3 inputs (no radius)", size=24, weight=650, fill=MUTED, anchor="middle"))
    s.append(rect(535, 1274, 750, 68, fill=ORANGE_BG, stroke=ORANGE, sw=2, r=16))
    s.append(txt(910, 1319, "T̂*(τ) = T*ₐₘᵦ + τ Nᵀ(τ,C*,T*ₐₘᵦ)   ·   hard initial condition", size=26, weight=650, fill=ORANGE, anchor="middle"))

    # Input arrows
    s.append(path([(415, 640), (455, 640), (455, 550), (500, 550)], stroke=INK, sw=5, arrow=True))
    s.append(path([(415, 760), (455, 760), (455, 1120), (500, 1120)], stroke=INK, sw=5, arrow=True))

    # Physics cards
    s.append(f'<g filter="url(#shadow)">{rect(1390, 300, 740, 320, fill="#FFFFFF", stroke=GREEN, sw=4, r=30)}</g>')
    s.append(pill(1420, 326, 300, 58, "SOLID DIFFUSION", fill=GREEN_BG, stroke=GREEN, size=25, weight=750))
    s.append(txt(1428, 445, "Rᶜ = ∂ĉₛ/∂τ − κ(∂²ĉₛ/∂ρ² + 2/ρ · ∂ĉₛ/∂ρ)", size=30, weight=650, fill=INK))
    s.append(pill(1428, 492, 292, 62, "center: ∂ρĉₛ|₀ = 0", fill="#F4FBF7", stroke=GREEN, size=23))
    s.append(pill(1740, 492, 352, 62, "surface: ∂ρĉₛ|₁ = gₛᵤᵣf", fill="#F4FBF7", stroke=GREEN, size=23))
    s.append(txt(1428, 594, "Automatic differentiation evaluates PDE and boundary residuals.", size=23, weight=500, fill=MUTED))

    s.append(f'<g filter="url(#shadow)">{rect(1390, 650, 740, 370, fill=PURPLE_BG, stroke=PURPLE, sw=5, r=30)}</g>')
    s.append(pill(1420, 678, 260, 58, "PROJECT CORE", fill=PURPLE, stroke=PURPLE, size=25, weight=800, text_fill="#FFFFFF"))
    s.append(txt(1425, 797, "Analytical concentration → stress operator", size=33, weight=800, fill=PURPLE))
    s.append(txt(1425, 855, "A(ρ,τ) = 3/ρ³ ∫₀ρ ĉₛ(s,τ)s² ds", size=31, weight=650, fill=INK))
    s.append(txt(1425, 911, "σ̂ᵣ, σ̂θ = S[ĉₛ ; E, ν, Ω]", size=33, weight=700, fill=INK))
    s.append(pill(1425, 944, 455, 54, "NO TRAINABLE STRESS HEAD", fill="#FFFFFF", stroke=PURPLE, size=23, weight=800))
    s.append(txt(1902, 979, "consistency by construction", size=22, weight=650, fill=PURPLE))

    s.append(f'<g filter="url(#shadow)">{rect(1390, 1050, 740, 320, fill="#FFFFFF", stroke=ORANGE, sw=4, r=30)}</g>')
    s.append(pill(1420, 1076, 330, 58, "LUMPED THERMAL", fill=ORANGE_BG, stroke=ORANGE, size=25, weight=750))
    s.append(txt(1428, 1192, "Rᵀ = ∂T̂*/∂τ − q* + α(T̂* − T*ₐₘᵦ)", size=31, weight=650, fill=INK))
    s.append(multiline(1428, 1251, ["One temperature per cell state", "Heat forcing + convective cooling"], size=24, leading=39, fill=MUTED))

    # Network-to-physics arrows
    s.append(path([(1320, 520), (1355, 520), (1355, 455), (1390, 455)], stroke=INK, sw=5, arrow=True))
    s.append(path([(1320, 650), (1350, 650), (1350, 835), (1390, 835)], stroke=PURPLE, sw=6, arrow=True))
    s.append(path([(1320, 1120), (1355, 1120), (1355, 1200), (1390, 1200)], stroke=INK, sw=5, arrow=True))

    # Loss panel
    s.append(f'<g filter="url(#shadow)">{rect(2200, 300, 400, 1070, fill="#FFFFFF", stroke=RED, sw=4, r=30)}</g>')
    s.append(pill(2230, 326, 340, 58, "WEIGHTED OBJECTIVE", fill=RED_BG, stroke=RED, size=25, weight=750))
    s.append(rect(2230, 430, 340, 190, fill="#F7FAFC", stroke=LINE, sw=2, r=20))
    s.append(txt(2400, 479, "Supervised data", size=28, weight=750, fill=INK, anchor="middle"))
    s.append(multiline(2400, 532, ["Ldata = MSE(c*)", "+ MSE(T*)", "sparse SPMe labels"], size=25, leading=36, fill=MUTED, anchor="middle"))
    s.append(rect(2230, 660, 340, 274, fill=RED_BG, stroke=RED, sw=2, r=20))
    s.append(txt(2400, 710, "Physics penalty", size=28, weight=750, fill=RED, anchor="middle"))
    s.append(multiline(2400, 762, ["0.05 Ldiffusion", "+ 0.05 Lthermal", "+ 0.10 Lcenter", "+ 0.10 Lsurface"], size=25, leading=40, fill=INK, anchor="middle"))
    s.append(rect(2230, 978, 340, 108, fill=RED, stroke=RED, sw=2, r=20))
    s.append(txt(2400, 1022, "TOTAL LOSS", size=24, weight=800, fill="#FFFFFF", anchor="middle"))
    s.append(txt(2400, 1063, "L = Ldata + Lphysics", size=28, weight=750, fill="#FFFFFF", anchor="middle"))
    s.append(pill(2230, 1132, 340, 58, "Adam · lr = 10⁻³", fill="#FFFFFF", stroke=RED, size=25, weight=750))
    s.append(pill(2230, 1212, 340, 58, "≤2500 epochs · patience 350", fill="#FFFFFF", stroke=RED, size=21, weight=650))
    s.append(txt(2400, 1320, "Gradient clipping: 10", size=23, weight=600, fill=MUTED, anchor="middle"))

    # Physics-to-loss arrows
    s.append(path([(2130, 470), (2162, 470), (2162, 760), (2200, 760)], stroke=INK, sw=5, arrow=True))
    s.append(path([(2130, 1210), (2162, 1210), (2162, 860), (2200, 860)], stroke=INK, sw=5, arrow=True))
    # predicted c and T to data loss, shown as a shared light path
    s.append(path([(1320, 760), (1360, 760), (1360, 1460), (2175, 1460), (2175, 540), (2200, 540)], stroke=RED, sw=4, arrow=True, dash="12 12", opacity=0.72))
    s.append(txt(1780, 1500, "predictions + labels", size=24, weight=650, fill=RED, anchor="middle"))

    # Output/evidence panel
    s.append(f'<g filter="url(#shadow)">{rect(2680, 300, 440, 1070, fill="#FFFFFF", stroke=TEAL, sw=4, r=30)}</g>')
    s.append(pill(2710, 326, 380, 58, "PREDICTED BATTERY STATES", fill=TEAL_BG, stroke=TEAL, size=23, weight=750))
    s.append(rect(2710, 425, 380, 92, fill=GREEN_BG, stroke=GREEN, sw=2, r=18))
    s.append(txt(2740, 465, "ĉₛ(ρ,τ)", size=30, weight=800, fill=GREEN))
    s.append(txt(2870, 465, "radial concentration profile", size=21, weight=600, fill=MUTED))
    s.append(rect(2710, 538, 380, 92, fill=ORANGE_BG, stroke=ORANGE, sw=2, r=18))
    s.append(txt(2740, 578, "T̂(τ)", size=30, weight=800, fill=ORANGE))
    s.append(txt(2845, 578, "lumped cell temperature", size=21, weight=600, fill=MUTED))
    s.append(rect(2710, 651, 380, 92, fill=PURPLE_BG, stroke=PURPLE, sw=2, r=18))
    s.append(txt(2740, 691, "σ̂ᵣ, σ̂θ", size=30, weight=800, fill=PURPLE))
    s.append(txt(2865, 691, "derived particle stress", size=21, weight=600, fill=MUTED))
    s.append(txt(2710, 807, "Nine paper metrics", size=31, weight=800, fill=INK))
    s.append(line(2710, 830, 3090, 830, stroke=TEAL, sw=4))
    s.append(txt(2722, 883, "ACCURACY (6)", size=23, weight=800, fill=TEAL))
    s.append(multiline(2722, 926, ["Rel-L₂(c); surface-c MAE", "ΔT RMSE; max ΔT error", "σθ RMSE; peak-stress error"], size=22, leading=36, fill=INK))
    s.append(txt(2722, 1065, "PHYSICS (2)", size=23, weight=800, fill=TEAL))
    s.append(multiline(2722, 1108, ["Mass-balance RMSE", "c–stress consistency RMSE"], size=22, leading=36, fill=INK))
    s.append(txt(2722, 1208, "EFFICIENCY (1)", size=23, weight=800, fill=TEAL))
    s.append(txt(2722, 1251, "Median speedup vs. PyBaMM SPMe", size=22, weight=500, fill=INK))
    s.append(pill(2710, 1292, 380, 52, "held-out conditions · 3 seeds", fill="#FFFFFF", stroke=TEAL, size=21, weight=700))

    # One clean trained-model path avoids crossing the objective panel.  The
    # three colored state cards at the destination identify the returned fields.
    s.append(path([(2600, 1032), (2645, 1032), (2645, 835), (2680, 835)], stroke=TEAL, sw=6, arrow=True))
    s.append(txt(2622, 1000, "trained model", size=21, weight=700, fill=TEAL, anchor="middle"))

    # Back-propagation loop, split explicitly to both trainable subnetworks.
    s.append(path([(2400, 1086), (2400, 1550), (460, 1550)], stroke=RED, sw=5, dash="16 12"))
    s.append(path([(460, 1550), (910, 1550), (910, 1370)], stroke=RED, sw=5, arrow=True, dash="16 12"))
    s.append(path([(460, 1550), (460, 280), (910, 280), (910, 300)], stroke=RED, sw=5, arrow=True, dash="16 12"))
    s.append(txt(1660, 1593, "backpropagation updates both subnetworks; the stress operator has no trainable parameters", size=26, weight=700, fill=RED, anchor="middle"))

    # Scope strip
    s.append(rect(80, 1640, 3040, 100, fill="#EDF2F7", stroke=LINE, sw=2, r=20))
    s.append(txt(110, 1682, "DEFENSIBLE SCOPE", size=23, weight=850, fill=INK))
    s.append(txt(110, 1720, "Constant-current discharge surrogate · PyBaMM SPMe reference · linear-elastic stress indicator · no degradation, SOH, RUL, or experimental-validation claim", size=25, weight=600, fill=MUTED))

    s.append("</svg>")
    return "".join(s)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SVG_PATH.write_text(build_svg(), encoding="utf-8")
    print(SVG_PATH.resolve())


if __name__ == "__main__":
    main()
