"""Apply the approved mixed figure placement to the IEEE manuscript."""
from __future__ import annotations

from pathlib import Path


FILES = [
    Path(r"C:\Users\naresh v\.codex\.chatgpt-projects\g-p-6aa8b506b49c819199fd4a101955204f\paper\battery_pinn_ieee_conference_table10_neat.tex"),
    Path(r"E:\Naresh_Projects\PINN_EV_battery\paper_preview\battery_pinn_ieee_conference_table10_neat.tex"),
]


def update(path: Path) -> None:
    text = path.read_text(encoding="utf-8-sig")
    if r"\usepackage{stfloats}" not in text:
        text = text.replace(
            r"\usepackage{graphicx}",
            "\\usepackage{graphicx}\n\\usepackage{stfloats} % enables bottom placement of full-width IEEE figures",
            1,
        )

    old_design = r"""\begin{figure*}[t]
\centering
\includegraphics[width=0.96\textwidth]{experimental_design.png}"""
    new_design = r"""\begin{figure*}[!b]
\centering
\includegraphics[width=0.96\textwidth]{experimental_design.png}"""
    if old_design not in text and new_design not in text:
        raise RuntimeError(f"Experimental-design block not found in {path}")
    text = text.replace(old_design, new_design, 1)

    old_latency = r"""\begin{figure*}[!t]
\centering
\includegraphics[width=0.70\textwidth]{inference_efficiency_protocol.png}
\caption{Latency protocol using identical held-out queries and hardware, with warm-up and synchronization before PINN--SPMe comparison.}
\label{fig:latency_protocol}
\end{figure*}"""
    new_latency = r"""\begin{figure}[!b]
\centering
\includegraphics[width=0.98\columnwidth]{inference_efficiency_protocol_column.png}
\caption{Latency protocol using identical held-out queries and hardware, with warm-up and synchronization before PINN--SPMe comparison.}
\label{fig:latency_protocol}
\end{figure}"""
    if old_latency not in text and new_latency not in text:
        raise RuntimeError(f"Latency block not found in {path}")
    text = text.replace(old_latency, new_latency, 1)

    path.write_text(text, encoding="utf-8")
    print(path)


for file_path in FILES:
    update(file_path)
