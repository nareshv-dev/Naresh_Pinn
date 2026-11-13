"""
src/evaluation/figures.py

Figure generation for Stage I evaluation.

Figures produced:
  1. voltage_vs_ref        : predicted vs. reference voltage over time
  2. temperature_vs_ref    : predicted vs. reference temperature over time
  3. concentration_heatmap : c_s(r*, t*) predicted vs reference (2D heatmap)
  4. loss_curves           : training loss curves (all components)
  5. residual_histograms   : histogram of physics residual values
  6. summary_table         : comparison table MLP vs PINN

All figures saved to outputs/figures/ with descriptive filenames.
DPI and figure style are configurable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Style constants
STYLE = {
    "figsize_1col": (8, 5),
    "figsize_2col": (14, 5),
    "figsize_heatmap": (14, 5),
    "figsize_table": (12, 4),
    "dpi": 150,
    "color_ref": "#2C3E50",
    "color_mlp": "#E74C3C",
    "color_pinn": "#3498DB",
    "lw": 1.8,
    "alpha_fill": 0.12,
}


class FigureGenerator:
    """
    Generates and saves all Stage I evaluation figures.

    Parameters
    ----------
    figures_dir : Path — output directory
    config : dict — YAML config (for figure toggles)
    nondim — Nondimensionalizer instance (for inverse transforms)
    """

    def __init__(self, figures_dir: Path, config: dict, nondim) -> None:
        self.out_dir = figures_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = config
        self.nondim = nondim

    # ── Figure 1: Voltage vs Reference ────────────────────────────────────────

    def voltage_vs_reference(
        self,
        t_s: np.ndarray,         # (n_t,) SI time
        V_ref: np.ndarray,       # (n_t,) SI voltage [V]
        V_mlp: Optional[np.ndarray] = None,    # (n_t,) [V]
        V_pinn: Optional[np.ndarray] = None,   # (n_t,) [V]
        label: str = "validation",
        save: bool = True,
    ) -> Path:
        fig, ax = plt.subplots(figsize=STYLE["figsize_1col"], dpi=STYLE["dpi"])

        ax.plot(t_s, V_ref, label="PyBaMM Reference", color=STYLE["color_ref"],
                lw=STYLE["lw"], ls="--")
        if V_mlp is not None:
            ax.plot(t_s, V_mlp, label="EXP-002 MLP", color=STYLE["color_mlp"],
                    lw=STYLE["lw"])
        if V_pinn is not None:
            ax.plot(t_s, V_pinn, label="EXP-003 PINN", color=STYLE["color_pinn"],
                    lw=STYLE["lw"])

        ax.set_xlabel("Time [s]", fontsize=11)
        ax.set_ylabel("Terminal Voltage [V]", fontsize=11)
        ax.set_title(f"Voltage: Prediction vs. PyBaMM Reference ({label})", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.4)
        fig.tight_layout()

        path = self.out_dir / f"voltage_vs_ref_{label}.png"
        if save:
            fig.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight")
            logger.info("Saved: %s", path)
        plt.close(fig)
        return path

    # ── Figure 2: Temperature vs Reference ────────────────────────────────────

    def temperature_vs_reference(
        self,
        t_s: np.ndarray,
        T_ref: np.ndarray,       # [K]
        T_mlp: Optional[np.ndarray] = None,
        T_pinn: Optional[np.ndarray] = None,
        label: str = "validation",
        save: bool = True,
    ) -> Path:
        fig, ax = plt.subplots(figsize=STYLE["figsize_1col"], dpi=STYLE["dpi"])

        ax.plot(t_s, T_ref - 273.15, label="PyBaMM Reference",
                color=STYLE["color_ref"], lw=STYLE["lw"], ls="--")
        if T_mlp is not None:
            ax.plot(t_s, T_mlp - 273.15, label="EXP-002 MLP",
                    color=STYLE["color_mlp"], lw=STYLE["lw"])
        if T_pinn is not None:
            ax.plot(t_s, T_pinn - 273.15, label="EXP-003 PINN",
                    color=STYLE["color_pinn"], lw=STYLE["lw"])

        ax.set_xlabel("Time [s]", fontsize=11)
        ax.set_ylabel("Cell Temperature [°C]", fontsize=11)
        ax.set_title(f"Temperature: Prediction vs. PyBaMM Reference ({label})", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.4)
        fig.tight_layout()

        path = self.out_dir / f"temperature_vs_ref_{label}.png"
        if save:
            fig.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight")
            logger.info("Saved: %s", path)
        plt.close(fig)
        return path

    # ── Figure 3: Concentration Heatmap ───────────────────────────────────────

    def concentration_heatmap(
        self,
        t_star: np.ndarray,      # (n_t,)
        r_star: np.ndarray,      # (n_r,)
        c_ref: np.ndarray,       # (n_t, n_r) [mol/m³]
        c_pred: np.ndarray,      # (n_t, n_r) [mol/m³]
        model_label: str = "MLP",
        label: str = "validation",
        save: bool = True,
    ) -> Path:
        # Relative error
        c_err = np.abs(c_pred - c_ref) / (np.abs(c_ref) + 1e-8)

        fig, axes = plt.subplots(1, 3, figsize=STYLE["figsize_heatmap"], dpi=STYLE["dpi"])

        for ax, data, title, cmap in zip(
            axes,
            [c_ref, c_pred, c_err * 100],
            ["PyBaMM Reference c_s [mol/m³]", f"{model_label} c_s [mol/m³]", "Rel. Error [%]"],
            ["viridis", "viridis", "Reds"],
        ):
            im = ax.pcolormesh(t_star, r_star, data.T, cmap=cmap, shading="auto")
            ax.set_xlabel("t* [dimensionless]", fontsize=10)
            ax.set_ylabel("r* [dimensionless]", fontsize=10)
            ax.set_title(title, fontsize=10)
            fig.colorbar(im, ax=ax)

        fig.suptitle(
            f"Negative Particle Concentration (SPMe x-averaged, {label})", fontsize=11
        )
        fig.tight_layout()

        path = self.out_dir / f"concentration_heatmap_{model_label.lower()}_{label}.png"
        if save:
            fig.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight")
            logger.info("Saved: %s", path)
        plt.close(fig)
        return path

    # ── Figure 4: Loss Curves ─────────────────────────────────────────────────

    def loss_curves(
        self,
        history: List[Dict],
        experiment_id: str = "EXP",
        save: bool = True,
    ) -> Path:
        epochs = [r["epoch"] for r in history]

        is_pinn = "train_diffusion" in history[0]

        if is_pinn:
            fig, axes = plt.subplots(2, 3, figsize=(16, 8), dpi=STYLE["dpi"])
            axes = axes.flatten()
            components = [
                ("train_total", "Total Loss"),
                ("train_data", "Data Loss"),
                ("train_diffusion", "Diffusion Residual"),
                ("train_thermal", "Thermal Residual"),
                ("train_ic", "IC Loss"),
                ("train_bc", "BC Loss"),
            ]
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=STYLE["dpi"])
            axes = axes.flatten()
            components = [
                ("train_loss", "Train Loss"),
            ]

        # Main loss
        for i, (key, title) in enumerate(components):
            vals = [r.get(key, float("nan")) for r in history]
            axes[i].semilogy(epochs, vals, color="#2E86AB", lw=1.5, label=key)

            val_key = "val_loss" if not is_pinn else "val_loss"
            val_vals = [r.get(val_key, None) for r in history]
            val_vals_clean = [(v if v is not None else float("nan")) for v in val_vals]
            if any(not np.isnan(v) for v in val_vals_clean):
                axes[i].semilogy(epochs, val_vals_clean, color="#F6511D",
                                 lw=1.2, ls="--", label="val_loss", alpha=0.8)

            axes[i].set_title(title, fontsize=10)
            axes[i].set_xlabel("Epoch", fontsize=9)
            axes[i].set_ylabel("Loss (log scale)", fontsize=9)
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.4)

        # Hide extra axes if any
        for j in range(len(components), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"{experiment_id} Training Loss Curves", fontsize=12)
        fig.tight_layout()

        path = self.out_dir / f"loss_curves_{experiment_id.lower()}.png"
        if save:
            fig.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight")
            logger.info("Saved: %s", path)
        plt.close(fig)
        return path

    # ── Figure 5: Residual Histograms ─────────────────────────────────────────

    def residual_histograms(
        self,
        residuals: Dict[str, np.ndarray],
        label: str = "validation",
        save: bool = True,
    ) -> Path:
        n_res = len(residuals)
        fig, axes = plt.subplots(1, n_res, figsize=(4 * n_res, 4), dpi=STYLE["dpi"])
        if n_res == 1:
            axes = [axes]

        colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]
        for (name, vals), ax, color in zip(residuals.items(), axes, colors):
            ax.hist(vals.flatten(), bins=50, color=color, alpha=0.75, edgecolor="white")
            ax.axvline(0, color="black", lw=1.2, ls="--", alpha=0.8)
            ax.set_title(name.replace("_", " "), fontsize=9)
            ax.set_xlabel("Residual value", fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            # Annotate RMS
            rms = float(np.sqrt(np.mean(vals ** 2)))
            ax.text(0.98, 0.95, f"RMS={rms:.3e}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Physics Residual Distributions ({label})", fontsize=11)
        fig.tight_layout()

        path = self.out_dir / f"residual_histograms_{label}.png"
        if save:
            fig.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight")
            logger.info("Saved: %s", path)
        plt.close(fig)
        return path

    # ── Figure 6: Summary Comparison Table ────────────────────────────────────

    def summary_table(
        self,
        metrics_mlp: Dict[str, float],
        metrics_pinn: Dict[str, float],
        split: str = "validation",
        save: bool = True,
    ) -> Path:
        metric_display = [
            ("voltage_rmse_V", "Voltage RMSE [V]"),
            ("temperature_rmse_K", "Temperature RMSE [K]"),
            ("concentration_rel_l2", "Concentration Rel. L2"),
            ("n_parameters", "Trainable Parameters"),
            ("median_latency_ms", "Median Latency [ms]"),
            ("p95_latency_ms", "P95 Latency [ms]"),
        ]

        rows = []
        for key, display_name in metric_display:
            mlp_val = metrics_mlp.get(key, float("nan"))
            pinn_val = metrics_pinn.get(key, float("nan"))
            # Format integers as int
            if key == "n_parameters":
                rows.append([display_name, f"{int(mlp_val):,}", f"{int(pinn_val):,}"])
            else:
                rows.append([display_name, f"{mlp_val:.4g}", f"{pinn_val:.4g}"])

        fig, ax = plt.subplots(figsize=STYLE["figsize_table"], dpi=STYLE["dpi"])
        ax.axis("off")

        headers = ["Metric", "EXP-002 MLP", "EXP-003 PINN"]
        col_widths = [0.45, 0.25, 0.25]
        table = ax.table(
            cellText=rows,
            colLabels=headers,
            cellLoc="center",
            loc="center",
            colWidths=col_widths,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)

        # Style header
        for j in range(len(headers)):
            table[(0, j)].set_facecolor("#2C3E50")
            table[(0, j)].set_text_props(color="white", fontweight="bold")

        # Alternate row colors
        for i in range(1, len(rows) + 1):
            color = "#EBF5FB" if i % 2 == 0 else "white"
            for j in range(len(headers)):
                table[(i, j)].set_facecolor(color)

        ax.set_title(
            f"Stage I Results: EXP-002 vs EXP-003 ({split} split)\n"
            "NOTE: All values measured from executed code.",
            fontsize=11, pad=20,
        )
        fig.tight_layout()

        path = self.out_dir / f"summary_table_{split}.png"
        if save:
            fig.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight")
            logger.info("Saved: %s", path)
        plt.close(fig)
        return path
