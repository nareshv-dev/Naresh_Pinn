"""
src/evaluation/metrics.py

Evaluation metrics for Stage I.

All metrics are computed in SI units (after inverse nondimensionalization).
Only exception: relative L2 error (dimensionless by definition).

Metrics implemented:
  - voltage_rmse_V           : RMSE of terminal voltage [V]
  - temperature_rmse_K       : RMSE of cell temperature [K]
  - concentration_rel_l2     : Relative L2 error of c_s [dimensionless]
  - per_residual_norm        : L2 norm of each physics residual
  - ic_violation             : L2 norm of IC residual at t=0
  - bc_center_violation      : L2 norm of BC symmetry residual
  - bc_surface_violation     : L2 norm of BC surface flux residual
  - n_parameters             : total trainable parameter count
  - training_time_s          : total training time [s]
  - median_latency_ms        : median inference latency [ms]
  - p95_latency_ms           : 95th percentile inference latency [ms]
  - peak_memory_mb           : peak CPU/GPU memory usage [MB] (if available)

IMPORTANT: All metrics are computed from actually executed code.
No values are presented until the code has run and produced them.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def compute_metrics(
    model: nn.Module,
    inputs: torch.Tensor,           # (N, 4) normalized
    targets: torch.Tensor,          # (N, 3) normalized  [V*, c*, T*]
    nondim,                          # Nondimensionalizer instance
    device: str = "cpu",
    n_latency_trials: int = 100,
) -> Dict[str, float]:
    """
    Compute all regression metrics for a model on a dataset split.

    Parameters
    ----------
    model : nn.Module — MLP or PINN
    inputs : Tensor (N, 4)  [t*, r*, I*, T_amb*]
    targets : Tensor (N, 3) [V*, c*, T*]
    nondim : Nondimensionalizer — for inverse transforms
    device : str
    n_latency_trials : int

    Returns
    -------
    dict of scalar metric values.
    """
    model.eval()
    inputs = inputs.to(device)
    targets = targets.to(device)

    with torch.no_grad():
        preds = model(inputs)   # (N, 3) [V*, c*, T*]

    preds_np = preds.cpu().numpy().astype(np.float64)
    targets_np = targets.cpu().numpy().astype(np.float64)

    # -- Inverse nondimensionalize --
    # Column indices: 0=V*, 1=c*, 2=T*
    V_pred = nondim.voltage_from_star(preds_np[:, 0])     # [V]
    V_true = nondim.voltage_from_star(targets_np[:, 0])   # [V]

    c_pred = nondim.concentration_from_star(preds_np[:, 1])   # [mol/m³]
    c_true = nondim.concentration_from_star(targets_np[:, 1]) # [mol/m³]

    T_pred = nondim.temperature_from_star(preds_np[:, 2])  # [K]
    T_true = nondim.temperature_from_star(targets_np[:, 2])  # [K]

    # -- Voltage RMSE [V] --
    voltage_rmse_V = float(np.sqrt(np.mean((V_pred - V_true) ** 2)))

    # -- Temperature RMSE [K] --
    temperature_rmse_K = float(np.sqrt(np.mean((T_pred - T_true) ** 2)))

    # -- Concentration relative L2 error [dimensionless] --
    c_denom = float(np.sqrt(np.mean(c_true ** 2)))
    if c_denom < 1e-8:
        concentration_rel_l2 = float("nan")
    else:
        concentration_rel_l2 = float(
            np.sqrt(np.mean((c_pred - c_true) ** 2)) / c_denom
        )

    # -- Latency measurement --
    latency_ms = _measure_latency(model, inputs, n_latency_trials, device)

    # -- Memory (psutil, best effort) --
    peak_memory_mb = _measure_memory(model, inputs, device)

    metrics = {
        "voltage_rmse_V": voltage_rmse_V,
        "temperature_rmse_K": temperature_rmse_K,
        "concentration_rel_l2": concentration_rel_l2,
        "n_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "median_latency_ms": latency_ms["median"],
        "p95_latency_ms": latency_ms["p95"],
        "peak_memory_mb": peak_memory_mb,
    }

    logger.info("Metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    return metrics


def compute_residual_norms(
    pinn,                               # PINN model
    col_interior: torch.Tensor,         # (M, 4)
    col_ic: torch.Tensor,               # (M_ic, 4)
    col_bc_center: torch.Tensor,        # (M_bc, 4)
    col_bc_surface: torch.Tensor,       # (M_bc, 4)
    c_s0_star: float,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Compute L2 norm of each physics residual.

    Returns
    -------
    dict with keys:
      diffusion_residual_l2, thermal_residual_l2,
      ic_concentration_l2, ic_thermal_l2,
      bc_center_l2, bc_surface_l2
    """
    from src.physics.residuals import SolidDiffusionResidual, LumpedThermalResidual

    pinn.eval()
    col_interior = col_interior.to(device)
    col_ic = col_ic.to(device)
    col_bc_center = col_bc_center.to(device)
    col_bc_surface = col_bc_surface.to(device)

    diff_res = pinn.diff_residual
    therm_res = pinn.therm_residual

    def split_coords(batch):
        t = batch[:, 0:1].detach().requires_grad_(True)
        r = batch[:, 1:2].detach().requires_grad_(True)
        I = batch[:, 2:3].detach()
        T_amb = batch[:, 3:4].detach()
        return t, r, I, T_amb

    t_int, r_int, I_int, T_int = split_coords(col_interior)
    t_ic, r_ic, I_ic, T_ic = split_coords(col_ic)
    t_bcc, r_bcc, I_bcc, T_bcc = split_coords(col_bc_center)
    t_bcs, r_bcs, I_bcs, T_bcs = split_coords(col_bc_surface)

    c_s0 = torch.full((len(col_ic), 1), c_s0_star, dtype=torch.float32, device=device)

    # Diffusion residual
    res_diff = diff_res.interior_residual(pinn.net, t_int, r_int, I_int, T_int)
    diff_l2 = float(res_diff.pow(2).mean().sqrt().item())

    # Thermal residual
    res_therm = therm_res.interior_residual(pinn.net, t_int, r_int, I_int, T_int)
    therm_l2 = float(res_therm.pow(2).mean().sqrt().item())

    # IC concentration
    res_ic_c = diff_res.ic_residual(pinn.net, t_ic, r_ic, I_ic, T_ic, c_s0)
    ic_c_l2 = float(res_ic_c.pow(2).mean().sqrt().item())

    # IC thermal
    res_ic_t = therm_res.ic_residual(pinn.net, t_ic, r_ic, I_ic, T_ic)
    ic_t_l2 = float(res_ic_t.pow(2).mean().sqrt().item())

    # BC center
    res_bcc = diff_res.bc_center_residual(pinn.net, t_bcc, r_bcc, I_bcc, T_bcc)
    bcc_l2 = float(res_bcc.pow(2).mean().sqrt().item())

    # BC surface
    res_bcs = diff_res.bc_surface_residual(pinn.net, t_bcs, r_bcs, I_bcs, T_bcs)
    bcs_l2 = float(res_bcs.pow(2).mean().sqrt().item())

    norms = {
        "diffusion_residual_l2": diff_l2,
        "thermal_residual_l2": therm_l2,
        "ic_concentration_l2": ic_c_l2,
        "ic_thermal_l2": ic_t_l2,
        "bc_center_l2": bcc_l2,
        "bc_surface_l2": bcs_l2,
    }
    logger.info("Residual norms: %s", {k: f"{v:.4e}" for k, v in norms.items()})
    return norms


def _measure_latency(
    model: nn.Module,
    inputs: torch.Tensor,
    n_trials: int,
    device: str,
) -> Dict[str, float]:
    """Measure inference latency over n_trials forward passes."""
    model.eval()
    # Warmup
    with torch.no_grad():
        for _ in range(min(5, n_trials)):
            _ = model(inputs[:1])

    latencies_ms = []
    with torch.no_grad():
        for _ in range(n_trials):
            t0 = time.perf_counter()
            _ = model(inputs[:1])
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    arr = np.array(latencies_ms)
    return {
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


def _measure_memory(
    model: nn.Module,
    inputs: torch.Tensor,
    device: str,
) -> float:
    """Best-effort peak memory measurement [MB]."""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 ** 2)
        with torch.no_grad():
            _ = model(inputs)
        mem_after = process.memory_info().rss / (1024 ** 2)
        return max(0.0, mem_after - mem_before)
    except Exception:
        return float("nan")
