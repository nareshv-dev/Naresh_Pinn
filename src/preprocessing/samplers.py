"""
src/preprocessing/samplers.py

Samplers for labeled data points, interior collocation, IC, and BC points.

All samplers operate in DIMENSIONLESS (normalized) coordinates.
Sampling is reproducible via deterministic seeds.

SAMPLER TYPES:
  LabeledSampler
    - Samples (t*, r*) pairs from actual HDF5 run data
    - Returns the corresponding target values (V*, c*, T*)
    - For train/val/test sets (using labeled observations)

  CollocationSampler
    - Interior: samples (t*, r*) ∈ (0, t_max*] × (0, 1) for PDE residuals
    - IC: samples r* ∈ (0, 1) at t* = 0 with known c*(r, 0) = c_s0/c_s_max
    - BC at r* = 0 (symmetry): no-flux, ∂c*/∂r* = 0
    - BC at r* = 1 (surface): flux determined by current I*
    - Uses Sobol or Latin Hypercube quasi-random sequences (scipy.stats.qmc)
    - Outputs Parquet files for reproducibility

PARQUET SCHEMA:
  Labeled index:
    run_id, sample_id, t_star, r_star, I_star, T_amb_star,
    V_star, c_star, T_star, split

  Collocation index:
    run_id, sample_id, point_type, t_star, r_star, I_star, T_amb_star,
    c_star_ic (IC only), split
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats.qmc import Sobol, LatinHypercube, scale as qmc_scale

logger = logging.getLogger(__name__)


class LabeledSampler:
    """
    Samples labeled (input, output) pairs from normalized run arrays.

    Each (t*, r*) point is paired with its outputs (V*, c*, T*).
    Since V and T are functions of t only (not r), they are broadcast.
    The c_star output corresponds to the specific (t, r) grid point.

    Parameters
    ----------
    n_samples_per_run : int
        Number of (t*, r*) samples to draw per run.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, n_samples_per_run: int, seed: int = 42) -> None:
        self.n_per_run = n_samples_per_run
        self.rng = np.random.default_rng(seed)

    def sample_run(
        self,
        nondim_arrays: dict,
        run_id: str,
        split: str,
    ) -> pd.DataFrame:
        """
        Sample from one nondimensionalized run.

        Parameters
        ----------
        nondim_arrays : dict
            Output of Nondimensionalizer.nondimensionalize_run().
            Keys: t_star (n_t,), r_star (n_r,), I_star (n_t,),
                  T_amb_star (float), V_star (n_t,),
                  c_star (n_t, n_r), T_star (n_t,)
        run_id : str
        split : str — "train", "validation", "test"

        Returns
        -------
        pd.DataFrame with labeled sample rows.
        """
        t_star = nondim_arrays["t_star"]       # (n_t,)
        r_star = nondim_arrays["r_star"]       # (n_r,)
        I_star = nondim_arrays["I_star"]       # (n_t,)
        T_amb_star = float(nondim_arrays["T_amb_star"])
        V_star = nondim_arrays["V_star"]       # (n_t,)
        c_star = nondim_arrays["c_star"]       # (n_t, n_r)
        T_star = nondim_arrays["T_star"]       # (n_t,)

        n_t, n_r = c_star.shape
        n = min(self.n_per_run, n_t * n_r)

        # Random sample of (t_idx, r_idx) pairs
        t_idx = self.rng.integers(0, n_t, size=n)
        r_idx = self.rng.integers(0, n_r, size=n)

        rows = {
            "run_id": [run_id] * n,
            "sample_id": [f"{run_id}_s{i}" for i in range(n)],
            "split": [split] * n,
            "t_star": t_star[t_idx],
            "r_star": r_star[r_idx],
            "I_star": I_star[t_idx],
            "T_amb_star": np.full(n, T_amb_star),
            # Targets
            "V_star": V_star[t_idx],          # V depends only on t
            "c_star": c_star[t_idx, r_idx],   # c depends on (t, r)
            "T_star": T_star[t_idx],           # T depends only on t
        }
        return pd.DataFrame(rows)

    def sample_multiple_runs(
        self,
        run_list: list[dict],
        split: str,
    ) -> pd.DataFrame:
        """
        Sample from a list of runs.

        Parameters
        ----------
        run_list : list[dict]
            Each dict has keys: "run_id" and "nondim_arrays".
        split : str

        Returns
        -------
        Combined pd.DataFrame.
        """
        frames = []
        for run in run_list:
            df = self.sample_run(run["nondim_arrays"], run["run_id"], split)
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


class CollocationSampler:
    """
    Samples physics collocation points in dimensionless space.

    Generates four types of points:
      "interior"    : t* ∈ (0, t_max*], r* ∈ (0, 1) — for PDE residuals
      "ic"          : t* = 0, r* ∈ [0, 1]            — initial condition
      "bc_center"   : t* ∈ (0, t_max*], r* = 0       — symmetry BC
      "bc_surface"  : t* ∈ (0, t_max*], r* = 1       — surface flux BC

    Parameters
    ----------
    n_interior : int  — interior collocation points
    n_ic : int        — IC collocation points
    n_bc : int        — BC points (split equally between center and surface)
    sampler_type : str — "sobol" or "latin_hypercube"
    seed : int
    """

    def __init__(
        self,
        n_interior: int,
        n_ic: int,
        n_bc: int,
        sampler_type: str = "sobol",
        seed: int = 42,
    ) -> None:
        self.n_interior = n_interior
        self.n_ic = n_ic
        self.n_bc = n_bc
        self.sampler_type = sampler_type
        self.seed = seed

    def _qmc_2d(self, n: int, offset: int = 0) -> np.ndarray:
        """
        Generate n quasi-random points in [0,1]² using chosen sampler.
        Returns shape (n, 2).
        """
        if self.sampler_type == "sobol":
            sampler = Sobol(d=2, scramble=True, seed=self.seed + offset)
            pts = sampler.random(n)
        else:  # latin_hypercube
            sampler = LatinHypercube(d=2, seed=self.seed + offset)
            pts = sampler.random(n)
        return pts.astype(np.float64)

    def _qmc_1d(self, n: int, offset: int = 0) -> np.ndarray:
        """Generate n quasi-random points in [0,1]. Returns shape (n,)."""
        if self.sampler_type == "sobol":
            sampler = Sobol(d=1, scramble=True, seed=self.seed + offset)
            pts = sampler.random(n)
        else:
            sampler = LatinHypercube(d=1, seed=self.seed + offset)
            pts = sampler.random(n)
        return pts[:, 0].astype(np.float64)

    def sample(
        self,
        t_max_star: float,
        I_star_func,            # callable: t_star → I_star (1D interp)
        T_amb_star: float,
        c_s0_star: float,       # initial concentration (dimensionless)
        run_id: str,
        split: str,
    ) -> pd.DataFrame:
        """
        Generate all collocation points for one operating condition.

        Parameters
        ----------
        t_max_star : float — maximum dimensionless time for this condition
        I_star_func : callable — maps t_star → I_star (from run data)
        T_amb_star : float — dimensionless ambient temperature
        c_s0_star : float — initial c* = c_s(t=0) / c_s_max (IC value)
        run_id : str
        split : str

        Returns
        -------
        pd.DataFrame with collocation rows.
        """
        rows_list = []

        # ── Interior points: (t*, r*) ∈ (0, t_max*] × (0, 1) ────────────────
        pts_int = self._qmc_2d(self.n_interior, offset=0)
        t_int = pts_int[:, 0] * t_max_star   # scale from [0,1] to [0, t_max*]
        t_int = np.clip(t_int, 1e-6, t_max_star)  # avoid t*=0 boundary
        r_int = pts_int[:, 1]
        r_int = np.clip(r_int, 1e-6, 1.0 - 1e-6)  # avoid r*=0,1 boundaries
        I_int = np.array([I_star_func(ti) for ti in t_int])

        interior_df = pd.DataFrame({
            "run_id": run_id,
            "sample_id": [f"{run_id}_col_int_{i}" for i in range(self.n_interior)],
            "split": split,
            "point_type": "interior",
            "t_star": t_int,
            "r_star": r_int,
            "I_star": I_int,
            "T_amb_star": T_amb_star,
            "c_star_ic": np.nan,
        })
        rows_list.append(interior_df)

        # ── IC points: t* = 0, r* ∈ [0, 1] ──────────────────────────────────
        r_ic = self._qmc_1d(self.n_ic, offset=100)
        ic_df = pd.DataFrame({
            "run_id": run_id,
            "sample_id": [f"{run_id}_col_ic_{i}" for i in range(self.n_ic)],
            "split": split,
            "point_type": "ic",
            "t_star": np.zeros(self.n_ic),
            "r_star": r_ic,
            "I_star": np.array([I_star_func(0.0)] * self.n_ic),
            "T_amb_star": T_amb_star,
            "c_star_ic": np.full(self.n_ic, c_s0_star),   # known IC value
        })
        rows_list.append(ic_df)

        # ── BC center (symmetry): t* ∈ (0, t_max*], r* = 0 ──────────────────
        n_bc_each = self.n_bc // 2
        t_bc_c = self._qmc_1d(n_bc_each, offset=200) * t_max_star
        t_bc_c = np.clip(t_bc_c, 1e-6, t_max_star)
        I_bc_c = np.array([I_star_func(ti) for ti in t_bc_c])
        bc_center_df = pd.DataFrame({
            "run_id": run_id,
            "sample_id": [f"{run_id}_col_bcc_{i}" for i in range(n_bc_each)],
            "split": split,
            "point_type": "bc_center",
            "t_star": t_bc_c,
            "r_star": np.zeros(n_bc_each),  # r* = 0
            "I_star": I_bc_c,
            "T_amb_star": T_amb_star,
            "c_star_ic": np.nan,
        })
        rows_list.append(bc_center_df)

        # ── BC surface (intercalation flux): t* ∈ (0, t_max*], r* = 1 ───────
        t_bc_s = self._qmc_1d(n_bc_each, offset=300) * t_max_star
        t_bc_s = np.clip(t_bc_s, 1e-6, t_max_star)
        I_bc_s = np.array([I_star_func(ti) for ti in t_bc_s])
        bc_surface_df = pd.DataFrame({
            "run_id": run_id,
            "sample_id": [f"{run_id}_col_bcs_{i}" for i in range(n_bc_each)],
            "split": split,
            "point_type": "bc_surface",
            "t_star": t_bc_s,
            "r_star": np.ones(n_bc_each),   # r* = 1
            "I_star": I_bc_s,
            "T_amb_star": T_amb_star,
            "c_star_ic": np.nan,
        })
        rows_list.append(bc_surface_df)

        return pd.concat(rows_list, ignore_index=True)

    def sample_multiple_runs(
        self,
        run_list: list[dict],
        split: str,
    ) -> pd.DataFrame:
        """
        Sample collocation from multiple runs.

        Each dict in run_list must have:
          run_id, t_max_star, I_star_func, T_amb_star, c_s0_star
        """
        frames = []
        for run in run_list:
            df = self.sample(
                t_max_star=run["t_max_star"],
                I_star_func=run["I_star_func"],
                T_amb_star=run["T_amb_star"],
                c_s0_star=run["c_s0_star"],
                run_id=run["run_id"],
                split=split,
            )
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
