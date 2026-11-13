"""
src/preprocessing/nondim.py

Physically documented nondimensionalization for Stage I.

All scale constants come from configs/simulation_v0.yaml (nondim section).
They are NEVER hardcoded here — the module reads them from the config dict
and saves them to the training configuration artifact.

NONDIMENSIONALIZATION SUMMARY
==============================
This nondimensionalization is derived from the SPM/SPMe governing equations
for a single particle with lumped thermal coupling (Chen2020/LG M50).

Solid diffusion equation (radial, negative electrode):
    ∂c_s/∂t = D_s/r² · ∂/∂r(r² · ∂c_s/∂r)

Characteristic scales:
  Time:          t*   = t / t_ref          where t_ref = R_s_neg² / D_s0_neg
  Radius:        r*   = r / R_s_neg        so r* ∈ [0, 1]
  Concentration: c*   = c_s / c_s_max_neg  so c* ∈ [0, 1]
  Temperature:   T*   = (T - T_ref) / ΔT   linear shift and scale
  Voltage:       V*   = (V - V_ref) / ΔV   linear shift and scale
  Current:       I*   = I / I_ref          reference = nominal 1C current

These choices make the diffusion equation dimensionless:
    ∂c*/∂t* = 1/(r*²) · ∂/∂r*(r*² · ∂c*/∂r*)
with modified BC flux in dimensionless form.

References:
  - Newman & Thomas-Alyea (2004), Electrochemical Systems, Eq. 11.22
  - Plett (2015), Battery Management Systems Vol. 1, Ch. 3
  - Chen et al. (2020), J. Electrochem. Soc. 167 080534 (parameter source)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class Nondimensionalizer:
    """
    Converts SI-unit arrays to/from dimensionless form.

    Scale constants are read from the YAML config 'nondim' section:

      t_ref_s             : t_ref = R_s_neg² / D_s0_neg  [s]
      R_s_neg_m           : negative particle radius      [m]
      c_s_max_neg_mol_m3  : c_s_max (negative electrode) [mol/m³]
      T_ref_K             : temperature reference         [K]
      delta_T_K           : temperature scale             [K]
      V_ref_V             : voltage reference             [V]
      delta_V_V           : voltage scale                 [V]
      I_ref_A             : current reference (1C)        [A]

    All transforms are LINEAR (shift + scale) for invertibility.
    """

    def __init__(self, nondim_cfg: dict) -> None:
        cfg = nondim_cfg
        # --- Read scale constants (all have SI units as documented above) ---
        self.t_ref_s: float = float(cfg["t_ref_s"])
        self.R_s_neg_m: float = float(cfg["R_s_neg_m"])
        self.c_s_max_neg_mol_m3: float = float(cfg["c_s_max_neg_mol_m3"])
        self.T_ref_K: float = float(cfg["T_ref_K"])
        self.delta_T_K: float = float(cfg["delta_T_K"])
        self.V_ref_V: float = float(cfg["V_ref_V"])
        self.delta_V_V: float = float(cfg["delta_V_V"])
        self.I_ref_A: float = float(cfg["I_ref_A"])

        logger.info(
            "Nondimensionalizer initialized: t_ref=%.1f s, R_s=%.2e m, "
            "c_s_max=%.1f mol/m3, T_ref=%.2f K, dT=%.1f K, "
            "V_ref=%.2f V, dV=%.2f V, I_ref=%.2f A",
            self.t_ref_s, self.R_s_neg_m, self.c_s_max_neg_mol_m3,
            self.T_ref_K, self.delta_T_K, self.V_ref_V, self.delta_V_V, self.I_ref_A,
        )

    # ── Forward: SI → dimensionless ──────────────────────────────────────────

    def time_to_star(self, t_s: np.ndarray) -> np.ndarray:
        """t* = t_s / t_ref_s    [dimensionless]"""
        return np.asarray(t_s, dtype=np.float64) / self.t_ref_s

    def radius_to_star(self, r_m: np.ndarray) -> np.ndarray:
        """r* = r_m / R_s_neg_m    [dimensionless, 0–1]"""
        return np.asarray(r_m, dtype=np.float64) / self.R_s_neg_m

    def concentration_to_star(self, c_mol_m3: np.ndarray) -> np.ndarray:
        """c* = c_mol_m3 / c_s_max_neg_mol_m3    [dimensionless, 0–1]"""
        return np.asarray(c_mol_m3, dtype=np.float64) / self.c_s_max_neg_mol_m3

    def temperature_to_star(self, T_K: np.ndarray) -> np.ndarray:
        """T* = (T_K - T_ref_K) / delta_T_K    [dimensionless]"""
        return (np.asarray(T_K, dtype=np.float64) - self.T_ref_K) / self.delta_T_K

    def voltage_to_star(self, V_V: np.ndarray) -> np.ndarray:
        """V* = (V_V - V_ref_V) / delta_V_V    [dimensionless]"""
        return (np.asarray(V_V, dtype=np.float64) - self.V_ref_V) / self.delta_V_V

    def current_to_star(self, I_A: np.ndarray) -> np.ndarray:
        """I* = I_A / I_ref_A    [dimensionless]"""
        return np.asarray(I_A, dtype=np.float64) / self.I_ref_A

    # ── Inverse: dimensionless → SI ──────────────────────────────────────────

    def time_from_star(self, t_star: np.ndarray) -> np.ndarray:
        return np.asarray(t_star, dtype=np.float64) * self.t_ref_s

    def radius_from_star(self, r_star: np.ndarray) -> np.ndarray:
        return np.asarray(r_star, dtype=np.float64) * self.R_s_neg_m

    def concentration_from_star(self, c_star: np.ndarray) -> np.ndarray:
        return np.asarray(c_star, dtype=np.float64) * self.c_s_max_neg_mol_m3

    def temperature_from_star(self, T_star: np.ndarray) -> np.ndarray:
        return np.asarray(T_star, dtype=np.float64) * self.delta_T_K + self.T_ref_K

    def voltage_from_star(self, V_star: np.ndarray) -> np.ndarray:
        return np.asarray(V_star, dtype=np.float64) * self.delta_V_V + self.V_ref_V

    def current_from_star(self, I_star: np.ndarray) -> np.ndarray:
        return np.asarray(I_star, dtype=np.float64) * self.I_ref_A

    # ── Bulk conversion: full run arrays → dimensionless grid ─────────────────

    def nondimensionalize_run(
        self, arrays: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """
        Convert all SI arrays from one HDF5 run to dimensionless arrays.

        Parameters
        ----------
        arrays : dict
            Output of src.data.loaders.load_run_hdf5()

        Returns
        -------
        dict with keys:
          t_star, r_star, I_star, T_amb_star (scalar),
          V_star, c_star (shape n_t × n_r), T_star
        """
        t_star = self.time_to_star(arrays["time_s"])
        r_star = self.radius_to_star(arrays["r_neg_m"])
        I_star = self.current_to_star(arrays["current_A"])
        T_amb_K = float(arrays["ambient_temperature_K"])
        T_amb_star = float(self.temperature_to_star(np.array([T_amb_K]))[0])
        V_star = self.voltage_to_star(arrays["voltage_V"])
        c_star = self.concentration_to_star(arrays["concentration_neg_mol_m3"])
        T_star = self.temperature_to_star(arrays["temperature_K"])

        return {
            "t_star": t_star,          # (n_t,)
            "r_star": r_star,          # (n_r,)
            "I_star": I_star,          # (n_t,)
            "T_amb_star": T_amb_star,  # scalar float
            "V_star": V_star,          # (n_t,)
            "c_star": c_star,          # (n_t, n_r)
            "T_star": T_star,          # (n_t,)
        }

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return all scale constants as a plain dict (for JSON serialization)."""
        return {
            "t_ref_s": self.t_ref_s,
            "R_s_neg_m": self.R_s_neg_m,
            "c_s_max_neg_mol_m3": self.c_s_max_neg_mol_m3,
            "T_ref_K": self.T_ref_K,
            "delta_T_K": self.delta_T_K,
            "V_ref_V": self.V_ref_V,
            "delta_V_V": self.delta_V_V,
            "I_ref_A": self.I_ref_A,
        }

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved nondim constants to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> "Nondimensionalizer":
        with open(path) as f:
            cfg = json.load(f)
        return cls(cfg)
