"""
src/data/loaders.py

PyTorch Dataset classes for the Stage I PINN pipeline.

Two dataset flavors:

  LabeledDataset
    - Returns labeled (input, target) pairs from processed Parquet index files.
    - Used by EXP-002 (MLP) and the data-loss term in EXP-003 (PINN).

  PINNDataset
    - Returns a combined batch with labeled points, interior collocation points,
      IC points, and BC points.
    - Used exclusively by EXP-003.

All inputs/outputs are in NORMALIZED units (produced by preprocessing pipeline).
Raw HDF5 files are never modified.

Normalization conventions (see src/preprocessing/nondim.py):
  t_star     = t_s / t_ref_s                    [dimensionless]
  r_star     = r_m / R_s_neg_m                  [dimensionless, 0–1]
  I_star     = I_A / I_ref_A                    [dimensionless]
  T_amb_star = (T_amb_K - T_ref_K) / delta_T_K  [dimensionless]
  V_star     = (V_V - V_ref_V) / delta_V_V      [dimensionless]
  c_star     = c_mol_m3 / c_s_max_neg_mol_m3    [dimensionless, 0–1]
  T_star     = (T_K - T_ref_K) / delta_T_K      [dimensionless]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class LabeledDataset(Dataset):
    """
    Labeled dataset for supervised training.

    Each sample is:
      inputs  : [t_star, r_star, I_star, T_amb_star]  shape (4,)
      targets : [V_star, c_star, T_star]               shape (3,)

    Parameters
    ----------
    parquet_index : Path | str
        Parquet file with columns:
          run_id, hdf5_path, t_idx, r_idx, t_star, r_star,
          I_star, T_amb_star, V_star, c_star, T_star
    device : str
        Torch device to move tensors to.
    """

    # Input/output feature names (for documentation)
    INPUT_NAMES = ["t_star", "r_star", "I_star", "T_amb_star"]
    TARGET_NAMES = ["V_star", "c_star", "T_star"]

    def __init__(self, parquet_index: Path | str, device: str = "cpu") -> None:
        self.device = device
        df = pd.read_parquet(str(parquet_index))
        self._validate_columns(df)

        self.inputs = torch.tensor(
            df[self.INPUT_NAMES].values, dtype=torch.float32
        ).to(device)
        self.targets = torch.tensor(
            df[self.TARGET_NAMES].values, dtype=torch.float32
        ).to(device)

        logger.info(
            "LabeledDataset: %d samples from %s", len(self.inputs), parquet_index
        )

    def _validate_columns(self, df: pd.DataFrame) -> None:
        required = self.INPUT_NAMES + self.TARGET_NAMES
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Parquet index missing columns: {missing}")

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.targets[idx]


class CollocationDataset(Dataset):
    """
    Collocation point dataset for PINN physics losses.

    Returns points for:
      - Interior physics (diffusion + thermal residuals)
      - Initial condition (t_star = 0)
      - Boundary condition (r_star = 0 or r_star = 1)

    Parameters
    ----------
    parquet_index : Path | str
        Parquet file with columns:
          point_type, t_star, r_star, I_star, T_amb_star,
          c_star_ic (for IC points), c_star_bc (for BC points)
    device : str
    """

    COORD_NAMES = ["t_star", "r_star", "I_star", "T_amb_star"]

    def __init__(self, parquet_index: Path | str, device: str = "cpu") -> None:
        self.device = device
        df = pd.read_parquet(str(parquet_index))
        required = self.COORD_NAMES + ["point_type"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Parquet collocation index missing columns: {missing}")

        self.coords = torch.tensor(
            df[self.COORD_NAMES].values, dtype=torch.float32
        ).to(device)
        self.point_types = df["point_type"].values  # "interior", "ic", "bc_center", "bc_surface"

        # Optional: IC/BC reference values stored for loss computation
        self.ic_ref: Optional[torch.Tensor] = None
        if "c_star_ic" in df.columns:
            self.ic_ref = torch.tensor(
                df["c_star_ic"].values, dtype=torch.float32
            ).to(device)

        logger.info(
            "CollocationDataset: %d points  types=%s",
            len(self.coords),
            dict(zip(*np.unique(self.point_types, return_counts=True))),
        )

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int):
        sample = {"coords": self.coords[idx], "point_type": self.point_types[idx]}
        if self.ic_ref is not None:
            sample["ic_ref"] = self.ic_ref[idx]
        return sample


class PINNDataset(Dataset):
    """
    Combined dataset for PINN training.

    Wraps both a LabeledDataset and CollocationDataset.
    Returns a flat dict with all needed tensors per batch.

    NOTE: This class pairs samples from the two datasets by cycling the
    shorter one. In practice, training uses separate DataLoaders with
    zip() in the trainer for more control. This class provides an
    alternative combined interface.

    Parameters
    ----------
    labeled_parquet : Path
    collocation_parquet : Path
    device : str
    """

    def __init__(
        self,
        labeled_parquet: Path | str,
        collocation_parquet: Path | str,
        device: str = "cpu",
    ) -> None:
        self.labeled = LabeledDataset(labeled_parquet, device=device)
        self.collocation = CollocationDataset(collocation_parquet, device=device)

    def __len__(self) -> int:
        return max(len(self.labeled), len(self.collocation))

    def __getitem__(self, idx: int):
        lab_idx = idx % len(self.labeled)
        col_idx = idx % len(self.collocation)
        inp, tgt = self.labeled[lab_idx]
        col = self.collocation[col_idx]
        return {
            "inputs": inp,
            "targets": tgt,
            "col_coords": col["coords"],
            "col_type": col["point_type"],
        }


# ── Utility: load full run arrays from HDF5 ───────────────────────────────────

def load_run_hdf5(hdf5_path: Path | str) -> dict[str, np.ndarray]:
    """
    Load all SI-unit arrays from an HDF5 run file.

    Returns
    -------
    dict with keys: time_s, r_neg_m, current_A, ambient_temperature_K,
    initial_soc, voltage_V, concentration_neg_mol_m3, temperature_K
    """
    arrays: dict[str, np.ndarray] = {}
    with h5py.File(str(hdf5_path), "r") as f:
        arrays["time_s"] = f["coordinates/time_s"][:]
        arrays["r_neg_m"] = f["coordinates/r_neg_m"][:]
        arrays["current_A"] = f["inputs/current_A"][:]
        arrays["ambient_temperature_K"] = float(f["inputs/ambient_temperature_K"][()])
        arrays["initial_soc"] = float(f["inputs/initial_soc"][()])
        arrays["voltage_V"] = f["outputs/voltage_V"][:]
        arrays["concentration_neg_mol_m3"] = f["outputs/concentration_neg_mol_m3"][:]
        arrays["temperature_K"] = f["outputs/temperature_K"][:]
    return arrays
