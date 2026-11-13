"""
src/preprocessing/scalers.py

Data normalization for Stage I.

DESIGN PRINCIPLES:
  1. Fit only on TRAINING data — never on validation or test.
  2. Save fitted statistics to a versioned JSON artifact.
  3. Load the same artifact at evaluation time.
  4. All transforms are applied AFTER nondimensionalization.
     (i.e., these scalers operate on dimensionless quantities.)
  5. Standard scalers: zero-mean, unit-variance (z-score) per feature.

IMPORTANT: Do not mix nondimensionalization and data normalization.
  nondim.py  → physically derived scales (from governing equations)
  scalers.py → data-driven statistics (mean/std from training set)

For Stage I, the PINN input features are already approximately O(1) after
nondimensionalization, so optional additional scaling is applied for
neural network training stability only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class StandardScaler:
    """
    Feature-wise zero-mean, unit-variance scaler.

    Operates on numpy arrays of shape (N, D) or (N,).
    """

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self._fitted: bool = False

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """
        Compute mean and std from training data.

        Parameters
        ----------
        X : np.ndarray, shape (N,) or (N, D)
            TRAINING data only. Never call with val/test data.
        """
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        # Prevent division by zero: replace 0 std with 1.0
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.std_

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return np.asarray(X_scaled, dtype=np.float64) * self.std_ + self.mean_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Scaler has not been fitted yet. Call fit() first.")

    def to_dict(self) -> dict:
        self._check_fitted()
        return {
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StandardScaler":
        sc = cls()
        sc.mean_ = np.array(d["mean"], dtype=np.float64)
        sc.std_ = np.array(d["std"], dtype=np.float64)
        sc._fitted = True
        return sc


class ScalerBundle:
    """
    Container for all feature scalers used in Stage I.

    Scalers:
      inputs  : [t_star, r_star, I_star, T_amb_star]  → shape (4,)
      targets : [V_star, c_star, T_star]               → shape (3,)

    Fit on TRAINING data only. Save to JSON for reproducibility.

    NOTE: For Stage I, physics-based nondim already gives O(1) features,
    so these scalers provide an additional data-adaptive normalization layer
    that improves gradient flow without introducing systematic bias.
    """

    def __init__(self) -> None:
        self.input_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self._fitted: bool = False
        self._fit_note: str = ""

    def fit(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        fit_note: str = "",
    ) -> "ScalerBundle":
        """
        Fit both scalers on training data.

        Parameters
        ----------
        inputs  : np.ndarray shape (N, 4) — normalized inputs [t*, r*, I*, T_amb*]
        targets : np.ndarray shape (N, 3) — normalized targets [V*, c*, T*]
        fit_note : str — descriptive note for provenance (e.g., run IDs used)
        """
        assert inputs.ndim == 2 and inputs.shape[1] == 4, (
            f"inputs must be (N, 4), got {inputs.shape}"
        )
        assert targets.ndim == 2 and targets.shape[1] == 3, (
            f"targets must be (N, 3), got {targets.shape}"
        )

        self.input_scaler.fit(inputs)
        self.target_scaler.fit(targets)
        self._fitted = True
        self._fit_note = fit_note

        logger.info(
            "ScalerBundle fitted on %d samples. Input mean=%s std=%s",
            len(inputs),
            np.round(self.input_scaler.mean_, 3),
            np.round(self.input_scaler.std_, 3),
        )
        logger.info(
            "Target mean=%s std=%s",
            np.round(self.target_scaler.mean_, 3),
            np.round(self.target_scaler.std_, 3),
        )
        return self

    def transform_inputs(self, inputs: np.ndarray) -> np.ndarray:
        return self.input_scaler.transform(inputs)

    def transform_targets(self, targets: np.ndarray) -> np.ndarray:
        return self.target_scaler.transform(targets)

    def inverse_transform_targets(self, targets_scaled: np.ndarray) -> np.ndarray:
        return self.target_scaler.inverse_transform(targets_scaled)

    def save(self, path: Path | str) -> None:
        """Save fitted scalers to versioned JSON artifact."""
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted ScalerBundle.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "version": "1.0",
            "fit_note": self._fit_note,
            "input_features": ["t_star", "r_star", "I_star", "T_amb_star"],
            "target_features": ["V_star", "c_star", "T_star"],
            "input_scaler": self.input_scaler.to_dict(),
            "target_scaler": self.target_scaler.to_dict(),
        }
        with open(path, "w") as f:
            json.dump(artifact, f, indent=2)
        logger.info("Saved ScalerBundle to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> "ScalerBundle":
        """Load fitted scalers from JSON artifact."""
        with open(path) as f:
            artifact = json.load(f)
        bundle = cls()
        bundle.input_scaler = StandardScaler.from_dict(artifact["input_scaler"])
        bundle.target_scaler = StandardScaler.from_dict(artifact["target_scaler"])
        bundle._fitted = True
        bundle._fit_note = artifact.get("fit_note", "")
        logger.info("Loaded ScalerBundle from %s  note='%s'", path, bundle._fit_note)
        return bundle
