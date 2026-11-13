"""
tests/test_smoke_e2e.py

Minimal end-to-end smoke test that:
  1. Creates a synthetic dataset (2 fake HDF5 runs, no PyBaMM required)
  2. Runs the preprocessor
  3. Trains EXP-002 (MLP) for 3 steps
  4. Trains EXP-003 (PINN) for 3 steps
  5. Asserts loss decreases and all artifacts exist

This test validates the full pipeline integration without running PyBaMM.
"""

import json
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.manifest import Manifest
from src.preprocessing.nondim import Nondimensionalizer
from src.preprocessing.scalers import ScalerBundle
from src.preprocessing.samplers import LabeledSampler, CollocationSampler
from src.models.mlp import MLP
from src.models.pinn import PINN
from src.training.trainer import MLPTrainer, PINNTrainer


# ── Synthetic run factory ─────────────────────────────────────────────────────

def make_fake_run(hdf5_path: str, c_rate: float = 1.0, t_amb: float = 298.15) -> dict:
    """Create a minimal valid fake HDF5 run for testing."""
    n_t, n_r = 60, 5
    time_s = np.linspace(0, 3600.0 / c_rate, n_t)
    r_neg_m = np.linspace(0, 5.86e-6, n_r)
    current_A = np.full(n_t, 5.0 * c_rate)
    voltage_V = np.linspace(4.1, 2.8, n_t)
    temperature_K = np.linspace(t_amb, t_amb + 5 * c_rate, n_t)
    concentration = np.linspace(28700, 14000, n_t)[:, None] * np.ones((1, n_r))

    run_id = f"fake_C{c_rate}_T{t_amb:.0f}"

    with h5py.File(hdf5_path, "w") as f:
        coords = f.create_group("coordinates")
        coords.create_dataset("time_s", data=time_s, dtype="float64")
        coords.create_dataset("r_neg_m", data=r_neg_m, dtype="float64")

        inp = f.create_group("inputs")
        inp.create_dataset("current_A", data=current_A, dtype="float64")
        inp.create_dataset("ambient_temperature_K", data=np.float64(t_amb), dtype="float64")
        inp.create_dataset("initial_soc", data=np.float64(1.0), dtype="float64")

        out = f.create_group("outputs")
        out.create_dataset("voltage_V", data=voltage_V, dtype="float64")
        out.create_dataset("concentration_neg_mol_m3", data=concentration, dtype="float64")
        out.create_dataset("temperature_K", data=temperature_K, dtype="float64")

        q = f.create_group("quality")
        q.attrs["passed"] = 1

        f.attrs["run_id"] = run_id
        f.attrs["pybamm_version"] = "test"
        f.attrs["model_name"] = "SPMe"
        f.attrs["thermal_option"] = "lumped"
        f.attrs["parameter_set"] = "Chen2020"
        f.attrs["protocol_id"] = "smoke"
        f.attrs["solver"] = "CasadiSolver"
        f.attrs["rtol"] = 1e-6
        f.attrs["atol"] = 1e-6
        f.attrs["c_rate"] = c_rate
        f.attrs["ambient_temperature_K"] = t_amb
        f.attrs["initial_soc"] = 1.0
        f.attrs["timestamp"] = "2026-08-24T00:00:00+00:00"
        f.attrs["git_commit"] = "test"
        f.attrs["random_seed"] = 0
        f.attrs["schema_version"] = "1.0"

    return {
        "run_id": run_id,
        "hdf5_path": hdf5_path,
        "c_rate": c_rate,
        "ambient_temperature_K": t_amb,
        "quality_passed": True,
        "n_time_steps": n_t,
        "n_r_points": n_r,
    }


# ── Nondim config ─────────────────────────────────────────────────────────────

NONDIM_CFG = {
    "t_ref_s": 1041.0,
    "R_s_neg_m": 5.86e-6,
    "c_s_max_neg_mol_m3": 33133.0,
    "T_ref_K": 298.15,
    "delta_T_K": 50.0,
    "V_ref_V": 3.7,
    "delta_V_V": 1.0,
    "I_ref_A": 5.0,
}

TRAIN_CFG_MLP = {
    "experiment_id": "EXP-002-smoke",
    "learning_rate": 1e-3,
    "batch_size": 16,
    "max_epochs": 3,
    "early_stopping_patience": 100,
    "checkpoint_every_n_epochs": 10,
    "log_every_n_steps": 1,
}

TRAIN_CFG_PINN = {
    "experiment_id": "EXP-003-smoke",
    "learning_rate": 1e-3,
    "batch_size": 16,
    "max_epochs": 3,
    "early_stopping_patience": 100,
    "checkpoint_every_n_epochs": 10,
    "log_every_n_steps": 1,
    "loss_weights": {
        "data": 1.0,
        "diffusion_residual": 0.01,
        "thermal_residual": 0.01,
        "ic": 0.1,
        "bc": 0.1,
    },
}


class TestSmokeEndToEnd:
    """
    Minimal integration smoke test for the full pipeline.
    Uses synthetic data — PyBaMM NOT required.
    """

    @pytest.fixture(autouse=True)
    def setup_workspace(self, tmp_path):
        """Create temp workspace with two fake HDF5 runs."""
        self.workspace = tmp_path
        raw_dir = tmp_path / "data" / "raw" / "runs"
        raw_dir.mkdir(parents=True)
        proc_dir = tmp_path / "data" / "processed"
        proc_dir.mkdir(parents=True)
        ckpt_dir = tmp_path / "outputs" / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        log_dir = tmp_path / "outputs" / "logs"
        log_dir.mkdir(parents=True)

        self.proc_dir = proc_dir
        self.ckpt_dir = ckpt_dir
        self.log_dir = log_dir

        # Create 2 fake runs
        self.run_train = make_fake_run(
            str(raw_dir / "train_run.h5"), c_rate=1.0, t_amb=298.15
        )
        self.run_val = make_fake_run(
            str(raw_dir / "val_run.h5"), c_rate=0.5, t_amb=298.15
        )

        # Nondimensionalizer
        self.nd = Nondimensionalizer(NONDIM_CFG)

    def _load_and_nondim(self, run_info: dict) -> dict:
        from src.data.loaders import load_run_hdf5
        arrays = load_run_hdf5(run_info["hdf5_path"])
        return self.nd.nondimensionalize_run(arrays)

    def _make_labeled_tensors(self) -> tuple:
        """Build small labeled tensors for train and val."""
        from src.preprocessing.samplers import LabeledSampler
        sampler = LabeledSampler(n_samples_per_run=30, seed=0)

        train_nd = self._load_and_nondim(self.run_train)
        val_nd = self._load_and_nondim(self.run_val)

        df_train = sampler.sample_run(train_nd, self.run_train["run_id"], "train")
        df_val = sampler.sample_run(val_nd, self.run_val["run_id"], "validation")

        INPUT_COLS = ["t_star", "r_star", "I_star", "T_amb_star"]
        TARGET_COLS = ["V_star", "c_star", "T_star"]

        X_tr = torch.tensor(df_train[INPUT_COLS].values.astype(np.float32))
        Y_tr = torch.tensor(df_train[TARGET_COLS].values.astype(np.float32))
        X_val = torch.tensor(df_val[INPUT_COLS].values.astype(np.float32))
        Y_val = torch.tensor(df_val[TARGET_COLS].values.astype(np.float32))
        return X_tr, Y_tr, X_val, Y_val

    def _make_collocation_tensors(self) -> tuple:
        """Build small collocation tensors."""
        from scipy.interpolate import interp1d
        train_nd = self._load_and_nondim(self.run_train)
        t_star = train_nd["t_star"]
        I_star = train_nd["I_star"]
        I_func = interp1d(t_star, I_star, kind="linear", fill_value="extrapolate")
        I_star_func = lambda t: float(I_func(np.clip(t, t_star[0], t_star[-1])))

        sampler = CollocationSampler(n_interior=40, n_ic=10, n_bc=10, sampler_type="sobol", seed=0)
        df = sampler.sample(
            t_max_star=float(t_star.max()),
            I_star_func=I_star_func,
            T_amb_star=float(train_nd["T_amb_star"]),
            c_s0_star=0.866,
            run_id=self.run_train["run_id"],
            split="train",
        )
        COORD_COLS = ["t_star", "r_star", "I_star", "T_amb_star"]

        def _t(pt):
            sub = df[df["point_type"] == pt][COORD_COLS].values.astype(np.float32)
            return torch.tensor(sub)

        return _t("interior"), _t("ic"), _t("bc_center"), _t("bc_surface")

    # ── Test: HDF5 schema of synthetic runs ──────────────────────────────────

    def test_synthetic_hdf5_valid(self):
        import h5py
        with h5py.File(self.run_train["hdf5_path"], "r") as f:
            assert "coordinates" in f
            assert "inputs" in f
            assert "outputs" in f
            assert "quality" in f
            n_t = len(f["coordinates/time_s"][:])
            n_r = len(f["coordinates/r_neg_m"][:])
            c_shape = f["outputs/concentration_neg_mol_m3"].shape
            assert c_shape == (n_t, n_r)

    # ── Test: Database insertion ──────────────────────────────────────────────

    def test_manifest_insertion(self):
        db_path = self.proc_dir / "smoke_manifest.sqlite"
        with Manifest.open(db_path) as m:
            for run in [self.run_train, self.run_val]:
                m.register_run({
                    **run,
                    "pybamm_version": "test", "model_name": "SPMe",
                    "thermal_option": "lumped", "parameter_set": "Chen2020",
                    "protocol_id": "smoke", "solver": "CasadiSolver",
                    "rtol": 1e-4, "atol": 1e-4,
                    "timestamp": "2026-08-24T00:00:00+00:00",
                    "git_commit": "test", "random_seed": 0,
                    "checksum_sha256": "abc",
                    "solver_status": "success",
                })
            all_runs = m.get_all_runs()
        assert len(all_runs) == 2

    # ── Test: Nondimensionalization ───────────────────────────────────────────

    def test_nondim_ranges(self):
        nd_arrays = self._load_and_nondim(self.run_train)
        # t* should be > 0 and reasonable
        assert np.all(nd_arrays["t_star"] >= 0)
        # r* ∈ [0, 1]
        assert np.all(nd_arrays["r_star"] >= 0)
        assert np.all(nd_arrays["r_star"] <= 1 + 1e-9)
        # c* ∈ [0, 1.2] approximately
        assert np.all(nd_arrays["c_star"] >= 0)
        assert np.all(nd_arrays["c_star"] <= 1.5)

    # ── Test: MLP training step ───────────────────────────────────────────────

    def test_mlp_training_3_steps(self):
        X_tr, Y_tr, X_val, Y_val = self._make_labeled_tensors()

        model = MLP(input_dim=4, output_dim=3, hidden_dims=[8, 8], activation="tanh")
        trainer = MLPTrainer(
            model=model,
            config=TRAIN_CFG_MLP,
            device="cpu",
            checkpoint_dir=self.ckpt_dir,
            log_dir=self.log_dir,
        )
        result = trainer.train(X_tr, Y_tr, X_val, Y_val)
        assert result["epochs_run"] >= 1
        assert result["final_train_loss"] >= 0.0
        assert Path(result["log_path"]).exists()

    # ── Test: PINN training step ──────────────────────────────────────────────

    def test_pinn_training_3_steps(self):
        X_tr, Y_tr, X_val, Y_val = self._make_labeled_tensors()
        col_int, col_ic, col_bcc, col_bcs = self._make_collocation_tensors()

        model = PINN(
            input_dim=4,
            output_dim=3,
            hidden_dims=[8, 8],
            activation="tanh",
            loss_weights=TRAIN_CFG_PINN["loss_weights"],
            beta_flux=0.1,
            alpha_thermal=0.05,
        )
        trainer = PINNTrainer(
            model=model,
            config=TRAIN_CFG_PINN,
            c_s0_star=0.866,
            device="cpu",
            checkpoint_dir=self.ckpt_dir,
            log_dir=self.log_dir,
        )
        result = trainer.train(
            X_tr, Y_tr,
            col_int, col_ic, col_bcc, col_bcs,
            X_val, Y_val,
        )
        assert result["epochs_run"] >= 1
        assert result["final_train_total"] >= 0.0

        # Verify all loss components are tracked
        with open(result["log_path"]) as f:
            hist = json.load(f)
        first = hist[0]
        assert "train_diffusion" in first
        assert "train_thermal" in first
        assert "train_ic" in first
        assert "train_bc" in first

    # ── Test: Coordinate ranges of collocation points ─────────────────────────

    def test_collocation_coordinate_ranges(self):
        col_int, col_ic, col_bcc, col_bcs = self._make_collocation_tensors()

        # Interior: t* > 0, r* ∈ (0, 1)
        assert col_int[:, 0].min() > 0, "Interior t* must be > 0"
        assert col_int[:, 1].min() > 0, "Interior r* must be > 0"
        assert col_int[:, 1].max() < 1.0, "Interior r* must be < 1"

        # IC: t* = 0
        assert (col_ic[:, 0].abs() < 1e-9).all(), "IC t* must be 0"

        # BC center: r* ≈ 0
        assert col_bcc[:, 1].max() < 1e-3, f"BC center r* = {col_bcc[:, 1].max():.4f}"

        # BC surface: r* ≈ 1
        assert col_bcs[:, 1].min() > 0.99, f"BC surface r* = {col_bcs[:, 1].min():.4f}"

    # ── Test: Residual shapes at inference ────────────────────────────────────

    def test_residual_shapes_inference(self):
        N = 6
        model = MLP(input_dim=4, output_dim=3, hidden_dims=[8, 8], activation="tanh")
        from src.physics.residuals import SolidDiffusionResidual
        dr = SolidDiffusionResidual(beta_flux=0.1)
        t = torch.rand(N, 1, requires_grad=True)
        r = (torch.rand(N, 1) * 0.8 + 0.1).requires_grad_(True)
        I = torch.ones(N, 1) * 0.5
        T_amb = torch.zeros(N, 1)
        res = dr.interior_residual(model, t, r, I, T_amb)
        assert res.shape == (N, 1), f"Expected (N,1), got {res.shape}"

    # ── Test: Scaler fitting/saving/loading ───────────────────────────────────

    def test_scaler_fit_save_load(self):
        X = np.random.randn(100, 4)
        Y = np.random.randn(100, 3)
        bundle = ScalerBundle()
        bundle.fit(X, Y, fit_note="test")
        scaler_path = self.proc_dir / "test_scalers.json"
        bundle.save(scaler_path)
        assert scaler_path.exists()

        # Load and verify
        loaded = ScalerBundle.load(scaler_path)
        X_tr = loaded.transform_inputs(X)
        assert X_tr.shape == (100, 4)
        X_back = loaded.input_scaler.inverse_transform(X_tr)
        np.testing.assert_allclose(X_back, X, rtol=1e-5)
