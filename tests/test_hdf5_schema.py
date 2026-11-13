"""
tests/test_hdf5_schema.py

Validates the HDF5 schema: groups, datasets, dtypes, coordinate ranges.
Uses a synthetic HDF5 file — does NOT require PyBaMM to be installed.
"""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest


def make_synthetic_hdf5(path: str) -> None:
    """Create a minimal valid HDF5 file matching the expected schema."""
    n_t, n_r = 50, 5
    time_s = np.linspace(0, 3600.0, n_t)
    r_neg_m = np.linspace(0, 5.86e-6, n_r)
    current_A = np.full(n_t, 5.0)
    voltage_V = np.linspace(4.1, 2.8, n_t)
    temperature_K = np.linspace(298.15, 305.0, n_t)
    concentration = np.tile(np.linspace(28700, 15000, n_t)[:, None], (1, n_r))

    with h5py.File(path, "w") as f:
        # coordinates
        coords = f.create_group("coordinates")
        coords.create_dataset("time_s", data=time_s, dtype="float64")
        coords["time_s"].attrs["units"] = "s"
        coords.create_dataset("r_neg_m", data=r_neg_m, dtype="float64")
        coords["r_neg_m"].attrs["units"] = "m"

        # inputs
        inp = f.create_group("inputs")
        inp.create_dataset("current_A", data=current_A, dtype="float64")
        inp.create_dataset("ambient_temperature_K", data=np.float64(298.15), dtype="float64")
        inp.create_dataset("initial_soc", data=np.float64(1.0), dtype="float64")

        # outputs
        out = f.create_group("outputs")
        out.create_dataset("voltage_V", data=voltage_V, dtype="float64")
        out.create_dataset("concentration_neg_mol_m3", data=concentration, dtype="float64")
        out.create_dataset("temperature_K", data=temperature_K, dtype="float64")

        # quality
        q = f.create_group("quality")
        q.attrs["passed"] = 1
        q.attrs["all_finite"] = 1
        q.attrs["voltage_in_bounds"] = 1
        q.attrs["temperature_in_bounds"] = 1
        q.attrs["concentration_in_bounds"] = 1
        q.attrs["monotonic_time"] = 1
        q.attrs["monotonic_r"] = 1
        q.attrs["shape_consistent"] = 1

        # metadata
        f.attrs["run_id"] = "test_run_001"
        f.attrs["pybamm_version"] = "24.9.0"
        f.attrs["model_name"] = "SPMe"
        f.attrs["thermal_option"] = "lumped"
        f.attrs["parameter_set"] = "Chen2020"
        f.attrs["protocol_id"] = "CC_discharge_smoke"
        f.attrs["solver"] = "CasadiSolver"
        f.attrs["rtol"] = 1.0e-6
        f.attrs["atol"] = 1.0e-6
        f.attrs["c_rate"] = 1.0
        f.attrs["ambient_temperature_K"] = 298.15
        f.attrs["initial_soc"] = 1.0
        f.attrs["timestamp"] = "2026-08-24T00:00:00+00:00"
        f.attrs["git_commit"] = "unknown"
        f.attrs["random_seed"] = 42
        f.attrs["schema_version"] = "1.0"


class TestHDF5Schema:
    """Validate the HDF5 schema produced by the simulation runner."""

    @pytest.fixture(autouse=True)
    def tmp_hdf5(self, tmp_path):
        """Create a synthetic HDF5 in a temp directory."""
        self.hdf5_path = str(tmp_path / "test_run.h5")
        make_synthetic_hdf5(self.hdf5_path)

    def test_groups_exist(self):
        with h5py.File(self.hdf5_path, "r") as f:
            assert "coordinates" in f, "Missing group: coordinates"
            assert "inputs" in f, "Missing group: inputs"
            assert "outputs" in f, "Missing group: outputs"
            assert "quality" in f, "Missing group: quality"

    def test_coordinates_datasets(self):
        with h5py.File(self.hdf5_path, "r") as f:
            assert "time_s" in f["coordinates"]
            assert "r_neg_m" in f["coordinates"]
            assert f["coordinates/time_s"].dtype == np.float64
            assert f["coordinates/r_neg_m"].dtype == np.float64

    def test_inputs_datasets(self):
        with h5py.File(self.hdf5_path, "r") as f:
            assert "current_A" in f["inputs"]
            assert "ambient_temperature_K" in f["inputs"]
            assert "initial_soc" in f["inputs"]

    def test_outputs_datasets(self):
        with h5py.File(self.hdf5_path, "r") as f:
            assert "voltage_V" in f["outputs"]
            assert "concentration_neg_mol_m3" in f["outputs"]
            assert "temperature_K" in f["outputs"]
            assert f["outputs/voltage_V"].dtype == np.float64
            assert f["outputs/temperature_K"].dtype == np.float64

    def test_coordinate_ranges(self):
        with h5py.File(self.hdf5_path, "r") as f:
            time_s = f["coordinates/time_s"][:]
            r_neg_m = f["coordinates/r_neg_m"][:]

            assert np.all(time_s >= 0), "Time must be non-negative"
            assert np.all(np.diff(time_s) > 0), "Time must be strictly monotonic"
            assert np.all(r_neg_m >= 0), "Radius must be non-negative"
            assert np.all(np.diff(r_neg_m) > 0), "Radius must be strictly monotonic"

    def test_concentration_shape(self):
        with h5py.File(self.hdf5_path, "r") as f:
            n_t = len(f["coordinates/time_s"][:])
            n_r = len(f["coordinates/r_neg_m"][:])
            c_shape = f["outputs/concentration_neg_mol_m3"].shape
            assert c_shape == (n_t, n_r), (
                f"concentration shape {c_shape} ≠ ({n_t}, {n_r})"
            )

    def test_voltage_shape_matches_time(self):
        with h5py.File(self.hdf5_path, "r") as f:
            n_t = len(f["coordinates/time_s"][:])
            assert f["outputs/voltage_V"].shape == (n_t,)
            assert f["outputs/temperature_K"].shape == (n_t,)
            assert f["inputs/current_A"].shape == (n_t,)

    def test_metadata_attributes(self):
        required_attrs = [
            "run_id", "pybamm_version", "model_name", "thermal_option",
            "parameter_set", "protocol_id", "solver", "rtol", "atol",
            "c_rate", "ambient_temperature_K", "initial_soc",
            "timestamp", "git_commit", "random_seed", "schema_version",
        ]
        with h5py.File(self.hdf5_path, "r") as f:
            for attr in required_attrs:
                assert attr in f.attrs, f"Missing metadata attribute: {attr}"

    def test_quality_group(self):
        with h5py.File(self.hdf5_path, "r") as f:
            qattrs = dict(f["quality"].attrs)
            assert "passed" in qattrs
            assert "all_finite" in qattrs

    def test_all_finite(self):
        with h5py.File(self.hdf5_path, "r") as f:
            assert np.all(np.isfinite(f["outputs/voltage_V"][:]))
            assert np.all(np.isfinite(f["outputs/temperature_K"][:]))
            assert np.all(np.isfinite(f["outputs/concentration_neg_mol_m3"][:]))

    def test_voltage_bounds(self):
        with h5py.File(self.hdf5_path, "r") as f:
            v = f["outputs/voltage_V"][:]
            assert np.all(v >= 2.4), f"Voltage min {v.min():.3f} < 2.4 V"
            assert np.all(v <= 4.3), f"Voltage max {v.max():.3f} > 4.3 V"

    def test_temperature_bounds(self):
        with h5py.File(self.hdf5_path, "r") as f:
            t = f["outputs/temperature_K"][:]
            assert np.all(t >= 230.0), f"Temperature min {t.min():.1f} K"
            assert np.all(t <= 360.0), f"Temperature max {t.max():.1f} K"

    def test_load_run_hdf5(self):
        """Test the data loader utility."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.data.loaders import load_run_hdf5

        arrays = load_run_hdf5(self.hdf5_path)
        expected_keys = [
            "time_s", "r_neg_m", "current_A", "ambient_temperature_K",
            "initial_soc", "voltage_V", "concentration_neg_mol_m3", "temperature_K",
        ]
        for k in expected_keys:
            assert k in arrays, f"Missing key in loaded arrays: {k}"
