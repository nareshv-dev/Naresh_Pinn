"""
tests/test_database.py

Validates SQLite manifest schema, insertion, queries, and split assignments.
Does NOT require PyBaMM or a real dataset.
"""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.manifest import Manifest


class TestManifestSchema:
    """Test SQLite manifest schema and CRUD operations."""

    @pytest.fixture
    def manifest(self, tmp_path):
        """Fresh in-memory manifest for each test."""
        db_path = tmp_path / "test_manifest.sqlite"
        m = Manifest.open(db_path)
        yield m
        m.close()

    def test_tables_exist(self, manifest):
        """All required tables must exist."""
        required_tables = [
            "simulation_runs", "protocols", "parameter_sets",
            "dataset_splits", "artifacts", "training_runs", "metrics",
        ]
        cur = manifest._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        existing = {row[0] for row in cur.fetchall()}
        for tbl in required_tables:
            assert tbl in existing, f"Missing table: {tbl}"

    def test_seed_parameter_set(self, manifest):
        manifest.seed_parameter_set(
            set_id="Chen2020",
            name="Chen2020 (LG M50)",
            chemistry="NMC/graphite",
            cell_identifier="LG M50",
        )
        cur = manifest._conn.execute(
            "SELECT * FROM parameter_sets WHERE id='Chen2020'"
        )
        row = cur.fetchone()
        assert row is not None
        assert row["chemistry"] == "NMC/graphite"
        assert row["cell_identifier"] == "LG M50"

    def test_seed_parameter_set_idempotent(self, manifest):
        """Duplicate insert should be silently ignored."""
        manifest.seed_parameter_set("Chen2020", "A", "B")
        manifest.seed_parameter_set("Chen2020", "A", "B")
        cur = manifest._conn.execute("SELECT COUNT(*) FROM parameter_sets WHERE id='Chen2020'")
        count = cur.fetchone()[0]
        assert count == 1

    def test_seed_protocol(self, manifest):
        manifest.seed_protocol(
            protocol_id="CC_discharge_v0",
            name="CC Discharge",
            protocol_type="CC_discharge",
            c_rate=1.0,
            voltage_cutoff_V=2.5,
        )
        cur = manifest._conn.execute(
            "SELECT * FROM protocols WHERE id='CC_discharge_v0'"
        )
        row = cur.fetchone()
        assert row is not None
        assert abs(row["voltage_cutoff_V"] - 2.5) < 1e-9

    def test_register_run_passed(self, manifest):
        meta = {
            "run_id": "run_test_001",
            "pybamm_version": "24.9.0",
            "model_name": "SPMe",
            "thermal_option": "lumped",
            "parameter_set": "Chen2020",
            "protocol_id": "CC_discharge_v0",
            "solver": "CasadiSolver",
            "rtol": 1e-6,
            "atol": 1e-6,
            "c_rate": 1.0,
            "ambient_temperature_K": 298.15,
            "initial_soc": 1.0,
            "timestamp": "2026-08-24T00:00:00+00:00",
            "git_commit": "abc123",
            "random_seed": 42,
            "hdf5_path": "/data/run_test_001.h5",
            "checksum_sha256": "deadbeef" * 8,
            "solver_status": "success",
            "quality_passed": True,
            "n_time_steps": 100,
            "n_r_points": 5,
        }
        manifest.register_run(meta)
        cur = manifest._conn.execute(
            "SELECT * FROM simulation_runs WHERE run_id='run_test_001'"
        )
        row = cur.fetchone()
        assert row is not None
        assert row["quality_passed"] == 1
        assert abs(row["c_rate"] - 1.0) < 1e-9
        assert row["model_name"] == "SPMe"

    def test_register_run_failed(self, manifest):
        """Failed runs must be stored with quality_passed=0."""
        meta = {
            "run_id": "run_fail_001",
            "pybamm_version": "24.9.0",
            "model_name": "SPMe",
            "thermal_option": "lumped",
            "parameter_set": "Chen2020",
            "protocol_id": "CC_discharge_v0",
            "solver": "CasadiSolver",
            "rtol": 1e-6,
            "atol": 1e-6,
            "c_rate": 3.0,
            "ambient_temperature_K": 298.15,
            "initial_soc": 1.0,
            "timestamp": "2026-08-24T00:00:00+00:00",
            "git_commit": None,
            "random_seed": 42,
            "hdf5_path": None,
            "checksum_sha256": None,
            "solver_status": "failed: solver error",
            "quality_passed": False,
            "n_time_steps": 0,
            "n_r_points": 0,
        }
        manifest.register_run(meta)
        cur = manifest._conn.execute(
            "SELECT quality_passed FROM simulation_runs WHERE run_id='run_fail_001'"
        )
        assert cur.fetchone()[0] == 0

    def test_assign_split(self, manifest):
        meta = {
            "run_id": "run_split_001",
            "pybamm_version": "24.9.0",
            "model_name": "SPMe",
            "thermal_option": "lumped",
            "parameter_set": None,
            "protocol_id": None,
            "solver": "CasadiSolver",
            "rtol": 1e-6,
            "atol": 1e-6,
            "c_rate": 1.0,
            "ambient_temperature_K": 298.15,
            "initial_soc": 1.0,
            "timestamp": "2026-08-24T00:00:00+00:00",
            "git_commit": None,
            "random_seed": 42,
            "hdf5_path": "/tmp/run.h5",
            "checksum_sha256": None,
            "solver_status": "success",
            "quality_passed": True,
            "n_time_steps": 100,
            "n_r_points": 5,
        }
        manifest.register_run(meta)
        manifest.assign_split("run_split_001", "train", "v0")

        cur = manifest._conn.execute(
            "SELECT split_name FROM dataset_splits WHERE run_id='run_split_001'"
        )
        row = cur.fetchone()
        assert row is not None
        assert row[0] == "train"

    def test_get_runs_by_split(self, manifest):
        """Only quality-passed runs should be returned."""
        for i, (qp, split) in enumerate([
            (True, "train"), (True, "validation"), (False, "train"),
        ]):
            meta = {
                "run_id": f"run_qp_{i}",
                "pybamm_version": "24.9.0", "model_name": "SPMe",
                "thermal_option": "lumped", "parameter_set": None,
                "protocol_id": None, "solver": "CasadiSolver",
                "rtol": 1e-6, "atol": 1e-6,
                "c_rate": 1.0, "ambient_temperature_K": 298.15, "initial_soc": 1.0,
                "timestamp": "2026-08-24T00:00:00+00:00", "git_commit": None,
                "random_seed": 42, "hdf5_path": f"/tmp/r{i}.h5",
                "checksum_sha256": None, "solver_status": "success",
                "quality_passed": qp, "n_time_steps": 10, "n_r_points": 3,
            }
            manifest.register_run(meta)
            manifest.assign_split(f"run_qp_{i}", split, "v0")

        train_runs = manifest.get_runs_by_split("train", "v0")
        # Only quality_passed=True train run should appear
        assert len(train_runs) == 1
        assert train_runs[0]["run_id"] == "run_qp_0"

    def test_log_metric(self, manifest):
        """Log a metric and verify retrieval."""
        # Register a training run first
        manifest.register_training_run({
            "id": "tr_001",
            "experiment_id": "EXP-002",
            "model_class": "MLP",
            "config_version": "v0",
            "hidden_dims": [128, 64],
            "activation": "tanh",
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "max_epochs": 100,
            "n_parameters": 10000,
            "status": "done",
        })
        manifest.log_metric("tr_001", "validation", "voltage_rmse_V", 0.0542, "V")
        manifest.log_metric("tr_001", "validation", "temperature_rmse_K", 0.31, "K")

        metrics = manifest.get_metrics("tr_001")
        assert len(metrics) == 2
        names = {m["metric_name"] for m in metrics}
        assert "voltage_rmse_V" in names
        assert "temperature_rmse_K" in names

    def test_register_artifact(self, manifest):
        manifest.register_artifact(
            "scaler_json", "/data/scalers_v0.json",
            "Fitted scalers", "v0"
        )
        cur = manifest._conn.execute(
            "SELECT * FROM artifacts WHERE artifact_type='scaler_json'"
        )
        row = cur.fetchone()
        assert row is not None
        assert row["path"] == "/data/scalers_v0.json"
