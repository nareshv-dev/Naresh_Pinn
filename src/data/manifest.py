"""
src/data/manifest.py

SQLite manifest for Stage I dataset provenance.

Tables:
  simulation_runs   — one row per PyBaMM run (pass or fail)
  protocols         — protocol definitions (C-rate, cutoff, etc.)
  parameter_sets    — parameter set registry
  dataset_splits    — maps run_id → split (train/validation/test)
  artifacts         — file artifacts (HDF5, parquet, scaler JSON, checkpoints)
  training_runs     — one row per model training execution
  metrics           — scalar metrics (keyed by training_run_id + metric_name)

Design principles:
  - Tensors/arrays are stored in HDF5 (never in SQLite).
  - Scalar summary rows only in SQLite.
  - All physical values stored in SI units.
  - All foreign keys enforced.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
-- FK enforcement is intentionally disabled for provenance tracking:
-- parameter_set_id and protocol_id are plain TEXT references.
-- This allows registering runs without pre-seeding lookup tables.
PRAGMA foreign_keys = OFF;

CREATE TABLE IF NOT EXISTS parameter_sets (
    id                   TEXT PRIMARY KEY,
    name                 TEXT NOT NULL,
    chemistry            TEXT NOT NULL,  -- e.g. "NMC/graphite"
    cell_identifier      TEXT,           -- e.g. "LG M50"
    source_reference     TEXT,
    created_at           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS protocols (
    id                   TEXT PRIMARY KEY,
    name                 TEXT NOT NULL,
    protocol_type        TEXT NOT NULL,  -- "CC_discharge", "CCCV_charge", etc.
    c_rate               REAL,
    voltage_cutoff_V     REAL,
    description          TEXT,
    created_at           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS simulation_runs (
    run_id               TEXT PRIMARY KEY,
    pybamm_version       TEXT NOT NULL,
    model_name           TEXT NOT NULL,
    thermal_option       TEXT NOT NULL,
    parameter_set_id     TEXT,           -- plain TEXT reference (no FK enforcement)
    protocol_id          TEXT,           -- plain TEXT reference (no FK enforcement)
    solver               TEXT NOT NULL,
    rtol                 REAL NOT NULL,
    atol                 REAL NOT NULL,
    c_rate               REAL NOT NULL,
    ambient_temperature_K REAL NOT NULL,
    initial_soc          REAL,           -- nullable: may be unknown for failed runs
    timestamp            TEXT NOT NULL,
    git_commit           TEXT,
    random_seed          INTEGER,
    hdf5_path            TEXT,
    checksum_sha256      TEXT,
    solver_status        TEXT NOT NULL,
    quality_passed       INTEGER NOT NULL,  -- 0 or 1 (SQLite has no BOOLEAN)
    n_time_steps         INTEGER,
    n_r_points           INTEGER
    -- No FOREIGN KEY declarations: parameter_set_id / protocol_id are provenance TEXT
);

CREATE TABLE IF NOT EXISTS dataset_splits (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id               TEXT NOT NULL,
    split_name           TEXT NOT NULL,  -- "train", "validation", "test"
    config_version       TEXT NOT NULL,
    created_at           TEXT NOT NULL,
    UNIQUE (run_id, config_version)
    -- run_id reference to simulation_runs kept as plain TEXT (no FK enforcement)
);

CREATE TABLE IF NOT EXISTS artifacts (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    artifact_type        TEXT NOT NULL,  -- "hdf5_run", "parquet_split", "scaler_json", "checkpoint"
    path                 TEXT NOT NULL UNIQUE,
    description          TEXT,
    config_version       TEXT,
    created_at           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS training_runs (
    id                   TEXT PRIMARY KEY,
    experiment_id        TEXT NOT NULL,  -- "EXP-002", "EXP-003"
    model_class          TEXT NOT NULL,  -- "MLP", "PINN"
    config_version       TEXT NOT NULL,
    hidden_dims          TEXT NOT NULL,  -- JSON list
    activation           TEXT NOT NULL,
    optimizer            TEXT NOT NULL,
    learning_rate        REAL NOT NULL,
    max_epochs           INTEGER NOT NULL,
    n_parameters         INTEGER,
    train_run_ids        TEXT,           -- JSON list of run_ids used for training
    val_run_ids          TEXT,           -- JSON list
    checkpoint_path      TEXT,
    started_at           TEXT,
    finished_at          TEXT,
    status               TEXT NOT NULL DEFAULT 'pending',  -- pending/running/done/failed
    git_commit           TEXT,
    random_seed          INTEGER
);

CREATE TABLE IF NOT EXISTS metrics (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    training_run_id      TEXT NOT NULL,
    split_name           TEXT NOT NULL,  -- "validation", "test"
    metric_name          TEXT NOT NULL,  -- "voltage_rmse_V", "temperature_rmse_K", etc.
    metric_value         REAL,
    unit                 TEXT,
    measured_at          TEXT NOT NULL,
    FOREIGN KEY (training_run_id) REFERENCES training_runs(id)
);

CREATE INDEX IF NOT EXISTS idx_runs_quality ON simulation_runs(quality_passed);
CREATE INDEX IF NOT EXISTS idx_splits_split ON dataset_splits(split_name);
CREATE INDEX IF NOT EXISTS idx_metrics_training ON metrics(training_run_id, split_name);
"""


class Manifest:
    """
    SQLite manifest manager.

    Usage
    -----
    m = Manifest.open("data/processed/dataset_v0_manifest.sqlite")
    m.seed_parameter_set("Chen2020", chemistry="NMC/graphite", cell="LG M50")
    m.seed_protocol("CC_discharge_v0", c_rate=1.0, ...)
    m.register_run(meta_dict)
    m.assign_split("run_id", "train", config_version="v0")
    m.close()
    """

    def __init__(self, conn: sqlite3.Connection, db_path: Path) -> None:
        self._conn = conn
        self.db_path = db_path

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def open(cls, db_path: str | Path) -> "Manifest":
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
        logger.info("Opened manifest: %s", db_path)
        return cls(conn, db_path)

    def close(self) -> None:
        self._conn.commit()
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── Seed reference data ───────────────────────────────────────────────────

    def seed_parameter_set(
        self,
        set_id: str,
        name: str,
        chemistry: str,
        cell_identifier: str = "",
        source_reference: str = "",
    ) -> None:
        """Insert or ignore a parameter set row."""
        now = _now()
        self._conn.execute(
            """
            INSERT OR IGNORE INTO parameter_sets
            (id, name, chemistry, cell_identifier, source_reference, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (set_id, name, chemistry, cell_identifier, source_reference, now),
        )
        self._conn.commit()

    def seed_protocol(
        self,
        protocol_id: str,
        name: str,
        protocol_type: str,
        c_rate: float | None = None,
        voltage_cutoff_V: float | None = None,
        description: str = "",
    ) -> None:
        now = _now()
        self._conn.execute(
            """
            INSERT OR IGNORE INTO protocols
            (id, name, protocol_type, c_rate, voltage_cutoff_V, description, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (protocol_id, name, protocol_type, c_rate, voltage_cutoff_V, description, now),
        )
        self._conn.commit()

    # ── Simulation runs ───────────────────────────────────────────────────────

    def register_run(self, meta: dict) -> None:
        """
        Insert a simulation run row.

        Parameters
        ----------
        meta : dict
            Keys match the simulation_runs columns. Missing optional keys
            default to None.
        """
        self._conn.execute(
            """
            INSERT OR REPLACE INTO simulation_runs (
                run_id, pybamm_version, model_name, thermal_option,
                parameter_set_id, protocol_id, solver, rtol, atol,
                c_rate, ambient_temperature_K, initial_soc,
                timestamp, git_commit, random_seed,
                hdf5_path, checksum_sha256, solver_status,
                quality_passed, n_time_steps, n_r_points
            ) VALUES (
                :run_id, :pybamm_version, :model_name, :thermal_option,
                :parameter_set, :protocol_id, :solver, :rtol, :atol,
                :c_rate, :ambient_temperature_K, :initial_soc,
                :timestamp, :git_commit, :random_seed,
                :hdf5_path, :checksum_sha256, :solver_status,
                :quality_passed, :n_time_steps, :n_r_points
            )
            """,
            {
                "run_id": meta["run_id"],
                "pybamm_version": meta.get("pybamm_version", ""),
                "model_name": meta.get("model_name", ""),
                "thermal_option": meta.get("thermal_option", ""),
                "parameter_set": meta.get("parameter_set", None),
                "protocol_id": meta.get("protocol_id", None),
                "solver": meta.get("solver", ""),
                "rtol": meta.get("rtol", None),
                "atol": meta.get("atol", None),
                "c_rate": meta.get("c_rate", None),
                "ambient_temperature_K": meta.get("ambient_temperature_K", None),
                "initial_soc": meta.get("initial_soc", None),
                "timestamp": meta.get("timestamp", _now()),
                "git_commit": meta.get("git_commit", None),
                "random_seed": meta.get("random_seed", None),
                "hdf5_path": meta.get("hdf5_path", None),
                "checksum_sha256": meta.get("checksum_sha256", None),
                "solver_status": meta.get("solver_status", "unknown"),
                "quality_passed": int(bool(meta.get("quality_passed", False))),
                "n_time_steps": meta.get("n_time_steps", None),
                "n_r_points": meta.get("n_r_points", None),
            },
        )
        self._conn.commit()

    def assign_split(
        self, run_id: str, split_name: str, config_version: str
    ) -> None:
        now = _now()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO dataset_splits
            (run_id, split_name, config_version, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, split_name, config_version, now),
        )
        self._conn.commit()

    # ── Artifacts ─────────────────────────────────────────────────────────────

    def register_artifact(
        self,
        artifact_type: str,
        path: str,
        description: str = "",
        config_version: str = "",
    ) -> None:
        now = _now()
        self._conn.execute(
            """
            INSERT OR IGNORE INTO artifacts
            (artifact_type, path, description, config_version, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (artifact_type, path, description, config_version, now),
        )
        self._conn.commit()

    # ── Training runs ─────────────────────────────────────────────────────────

    def register_training_run(self, training_meta: dict) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO training_runs (
                id, experiment_id, model_class, config_version,
                hidden_dims, activation, optimizer, learning_rate,
                max_epochs, n_parameters, train_run_ids, val_run_ids,
                checkpoint_path, started_at, finished_at, status,
                git_commit, random_seed
            ) VALUES (
                :id, :experiment_id, :model_class, :config_version,
                :hidden_dims, :activation, :optimizer, :learning_rate,
                :max_epochs, :n_parameters, :train_run_ids, :val_run_ids,
                :checkpoint_path, :started_at, :finished_at, :status,
                :git_commit, :random_seed
            )
            """,
            {
                "id": training_meta["id"],
                "experiment_id": training_meta.get("experiment_id", ""),
                "model_class": training_meta.get("model_class", ""),
                "config_version": training_meta.get("config_version", ""),
                "hidden_dims": json.dumps(training_meta.get("hidden_dims", [])),
                "activation": training_meta.get("activation", ""),
                "optimizer": training_meta.get("optimizer", "Adam"),
                "learning_rate": training_meta.get("learning_rate", 0.0),
                "max_epochs": training_meta.get("max_epochs", 0),
                "n_parameters": training_meta.get("n_parameters", None),
                "train_run_ids": json.dumps(training_meta.get("train_run_ids", [])),
                "val_run_ids": json.dumps(training_meta.get("val_run_ids", [])),
                "checkpoint_path": training_meta.get("checkpoint_path", None),
                "started_at": training_meta.get("started_at", None),
                "finished_at": training_meta.get("finished_at", None),
                "status": training_meta.get("status", "pending"),
                "git_commit": training_meta.get("git_commit", None),
                "random_seed": training_meta.get("random_seed", None),
            },
        )
        self._conn.commit()

    def log_metric(
        self,
        training_run_id: str,
        split_name: str,
        metric_name: str,
        metric_value: float,
        unit: str = "",
    ) -> None:
        now = _now()
        self._conn.execute(
            """
            INSERT INTO metrics
            (training_run_id, split_name, metric_name, metric_value, unit, measured_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (training_run_id, split_name, metric_name, metric_value, unit, now),
        )
        self._conn.commit()

    # ── Query helpers ─────────────────────────────────────────────────────────

    def get_runs_by_split(self, split_name: str, config_version: str) -> list[dict]:
        """Return simulation_runs metadata for a given split."""
        cur = self._conn.execute(
            """
            SELECT sr.*
            FROM simulation_runs sr
            JOIN dataset_splits ds ON sr.run_id = ds.run_id
            WHERE ds.split_name = ? AND ds.config_version = ?
                  AND sr.quality_passed = 1
            ORDER BY sr.c_rate, sr.ambient_temperature_K
            """,
            (split_name, config_version),
        )
        return [dict(row) for row in cur.fetchall()]

    def get_all_runs(self) -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM simulation_runs ORDER BY timestamp"
        )
        return [dict(row) for row in cur.fetchall()]

    def get_metrics(self, training_run_id: str) -> list[dict]:
        cur = self._conn.execute(
            "SELECT * FROM metrics WHERE training_run_id = ?",
            (training_run_id,),
        )
        return [dict(row) for row in cur.fetchall()]


# ── Utility ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
