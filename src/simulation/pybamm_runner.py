"""
src/simulation/pybamm_runner.py

PyBaMM simulation runner for Stage I data generation.

Model:    SPMe (Single Particle Model with Electrolyte)
Thermal:  lumped (single ODE for average cell temperature T(t))
Params:   Chen2020 (LG M50, NMC/graphite cell)

Outputs per run (all in SI units):
  coordinates/time_s          : shape (n_time,)
  coordinates/r_neg_m         : shape (n_r,)  — negative particle radial grid
  inputs/current_A            : shape (n_time,) — signed, positive = discharge
  inputs/ambient_temperature_K: scalar
  inputs/initial_soc          : scalar
  outputs/voltage_V           : shape (n_time,)
  outputs/concentration_neg_mol_m3 : shape (n_time, n_r)
  outputs/temperature_K       : shape (n_time,)
  quality/solver_status       : str
  quality/voltage_in_bounds   : bool
  quality/temperature_in_bounds : bool
  quality/concentration_in_bounds : bool
  quality/all_finite          : bool
  quality/passed              : bool
  metadata attributes:
    run_id, pybamm_version, model_name, thermal_option, parameter_set,
    protocol_id, solver, rtol, atol, timestamp, git_commit, random_seed, checksum

HDF5 schema verified by tests/test_hdf5_schema.py.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pybamm
import yaml

logger = logging.getLogger(__name__)


class PyBaMMRunner:
    """
    Runs a factorial sweep of PyBaMM SPMe + lumped thermal simulations,
    exports results to HDF5, and registers runs in the SQLite manifest.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration (simulation_v0.yaml or smoke_test.yaml).
    project_root : Path
        Absolute path to the project root directory.
    manifest : optional
        An open `src.data.manifest.Manifest` instance. If None, DB registration
        is skipped (useful for unit tests).
    """

    def __init__(
        self,
        config: dict,
        project_root: Path,
        manifest: Any = None,
    ) -> None:
        self.cfg = config
        self.root = project_root
        self.manifest = manifest
        self.pybamm_cfg = config["pybamm"]
        self.sweep_cfg = config["sweep"]
        self.quality_cfg = config["quality"]
        self.out_dir = project_root / config["paths"]["raw_data_dir"]
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Read parameter set name for provenance
        self.param_set_name: str = self.pybamm_cfg["parameter_set"]
        self.model_name: str = self.pybamm_cfg["model_name"]
        self.thermal_option: str = self.pybamm_cfg["model_options"]["thermal"]
        self.solver_name: str = self.pybamm_cfg["solver"]
        self.rtol: float = float(self.pybamm_cfg["solver_options"]["rtol"])
        self.atol: float = float(self.pybamm_cfg["solver_options"]["atol"])
        self.n_r: int = int(self.pybamm_cfg["mesh"]["n_points_r_neg"])
        self.protocol_id: str = self.sweep_cfg["protocol"]["protocol_id"]
        self.v_cutoff: float = float(
            self.sweep_cfg["protocol"]["voltage_cutoff_V"]
        )
        self.seed: int = int(config.get("random_seed", 42))
        self.git_commit: str = self._get_git_commit()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_sweep(self) -> list[dict]:
        """
        Execute all factorial combinations defined in config["sweep"].

        Returns
        -------
        list[dict]
            One metadata dict per run (whether passed or failed).
        """
        c_rates = self.sweep_cfg["c_rates"]
        temps = self.sweep_cfg["ambient_temperatures_K"]
        socs = self.sweep_cfg["initial_soc_values"]

        results: list[dict] = []
        for c_rate in c_rates:
            for t_amb in temps:
                for soc_init in socs:
                    meta = self._run_single(
                        c_rate=float(c_rate),
                        t_amb_K=float(t_amb),
                        soc_init=float(soc_init),
                    )
                    results.append(meta)
        return results

    # ------------------------------------------------------------------
    # Private: single simulation
    # ------------------------------------------------------------------

    def _run_single(
        self,
        c_rate: float,
        t_amb_K: float,
        soc_init: float,
    ) -> dict:
        """
        Run one SPMe + lumped-thermal simulation, export HDF5, and register.

        Returns
        -------
        dict
            Run metadata including quality flags and file path.
        """
        run_id = f"run_{self.model_name}_{self.param_set_name}_C{c_rate:.2f}_T{t_amb_K:.1f}_SOC{soc_init:.2f}_{uuid.uuid4().hex[:8]}"
        logger.info(
            "Starting run_id=%s  C-rate=%.2f  T_amb=%.1f K  SOC=%.2f",
            run_id, c_rate, t_amb_K, soc_init,
        )

        timestamp = datetime.now(timezone.utc).isoformat()
        hdf5_path = self.out_dir / f"{run_id}.h5"

        # --- Build and run simulation ---
        try:
            sol, param = self._build_and_solve(c_rate, t_amb_K, soc_init)
            solver_status = "success"
        except Exception as exc:
            logger.error("Simulation failed for run_id=%s: %s", run_id, exc)
            meta = self._make_failed_meta(
                run_id, c_rate, t_amb_K, soc_init, timestamp, str(exc)
            )
            if self.manifest:
                self.manifest.register_run(meta)
            return meta

        # --- Extract arrays ---
        try:
            arrays = self._extract_arrays(sol, param, c_rate, t_amb_K, soc_init)
        except Exception as exc:
            logger.error("Array extraction failed for run_id=%s: %s", run_id, exc)
            meta = self._make_failed_meta(
                run_id, c_rate, t_amb_K, soc_init, timestamp, f"extraction: {exc}"
            )
            if self.manifest:
                self.manifest.register_run(meta)
            return meta

        # --- Quality checks ---
        quality = self._quality_check(arrays)

        # --- Write HDF5 ---
        self._write_hdf5(hdf5_path, run_id, arrays, quality, {
            "c_rate": c_rate,
            "t_amb_K": t_amb_K,
            "soc_init": soc_init,
            "timestamp": timestamp,
        })

        # --- Checksum ---
        checksum = self._sha256(hdf5_path)

        meta = {
            "run_id": run_id,
            "pybamm_version": pybamm.__version__,
            "model_name": self.model_name,
            "thermal_option": self.thermal_option,
            "parameter_set": self.param_set_name,
            "protocol_id": self.protocol_id,
            "solver": self.solver_name,
            "rtol": self.rtol,
            "atol": self.atol,
            "c_rate": c_rate,
            "ambient_temperature_K": t_amb_K,
            "initial_soc": soc_init,
            "timestamp": timestamp,
            "git_commit": self.git_commit,
            "random_seed": self.seed,
            "hdf5_path": str(hdf5_path),
            "checksum_sha256": checksum,
            "solver_status": solver_status,
            "quality_passed": bool(quality["passed"]),
            "n_time_steps": int(arrays["time_s"].shape[0]),
            "n_r_points": int(arrays["r_neg_m"].shape[0]),
        }

        if self.manifest:
            self.manifest.register_run(meta)

        status_str = "PASSED" if quality["passed"] else "FAILED quality"
        logger.info("run_id=%s  %s  n_t=%d", run_id, status_str, meta["n_time_steps"])
        return meta

    # ------------------------------------------------------------------
    # Private: build & solve
    # ------------------------------------------------------------------

    def _build_and_solve(
        self, c_rate: float, t_amb_K: float, soc_init: float
    ):
        """
        Construct and solve the PyBaMM SPMe + lumped thermal model.

        Parameters
        ----------
        c_rate  : float — C-rate (positive = discharge)
        t_amb_K : float — ambient temperature [K]
        soc_init: float — initial state of charge [0–1]

        Returns
        -------
        (sol, param)
        """
        # Build model
        model_options = dict(self.pybamm_cfg["model_options"])  # shallow copy
        model = pybamm.lithium_ion.SPMe(options=model_options)

        # Parameter set
        param = pybamm.ParameterValues(self.param_set_name)

        # Override ambient temperature
        param["Ambient temperature [K]"] = t_amb_K

        # Build solver
        solver = pybamm.CasadiSolver(
            mode=self.pybamm_cfg["solver_options"]["mode"],
            rtol=self.rtol,
            atol=self.atol,
        )

        # Discharge experiment: CC to cutoff
        # Rate string: e.g., "1C" or "0.5C" or "2C"
        c_rate_str = f"{c_rate}C"
        experiment = pybamm.Experiment(
            [f"Discharge at {c_rate_str} until {self.v_cutoff:.1f}V"]
        )

        sim = pybamm.Simulation(
            model,
            parameter_values=param,
            experiment=experiment,
            solver=solver,
        )

        # Solve — PyBaMM uses its own time stepping internally
        sol = sim.solve(initial_soc=soc_init)

        return sol, param

    # ------------------------------------------------------------------
    # Private: array extraction
    # ------------------------------------------------------------------

    def _extract_arrays(
        self,
        sol,
        param,
        c_rate: float,
        t_amb_K: float,
        soc_init: float,
    ) -> dict[str, np.ndarray]:
        """
        Extract SI-unit arrays from the PyBaMM solution object.

        Returns
        -------
        dict with keys:
          time_s, r_neg_m, current_A, ambient_temperature_K (scalar),
          initial_soc (scalar), voltage_V, concentration_neg_mol_m3,
          temperature_K
        """
        var_cfg = self.pybamm_cfg["output_variables"]
        n_r = self.n_r

        # -- Time (SI: seconds) --
        time_s: np.ndarray = sol["Time [s]"].entries.astype(np.float64)
        n_t = len(time_s)

        # -- Voltage [V] --
        voltage_V: np.ndarray = sol[var_cfg["voltage"]].entries.astype(np.float64)

        # -- Temperature [K] --
        # We use "Volume-averaged cell temperature [K]" which returns shape (1, n_t)
        # or (n_t,). "Cell temperature [K]" returns (n_x, n_t) spatial distribution.
        temperature_raw = sol[var_cfg["temperature"]].entries.astype(np.float64)
        temperature_K = np.squeeze(temperature_raw)  # squeeze out any leading dim-1 axes
        # After squeeze: if scalar, broadcast to (n_t,); if (n_t,) already, fine
        if temperature_K.ndim == 0:
            temperature_K = np.full(n_t, float(temperature_K))
        elif temperature_K.ndim > 1:
            # Safety: take spatial mean over all axes except the time axis
            temperature_K = temperature_K.mean(axis=tuple(range(temperature_K.ndim - 1)))
        # Ensure shape (n_t,)
        if temperature_K.shape[0] != n_t:
            temperature_K = np.interp(
                np.linspace(0, 1, n_t),
                np.linspace(0, 1, temperature_K.shape[0]),
                temperature_K,
            )

        # -- Negative particle concentration [mol/m³] --
        # PyBaMM 26.x: "X-averaged negative particle concentration [mol.m-3]"
        # IMPORTANT: conc_var(r=...) expects PHYSICAL radius in METERS, NOT dimensionless.
        #   conc_var(r=dimensionless) → NaN (out of internal domain)
        #   conc_var(r=physical_m)   → correct values
        #
        # The X-averaged SPMe concentration is spatially uniform in r at any given time.
        # (Verified empirically: conc_var(r=r_physical, t=t) returns same value for all r)
        # The radial variation is enforced ONLY through the PINN diffusion residual.
        # We store a (n_t, n_r) grid where all columns are identical — this provides
        # the correct IC/BC signal for the PINN to learn from.
        R_s_neg = float(param["Negative particle radius [m]"])
        r_star = np.linspace(0.0, 1.0, n_r)        # dimensionless [0, 1]
        r_neg_m: np.ndarray = (r_star * R_s_neg).astype(np.float64)  # SI [m]

        conc_var = sol[var_cfg["concentration_neg"]]
        try:
            # Use physical radius in metres — conc_var(r=r_phys, t=t) → (n_r, n_t)
            conc_grid = conc_var(r=r_neg_m, t=time_s)
            if conc_grid.ndim == 2:
                if conc_grid.shape == (n_r, n_t):
                    conc_grid = conc_grid.T        # → (n_t, n_r)
                elif conc_grid.shape == (n_t, n_r):
                    pass                           # already correct
                else:
                    raise ValueError(f"Unexpected concentration shape: {conc_grid.shape}")
            else:
                raise ValueError(f"conc_var returned ndim={conc_grid.ndim}")
            # Sanity: if NaN slipped through, fall back
            if not np.all(np.isfinite(conc_grid)):
                raise ValueError("NaN/Inf in concentration after conc_var(r_phys, t)")
        except Exception as exc:
            logger.warning("conc_var(r_phys, t) failed (%s); using .entries fallback", exc)
            # Fallback: .entries shape (n_r_internal, n_t); broadcast over r uniformly
            raw = conc_var.entries.astype(np.float64)  # (n_r_internal, n_t)
            if raw.ndim == 2:
                # X-averaged: uniform in r — broadcast the mean over n_r grid
                conc_1d = raw.mean(axis=0)  # (n_t,) — mean over internal r points
                conc_grid = np.tile(conc_1d[:, np.newaxis], (1, n_r))  # (n_t, n_r)
            elif raw.ndim == 1:
                conc_grid = np.tile(raw[:, np.newaxis], (1, n_r))
            else:
                conc_grid = np.zeros((n_t, n_r))

        concentration_neg_mol_m3 = conc_grid.astype(np.float64)

        # -- Current [A] (positive = discharge) --
        current_A: np.ndarray = sol["Current [A]"].entries.astype(np.float64)

        return {
            "time_s": time_s,
            "r_neg_m": r_neg_m,
            "current_A": current_A,
            "ambient_temperature_K": np.float64(t_amb_K),
            "initial_soc": np.float64(soc_init),
            "voltage_V": voltage_V,
            "concentration_neg_mol_m3": concentration_neg_mol_m3,
            "temperature_K": temperature_K,
        }

    # ------------------------------------------------------------------
    # Private: quality checks
    # ------------------------------------------------------------------

    def _quality_check(self, arrays: dict) -> dict:
        """
        Run all quality checks. Returns a dict with boolean flags and
        an overall 'passed' key.
        """
        qcfg = self.quality_cfg
        q: dict[str, Any] = {}

        # Finite check
        q["all_finite"] = all(
            np.all(np.isfinite(v))
            for v in arrays.values()
            if isinstance(v, np.ndarray)
        )

        # Voltage bounds
        v = arrays["voltage_V"]
        q["voltage_in_bounds"] = bool(
            np.all(v >= qcfg["voltage_min_V"]) and np.all(v <= qcfg["voltage_max_V"])
        )

        # Temperature bounds
        t = arrays["temperature_K"]
        q["temperature_in_bounds"] = bool(
            np.all(t >= qcfg["temperature_min_K"])
            and np.all(t <= qcfg["temperature_max_K"])
        )

        # Concentration bounds
        c = arrays["concentration_neg_mol_m3"]
        q["concentration_in_bounds"] = bool(
            np.all(c >= qcfg["concentration_min_mol_m3"])
            and np.all(c <= qcfg["concentration_max_mol_m3"])
        )

        # Monotonic time
        time_s = arrays["time_s"]
        q["monotonic_time"] = bool(np.all(np.diff(time_s) > 0))

        # Monotonic r
        r = arrays["r_neg_m"]
        q["monotonic_r"] = bool(np.all(np.diff(r) > 0))

        # Shapes consistent
        n_t = len(time_s)
        n_r = len(r)
        q["shape_consistent"] = bool(
            arrays["voltage_V"].shape == (n_t,)
            and arrays["temperature_K"].shape == (n_t,)
            and arrays["concentration_neg_mol_m3"].shape == (n_t, n_r)
            and arrays["current_A"].shape == (n_t,)
        )

        q["passed"] = all(
            [
                q["all_finite"],
                q["voltage_in_bounds"],
                q["temperature_in_bounds"],
                q["concentration_in_bounds"],
                q["monotonic_time"],
                q["monotonic_r"],
                q["shape_consistent"],
            ]
        )
        return q

    # ------------------------------------------------------------------
    # Private: HDF5 export
    # ------------------------------------------------------------------

    def _write_hdf5(
        self,
        hdf5_path: Path,
        run_id: str,
        arrays: dict,
        quality: dict,
        run_info: dict,
    ) -> None:
        """
        Write all data to HDF5 with groups: coordinates, inputs, outputs, quality.
        Metadata stored as root-level attributes.
        """
        with h5py.File(hdf5_path, "w") as f:
            # ---------- coordinates ----------
            coords = f.create_group("coordinates")
            ds = coords.create_dataset(
                "time_s", data=arrays["time_s"], dtype="float64"
            )
            ds.attrs["units"] = "s"
            ds.attrs["description"] = "Simulation time in seconds (SI)"

            ds = coords.create_dataset(
                "r_neg_m", data=arrays["r_neg_m"], dtype="float64"
            )
            ds.attrs["units"] = "m"
            ds.attrs["description"] = (
                "Negative electrode particle radial coordinate (SI), "
                "uniformly sampled from 0 to R_s_neg"
            )

            # ---------- inputs ----------
            inp = f.create_group("inputs")
            ds = inp.create_dataset(
                "current_A", data=arrays["current_A"], dtype="float64"
            )
            ds.attrs["units"] = "A"
            ds.attrs["description"] = (
                "Applied current in Amperes (positive = discharge)"
            )
            ds.attrs["shape_note"] = "shape (n_time,)"

            ds = inp.create_dataset(
                "ambient_temperature_K",
                data=arrays["ambient_temperature_K"],
                dtype="float64",
            )
            ds.attrs["units"] = "K"
            ds.attrs["description"] = "Ambient temperature (constant per run)"

            ds = inp.create_dataset(
                "initial_soc",
                data=arrays["initial_soc"],
                dtype="float64",
            )
            ds.attrs["units"] = "dimensionless"
            ds.attrs["description"] = "Initial state of charge [0, 1]"

            # ---------- outputs ----------
            out = f.create_group("outputs")
            ds = out.create_dataset(
                "voltage_V", data=arrays["voltage_V"], dtype="float64"
            )
            ds.attrs["units"] = "V"
            ds.attrs["description"] = "Terminal voltage [V]"
            ds.attrs["shape_note"] = "shape (n_time,)"

            ds = out.create_dataset(
                "concentration_neg_mol_m3",
                data=arrays["concentration_neg_mol_m3"],
                dtype="float64",
            )
            ds.attrs["units"] = "mol.m-3"
            ds.attrs["description"] = (
                "X-averaged negative electrode particle lithium concentration. "
                "Stage I: SPMe x-averaged value broadcast over radial grid. "
                "True radial profile requires full DFN (Stage II)."
            )
            ds.attrs["shape_note"] = "shape (n_time, n_r)"

            ds = out.create_dataset(
                "temperature_K", data=arrays["temperature_K"], dtype="float64"
            )
            ds.attrs["units"] = "K"
            ds.attrs["description"] = (
                "Lumped cell temperature [K] from SPMe+lumped thermal model. "
                "This is the spatially-averaged temperature T(t), not T(x,t)."
            )
            ds.attrs["shape_note"] = "shape (n_time,)"

            # ---------- quality ----------
            qgrp = f.create_group("quality")
            for key, val in quality.items():
                qgrp.attrs[key] = int(val) if isinstance(val, bool) else val

            # ---------- metadata attributes ----------
            f.attrs["run_id"] = run_id
            f.attrs["pybamm_version"] = pybamm.__version__
            f.attrs["model_name"] = self.model_name
            f.attrs["thermal_option"] = self.thermal_option
            f.attrs["parameter_set"] = self.param_set_name
            f.attrs["protocol_id"] = self.protocol_id
            f.attrs["solver"] = self.solver_name
            f.attrs["rtol"] = self.rtol
            f.attrs["atol"] = self.atol
            f.attrs["c_rate"] = run_info["c_rate"]
            f.attrs["ambient_temperature_K"] = run_info["t_amb_K"]
            f.attrs["initial_soc"] = run_info["soc_init"]
            f.attrs["timestamp"] = run_info["timestamp"]
            f.attrs["git_commit"] = self.git_commit
            f.attrs["random_seed"] = self.seed
            f.attrs["config_version"] = self.cfg.get("config_version", "unknown")
            f.attrs["stage"] = int(self.cfg.get("stage", 1))
            # Schema version for forward compatibility
            f.attrs["schema_version"] = "1.0"

        logger.debug("Wrote HDF5: %s", hdf5_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_failed_meta(
        self,
        run_id: str,
        c_rate: float,
        t_amb_K: float,
        soc_init: float,
        timestamp: str,
        error_msg: str,
    ) -> dict:
        return {
            "run_id": run_id,
            "pybamm_version": pybamm.__version__,
            "model_name": self.model_name,
            "thermal_option": self.thermal_option,
            "parameter_set": self.param_set_name,
            "protocol_id": self.protocol_id,
            "solver": self.solver_name,
            "rtol": self.rtol,
            "atol": self.atol,
            "c_rate": c_rate,
            "ambient_temperature_K": t_amb_K,
            "initial_soc": soc_init,
            "timestamp": timestamp,
            "git_commit": self.git_commit,
            "random_seed": self.seed,
            "hdf5_path": None,
            "checksum_sha256": None,
            "solver_status": f"failed: {error_msg}",
            "quality_passed": False,
            "n_time_steps": 0,
            "n_r_points": 0,
        }

    @staticmethod
    def _sha256(path: Path) -> str:
        """Compute SHA-256 checksum of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _get_git_commit() -> str:
        """Attempt to read the current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return "unknown"
