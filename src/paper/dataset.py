"""Generate the exact 18-condition PyBaMM reference dataset from the paper."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pybamm
import yaml


@dataclass(frozen=True)
class Condition:
    condition_id: str
    split: str
    c_rate: float
    ambient_c: float


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def paper_conditions(config: dict) -> list[Condition]:
    spec = config["simulation"]["conditions"]
    conditions: list[Condition] = []
    index = 1
    for c_rate in spec["train"]["c_rates"]:
        for ambient_c in spec["train"]["ambient_C"]:
            conditions.append(Condition(f"C{index:02d}", "train", float(c_rate), float(ambient_c)))
            index += 1
    for split in ("validation", "test"):
        for c_rate, ambient_c in spec[split]["pairs"]:
            conditions.append(Condition(f"C{index:02d}", split, float(c_rate), float(ambient_c)))
            index += 1
    return conditions


def _numpy_stress(c: np.ndarray, rho: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    physics = cfg["physics"]
    shifted = c - c[:, :1]
    integral = np.trapezoid(shifted * rho[None, :] ** 2, rho, axis=1)
    shell = 0.5 * (
        shifted[:, 1:] * rho[None, 1:] ** 2 + shifted[:, :-1] * rho[None, :-1] ** 2
    ) * np.diff(rho)[None, :]
    cumulative = np.concatenate([np.zeros((len(c), 1)), np.cumsum(shell, axis=1)], axis=1)
    local_avg = np.divide(
        3.0 * cumulative,
        rho[None, :] ** 3,
        out=shifted.copy(),
        where=rho[None, :] > 0,
    )
    local_avg[:, 0] = shifted[:, 0]
    average = 3.0 * integral[:, None]
    factor = (
        float(physics["youngs_modulus_Pa"])
        * float(physics["partial_molar_volume_m3_mol"])
        / (3.0 * (1.0 - float(physics["poissons_ratio"])))
    )
    radial = 2.0 * factor * (average - local_avg)
    tangential = factor * (2.0 * average + local_avg - 3.0 * shifted)
    radial[:, -1] = 0.0
    return radial, tangential


def simulate_condition(condition: Condition, config: dict) -> pd.DataFrame:
    sim_cfg = config["simulation"]
    physics = config["physics"]
    ambient_k = condition.ambient_c + 273.15
    model = pybamm.lithium_ion.SPMe({"thermal": sim_cfg["thermal"]})
    parameters = pybamm.ParameterValues(sim_cfg["parameter_set"])
    parameters.update({
        "Ambient temperature [K]": ambient_k,
        "Initial temperature [K]": ambient_k,
        "Current function [A]": condition.c_rate * parameters["Nominal cell capacity [A.h]"],
    })
    solver = pybamm.CasadiSolver(mode="safe", rtol=1e-6, atol=1e-6)
    simulation = pybamm.Simulation(model, parameter_values=parameters, solver=solver)
    solution = simulation.solve(
        [0.0, float(sim_cfg["fixed_horizon_s"])],
        initial_soc=float(sim_cfg["initial_soc"]),
    )
    end_time = float(solution.t[-1])
    time_s = np.linspace(0.0, end_time, int(sim_cfg["time_points"]))
    radius_m = float(physics["particle_radius_m"])
    rho = np.linspace(0.0, 1.0, int(sim_cfg["radial_points"]))
    radius = rho * radius_m
    concentration = solution["X-averaged negative particle concentration [mol.m-3]"](
        t=time_s, r=radius
    ).T
    temperature = np.asarray(solution["Volume-averaged cell temperature [K]"](t=time_s)).reshape(-1)
    voltage = np.asarray(solution["Voltage [V]"](t=time_s)).reshape(-1)
    heat = np.asarray(solution["Volume-averaged total heating [W.m-3]"](t=time_s)).reshape(-1)
    interfacial_current = np.asarray(
        solution["X-averaged negative electrode interfacial current density [A.m-2]"](t=time_s)
    ).reshape(-1)
    sigma_r, sigma_t = _numpy_stress(concentration, rho, config)
    cmax = float(physics["max_concentration_mol_m3"])
    gradient_star = np.gradient(concentration / cmax, rho, axis=1)[:, -1]
    rows = []
    for ti, time_value in enumerate(time_s):
        for ri, radial_value in enumerate(rho):
            rows.append({
                "condition_id": condition.condition_id,
                "split": condition.split,
                "c_rate": condition.c_rate,
                "ambient_C": condition.ambient_c,
                "ambient_K": ambient_k,
                "time_s": float(time_value),
                "time_star": float(time_value / float(sim_cfg["fixed_horizon_s"])),
                "radius_star": float(radial_value),
                "concentration_mol_m3": float(concentration[ti, ri]),
                "concentration_star": float(concentration[ti, ri] / cmax),
                "temperature_K": float(temperature[ti]),
                "temperature_star": float((temperature[ti] - 273.15) / 60.0),
                "voltage_V": float(voltage[ti]),
                "heat_W_m3": float(heat[ti]),
                "interfacial_current_A_m2": float(interfacial_current[ti]),
                "surface_gradient_star": float(gradient_star[ti]),
                "sigma_radial_Pa": float(sigma_r[ti, ri]),
                "sigma_tangential_Pa": float(sigma_t[ti, ri]),
                "sigma_radial_MPa": float(sigma_r[ti, ri] / 1e6),
                "sigma_tangential_MPa": float(sigma_t[ti, ri] / 1e6),
            })
    return pd.DataFrame(rows)


def write_database(dataset: pd.DataFrame, config: dict, database_path: Path) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        dataset.to_sql("samples", connection, if_exists="replace", index=False, chunksize=5000)
        conditions = (
            dataset.groupby(["condition_id", "split", "c_rate", "ambient_C", "ambient_K"], as_index=False)
            .agg(duration_s=("time_s", "max"), sample_count=("time_s", "size"))
        )
        conditions.to_sql("conditions", connection, if_exists="replace", index=False)
        connection.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_samples_condition ON samples(condition_id);
            CREATE INDEX IF NOT EXISTS idx_samples_split ON samples(split);
            CREATE INDEX IF NOT EXISTS idx_samples_time_radius ON samples(time_s, radius_star);
            CREATE TABLE IF NOT EXISTS experiment_runs (
              run_id TEXT PRIMARY KEY, variant TEXT NOT NULL, label_budget INTEGER NOT NULL,
              seed INTEGER NOT NULL, mode TEXT NOT NULL, checkpoint_path TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS metrics (
              run_id TEXT NOT NULL, split TEXT NOT NULL, metric TEXT NOT NULL, value REAL,
              unit TEXT, FOREIGN KEY(run_id) REFERENCES experiment_runs(run_id)
            );
            """
        )


def generate_dataset(config_path: str | Path, force: bool = False) -> pd.DataFrame:
    config = load_config(config_path)
    parquet_path = Path(config["paths"]["dataset_parquet"])
    if parquet_path.exists() and not force:
        return pd.read_parquet(parquet_path)
    frames = []
    for condition in paper_conditions(config):
        print(f"Simulating {condition.condition_id}: {condition.split} {condition.c_rate}C {condition.ambient_c} C")
        frames.append(simulate_condition(condition, config))
    dataset = pd.concat(frames, ignore_index=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(parquet_path, index=False)
    write_database(dataset, config, Path(config["paths"]["database"]))
    metadata = {
        "rows": int(len(dataset)),
        "conditions": int(dataset["condition_id"].nunique()),
        "split_rows": dataset.groupby("split").size().astype(int).to_dict(),
        "pybamm_version": pybamm.__version__,
        "model": "SPMe with lumped thermal model",
        "parameter_set": config["simulation"]["parameter_set"],
        "stress": "Analytical Cheng-Verbrugge spherical-particle operator",
    }
    metadata_path = Path(config["paths"]["metadata"])
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return dataset
