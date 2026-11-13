"""Training, evaluation and artifact generation for the paper comparison."""
from __future__ import annotations

import json
import random
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .dataset import load_config, simulate_condition, paper_conditions
from .models import DataOnlyModel, PCMPPINN, PriorPINN
from .physics import (
    center_boundary_residual,
    diffusion_residual,
    mechanical_equilibrium_residual,
    surface_boundary_residual,
    thermal_residual,
)
from .stress import spherical_particle_stress


STRESS_SCALE_PA = 1.0e8


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)


def make_inputs(frame: pd.DataFrame) -> torch.Tensor:
    return torch.tensor(
        np.column_stack([
            frame["radius_star"].to_numpy(),
            frame["time_star"].to_numpy(),
            frame["c_rate"].to_numpy() / 2.0,
            frame["ambient_C"].to_numpy() / 60.0,
        ]), dtype=torch.float32,
    )


def make_model(variant: str, c0_star: float, config: dict) -> nn.Module:
    training = config["training"]
    kwargs = {"c0_star": c0_star, "width": int(training["hidden_width"]), "depth": int(training["hidden_layers"])}
    return {"data_only": DataOnlyModel, "prior_pinn": PriorPINN, "pc_mp_pinn": PCMPPINN}[variant](**kwargs)


def _sample_labels(train: pd.DataFrame, budget: int, seed: int) -> pd.DataFrame:
    return (
        train.groupby("condition_id", group_keys=False)
        .apply(lambda group: group.sample(n=min(budget, len(group)), random_state=seed), include_groups=False)
        .reset_index()
    )


def _data_loss(model: nn.Module, frame: pd.DataFrame, device: torch.device) -> torch.Tensor:
    x = make_inputs(frame).to(device)
    output = model(x)
    c = torch.tensor(frame["concentration_star"].to_numpy(), dtype=torch.float32, device=device).unsqueeze(1)
    t = torch.tensor(frame["temperature_star"].to_numpy(), dtype=torch.float32, device=device).unsqueeze(1)
    loss = nn.functional.mse_loss(output["c"], c) + nn.functional.mse_loss(output["T"], t)
    if "sigma_t" in output:
        sr = torch.tensor(frame["sigma_radial_Pa"].to_numpy() / STRESS_SCALE_PA, dtype=torch.float32, device=device).unsqueeze(1)
        st = torch.tensor(frame["sigma_tangential_Pa"].to_numpy() / STRESS_SCALE_PA, dtype=torch.float32, device=device).unsqueeze(1)
        loss = loss + 0.25 * (nn.functional.mse_loss(output["sigma_r"], sr) + nn.functional.mse_loss(output["sigma_t"], st))
    return loss


def train_one(
    dataset: pd.DataFrame,
    config: dict,
    variant: str,
    budget: int,
    seed: int,
    quick: bool,
) -> tuple[nn.Module, dict]:
    seed_everything(seed, bool(config["reproducibility"]["deterministic"]))
    device = torch.device("cpu")
    train = dataset[dataset.split == "train"].copy()
    validation = dataset[dataset.split == "validation"].copy()
    labels = _sample_labels(train, budget, seed)
    interior_candidates = train[(train.radius_star > 0.001) & (train.radius_star < 0.999)]
    interior_pool = (
        interior_candidates.groupby("condition_id", group_keys=False)
        .apply(
            lambda group: group.sample(
                n=min(int(config["training"]["interior_points_per_condition"]), len(group)),
                random_state=seed + 101,
            ),
            include_groups=False,
        )
        .reset_index()
    )
    boundary_pool = (
        train.groupby("condition_id", group_keys=False)
        .apply(
            lambda group: group.sample(
                n=min(int(config["training"]["boundary_points_per_condition"]), len(group)),
                random_state=seed + 202,
            ),
            include_groups=False,
        )
        .reset_index()
    )
    c0_star = float(train.loc[train.time_s == 0, "concentration_star"].mean())
    model = make_model(variant, c0_star, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    quick_cfg = config["training"]["quick"]
    epochs = int(quick_cfg["max_epochs"] if quick else config["training"]["max_epochs"])
    patience = int(quick_cfg["patience"] if quick else config["training"]["patience"])
    physics_batch = int(quick_cfg["physics_batch"] if quick else 512)
    weights = config["training"]["loss_weights"]
    best_loss, best_state, stale = float("inf"), None, 0
    history: list[dict] = []
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        batch = labels.sample(n=min(512, len(labels)), random_state=seed + epoch)
        optimizer.zero_grad(set_to_none=True)
        loss = _data_loss(model, batch, device)
        terms = {"data": float(loss.detach())}
        if variant != "data_only":
            physics_frame = interior_pool.sample(n=min(physics_batch, len(interior_pool)), random_state=seed * 1000 + epoch)
            boundary_frame = boundary_pool.sample(n=min(physics_batch, len(boundary_pool)), random_state=seed * 2000 + epoch)
            x = make_inputs(physics_frame).to(device)
            diff = diffusion_residual(
                model, x.clone(), float(config["physics"]["diffusivity_m2_s"]),
                float(config["simulation"]["fixed_horizon_s"]), float(config["physics"]["particle_radius_m"]),
            ).square().mean()
            heat_star = torch.tensor(
                physics_frame["heat_W_m3"].to_numpy()
                * float(config["simulation"]["fixed_horizon_s"])
                / (60.0 * float(config["physics"]["volumetric_heat_capacity_J_m3_K"])),
                dtype=torch.float32, device=device,
            ).unsqueeze(1)
            cooling_star = (
                float(config["simulation"]["fixed_horizon_s"])
                * float(config["physics"]["cooling_W_m3_K"])
                / float(config["physics"]["volumetric_heat_capacity_J_m3_K"])
            )
            therm = thermal_residual(model, x.clone(), heat_star, cooling_star).square().mean()
            x_boundary = make_inputs(boundary_frame).to(device)
            center = center_boundary_residual(model, x_boundary.clone()).square().mean()
            target_grad = torch.tensor(boundary_frame["surface_gradient_star"].to_numpy(), dtype=torch.float32, device=device).unsqueeze(1)
            surface = surface_boundary_residual(model, x_boundary.clone(), target_grad).square().mean()
            loss = loss + float(weights["diffusion"]) * diff + float(weights["thermal"]) * therm
            loss = loss + float(weights["center_bc"]) * center + float(weights["surface_bc"]) * surface
            terms.update(diffusion=float(diff.detach()), thermal=float(therm.detach()), center=float(center.detach()), surface=float(surface.detach()))
            if variant == "prior_pinn":
                mechanics = mechanical_equilibrium_residual(model, x.clone()).square().mean()
                loss = loss + float(weights["mechanics"]) * mechanics
                terms["mechanics"] = float(mechanics.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip"]))
        optimizer.step()
        if epoch % 5 == 0 or epoch == epochs - 1:
            model.eval()
            val_sample = validation.sample(n=min(1000, len(validation)), random_state=seed)
            with torch.no_grad():
                val_loss = float(_data_loss(model, val_sample, device))
            history.append({"epoch": epoch + 1, "train_loss": float(loss.detach()), "val_loss": val_loss, **terms})
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 5
                if stale >= patience:
                    break
    if best_state:
        model.load_state_dict(best_state)
    return model, {
        "variant": variant, "label_budget": budget, "seed": seed,
        "mode": "quick" if quick else "full", "epochs_completed": history[-1]["epoch"],
        "best_validation_loss": best_loss, "training_seconds": time.perf_counter() - start,
        "history": history,
    }


def predict(model: nn.Module, frame: pd.DataFrame, variant: str, config: dict) -> pd.DataFrame:
    model.eval()
    result = frame[["condition_id", "split", "c_rate", "ambient_C", "time_s", "radius_star"]].copy()
    with torch.no_grad():
        output = model(make_inputs(frame))
    result["c_true"] = frame["concentration_mol_m3"].to_numpy()
    result["c_pred"] = output["c"].numpy().reshape(-1) * float(config["physics"]["max_concentration_mol_m3"])
    result["T_true"] = frame["temperature_K"].to_numpy()
    result["T_pred"] = output["T"].numpy().reshape(-1) * 60.0 + 273.15
    result["sigma_t_true_MPa"] = frame["sigma_tangential_MPa"].to_numpy()
    result["sigma_r_true_MPa"] = frame["sigma_radial_MPa"].to_numpy()
    if variant != "pc_mp_pinn":
        result["sigma_t_pred_MPa"] = output["sigma_t"].numpy().reshape(-1) * STRESS_SCALE_PA / 1e6
        result["sigma_r_pred_MPa"] = output["sigma_r"].numpy().reshape(-1) * STRESS_SCALE_PA / 1e6
    else:
        result["sigma_t_pred_MPa"] = np.nan
        result["sigma_r_pred_MPa"] = np.nan
        radius = torch.tensor(sorted(frame.radius_star.unique()), dtype=torch.float32)
        for (_, _), index in result.groupby(["condition_id", "time_s"]).groups.items():
            ordered = result.loc[index].sort_values("radius_star")
            concentration = torch.tensor(ordered["c_pred"].to_numpy(), dtype=torch.float32).unsqueeze(0)
            radial, tangential = spherical_particle_stress(
                concentration, radius,
                float(config["physics"]["youngs_modulus_Pa"]),
                float(config["physics"]["poissons_ratio"]),
                float(config["physics"]["partial_molar_volume_m3_mol"]),
            )
            result.loc[ordered.index, "sigma_r_pred_MPa"] = radial.numpy().reshape(-1) / 1e6
            result.loc[ordered.index, "sigma_t_pred_MPa"] = tangential.numpy().reshape(-1) / 1e6
    return result


def compute_metrics(prediction: pd.DataFrame, variant: str, config: dict, speedup: float) -> dict[str, float]:
    c_true, c_pred = prediction.c_true.to_numpy(), prediction.c_pred.to_numpy()
    surface = prediction.radius_star == 1.0
    delta_true = prediction.T_true.to_numpy() - (prediction.ambient_C.to_numpy() + 273.15)
    delta_pred = prediction.T_pred.to_numpy() - (prediction.ambient_C.to_numpy() + 273.15)
    stress_error = prediction.sigma_t_pred_MPa.to_numpy() - prediction.sigma_t_true_MPa.to_numpy()
    mass_errors, consistency = [], []
    radius = np.sort(prediction.radius_star.unique())
    factor_args = config["physics"]
    for _, group in prediction.groupby(["condition_id", "time_s"]):
        group = group.sort_values("radius_star")
        true_avg = 3.0 * np.trapezoid(group.c_true.to_numpy() * radius**2, radius)
        pred_avg = 3.0 * np.trapezoid(group.c_pred.to_numpy() * radius**2, radius)
        mass_errors.append(pred_avg - true_avg)
        derived_t = _numpy_stress_for_metric(group.c_pred.to_numpy(), radius, factor_args)
        consistency.extend(group.sigma_t_pred_MPa.to_numpy() - derived_t)
    return {
        "concentration_relative_l2": float(np.linalg.norm(c_pred - c_true) / np.linalg.norm(c_true)),
        "surface_concentration_mae_mol_m3": float(np.mean(np.abs(c_pred[surface] - c_true[surface]))),
        "delta_temperature_rmse_K": float(np.sqrt(np.mean((delta_pred - delta_true) ** 2))),
        "max_delta_temperature_error_K": float(abs(np.max(delta_pred) - np.max(delta_true))),
        "tangential_stress_rmse_MPa": float(np.sqrt(np.mean(stress_error**2))),
        "peak_tensile_stress_error_MPa": float(abs(np.max(prediction.sigma_t_pred_MPa) - np.max(prediction.sigma_t_true_MPa))),
        "mass_balance_rmse_mol_m3": float(np.sqrt(np.mean(np.square(mass_errors)))),
        "concentration_stress_consistency_rmse_MPa": float(np.sqrt(np.mean(np.square(consistency)))),
        "median_speedup_vs_pybamm": float(speedup),
    }


def _numpy_stress_for_metric(c: np.ndarray, rho: np.ndarray, physics: dict) -> np.ndarray:
    shifted = c - c[0]
    integral = np.trapezoid(shifted * rho**2, rho)
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (shifted[1:] * rho[1:]**2 + shifted[:-1] * rho[:-1]**2) * np.diff(rho))])
    local = np.divide(3.0 * cumulative, rho**3, out=shifted.copy(), where=rho > 0)
    local[0] = shifted[0]
    factor = float(physics["youngs_modulus_Pa"]) * float(physics["partial_molar_volume_m3_mol"]) / (3.0 * (1.0 - float(physics["poissons_ratio"])))
    return factor * (6.0 * integral + local - 3.0 * shifted) / 1e6


def run_matrix(config_path: str | Path, quick: bool = True) -> pd.DataFrame:
    config = load_config(config_path)
    dataset = pd.read_parquet(config["paths"]["dataset_parquet"])
    checkpoint_dir = Path(config["paths"]["checkpoints"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # One real PyBaMM benchmark; model latency is measured per condition below.
    benchmark_condition = [c for c in paper_conditions(config) if c.split == "test"][0]
    benchmark_start = time.perf_counter()
    simulate_condition(benchmark_condition, config)
    pybamm_seconds = time.perf_counter() - benchmark_start
    records, prediction_frames = [], []
    database = Path(config["paths"]["database"])
    for variant in config["training"]["variants"]:
        for budget in config["training"]["label_budgets"]:
            for seed in config["reproducibility"]["seeds"]:
                run_id = f"{variant}-b{budget}-s{seed}-{'quick' if quick else 'full'}"
                print(f"Training {run_id}")
                model, summary = train_one(dataset, config, variant, int(budget), int(seed), quick)
                checkpoint_path = checkpoint_dir / f"{run_id}.pt"
                torch.save({"state_dict": model.state_dict(), "summary": summary, "config": config}, checkpoint_path)
                evaluation_frame = dataset[dataset.split.isin(["validation", "test"])].copy()
                latency_start = time.perf_counter()
                prediction = predict(model, evaluation_frame, variant, config)
                model_seconds_per_condition = (time.perf_counter() - latency_start) / evaluation_frame.condition_id.nunique()
                speedup = pybamm_seconds / max(model_seconds_per_condition, 1e-9)
                prediction["run_id"] = run_id
                prediction_frames.append(prediction)
                with sqlite3.connect(database) as connection:
                    connection.execute("DELETE FROM metrics WHERE run_id=?", (run_id,))
                    connection.execute(
                        "INSERT OR REPLACE INTO experiment_runs(run_id,variant,label_budget,seed,mode,checkpoint_path) VALUES(?,?,?,?,?,?)",
                        (run_id, variant, int(budget), int(seed), summary["mode"], str(checkpoint_path)),
                    )
                    for split in ("validation", "test"):
                        metrics = compute_metrics(prediction[prediction.split == split], variant, config, speedup)
                        for metric, value in metrics.items():
                            records.append({"run_id": run_id, "variant": variant, "label_budget": budget, "seed": seed, "mode": summary["mode"], "split": split, "metric": metric, "value": value})
                            connection.execute("INSERT INTO metrics(run_id,split,metric,value,unit) VALUES(?,?,?,?,?)", (run_id, split, metric, value, "see metric name"))
                Path(checkpoint_dir / f"{run_id}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metrics_frame = pd.DataFrame(records)
    metrics_path = Path(config["paths"]["metrics"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(metrics_path, index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(config["paths"]["predictions"], index=False)
    return metrics_frame
