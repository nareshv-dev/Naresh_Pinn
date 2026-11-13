"""Train a deployment checkpoint for smooth real-time digital-twin inference.

This does not replace the 27-run paper comparison. It trains the same proposed
architecture on all training-condition labels, with a small periodic physics
penalty, for use in the software demonstration.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paper.dataset import load_config
from src.paper.experiment import make_inputs, make_model, seed_everything
from src.paper.physics import center_boundary_residual, diffusion_residual, surface_boundary_residual, thermal_residual


def main() -> None:
    config = load_config(ROOT / "configs/paper_experiment.yaml")
    data = pd.read_parquet(ROOT / config["paths"]["dataset_parquet"])
    train = data[data.split == "train"].reset_index(drop=True)
    validation = data[data.split == "validation"].reset_index(drop=True)
    seed_everything(2026)
    c0 = float(train.loc[train.time_s == 0, "concentration_star"].mean())
    model = make_model("pc_mp_pinn", c0, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-4)
    rng = np.random.default_rng(2026)
    best_loss, best_state, stale = float("inf"), None, 0
    history = []
    start = time.perf_counter()
    for step in range(1, 2001):
        index = rng.integers(0, len(train), size=1536)
        batch = train.iloc[index]
        x = make_inputs(batch)
        output = model(x)
        c = torch.tensor(batch.concentration_star.to_numpy(), dtype=torch.float32).unsqueeze(1)
        temperature = torch.tensor(batch.temperature_star.to_numpy(), dtype=torch.float32).unsqueeze(1)
        loss_c = torch.mean((output["c"] - c) ** 2)
        loss_t = torch.mean((output["T"] - temperature) ** 2)
        loss = loss_c + 2.0 * loss_t
        if step % 20 == 0:
            physics_frame = train[(train.radius_star > .001) & (train.radius_star < .999)].sample(96, random_state=step)
            xp = make_inputs(physics_frame)
            diffusion = diffusion_residual(model, xp.clone(), float(config["physics"]["diffusivity_m2_s"]), float(config["simulation"]["fixed_horizon_s"]), float(config["physics"]["particle_radius_m"])).square().mean()
            heat = torch.tensor(physics_frame.heat_W_m3.to_numpy() * float(config["simulation"]["fixed_horizon_s"]) / (60.0 * float(config["physics"]["volumetric_heat_capacity_J_m3_K"])), dtype=torch.float32).unsqueeze(1)
            cooling = float(config["simulation"]["fixed_horizon_s"]) * float(config["physics"]["cooling_W_m3_K"]) / float(config["physics"]["volumetric_heat_capacity_J_m3_K"])
            thermal = thermal_residual(model, xp.clone(), heat, cooling).square().mean()
            boundary_frame = train.sample(96, random_state=step + 1)
            xb = make_inputs(boundary_frame)
            center = center_boundary_residual(model, xb.clone()).square().mean()
            target = torch.tensor(boundary_frame.surface_gradient_star.to_numpy(), dtype=torch.float32).unsqueeze(1)
            surface = surface_boundary_residual(model, xb.clone(), target).square().mean()
            loss = loss + 0.001 * diffusion + 0.002 * thermal + 0.005 * center + 0.005 * surface
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); optimizer.step()
        if step % 25 == 0:
            sample = validation.sample(min(4000, len(validation)), random_state=2026)
            with torch.inference_mode():
                predicted = model(make_inputs(sample))
                c_true = torch.tensor(sample.concentration_star.to_numpy(), dtype=torch.float32).unsqueeze(1)
                t_true = torch.tensor(sample.temperature_star.to_numpy(), dtype=torch.float32).unsqueeze(1)
                val_loss = float(torch.mean((predicted["c"] - c_true) ** 2) + 2.0 * torch.mean((predicted["T"] - t_true) ** 2))
            history.append({"step": step, "training_loss": float(loss.detach()), "validation_loss": val_loss})
            if val_loss < best_loss:
                best_loss = val_loss; stale = 0; best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            else:
                stale += 25
                if stale >= 350: break
    model.load_state_dict(best_state)
    with torch.inference_mode():
        predicted = model(make_inputs(validation))
    c_true = validation.concentration_star.to_numpy() * float(config["physics"]["max_concentration_mol_m3"])
    c_pred = predicted["c"].numpy().reshape(-1) * float(config["physics"]["max_concentration_mol_m3"])
    t_true = validation.temperature_K.to_numpy()
    t_pred = predicted["T"].numpy().reshape(-1) * 60.0 + 273.15
    metrics = {"concentration_relative_l2": float(np.linalg.norm(c_pred-c_true)/np.linalg.norm(c_true)), "temperature_rmse_K": float(np.sqrt(np.mean((t_pred-t_true)**2)))}
    output_dir = ROOT / "outputs/deployment"; output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"mode":"deployment_finetune","steps_completed":history[-1]["step"],"best_validation_loss":best_loss,"seconds":time.perf_counter()-start,"metrics":metrics,"history":history}
    torch.save({"state_dict":model.state_dict(),"summary":summary,"config":config}, output_dir / "pc_mp_pinn_live.pt")
    (output_dir / "pc_mp_pinn_live.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary | {"history":"omitted"},indent=2))


if __name__ == "__main__": main()
