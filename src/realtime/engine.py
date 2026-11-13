"""Checkpoint-backed inference for the live digital twin."""
from __future__ import annotations

import glob
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.paper.dataset import load_config
from src.paper.experiment import make_model
from src.paper.stress import spherical_particle_stress


class RealtimeInferenceEngine:
    def __init__(self, root: Path, digital_config: dict):
        self.root = root
        model_cfg = digital_config["model"]
        self.paper_config = load_config(root / model_cfg["experiment_config"])
        dataset = pd.read_parquet(root / self.paper_config["paths"]["dataset_parquet"], columns=["time_s", "concentration_star"])
        self.c0_star = float(dataset.loc[dataset.time_s == 0, "concentration_star"].mean())
        self.checkpoint_path = self._choose_checkpoint(model_cfg)
        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.model = make_model("pc_mp_pinn", self.c0_star, self.paper_config)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.summary = payload.get("summary", {})
        self.radius = torch.linspace(0.0, 1.0, int(digital_config["simulation"]["radial_points"]))

    def _choose_checkpoint(self, model_cfg: dict) -> Path:
        deployment = self.root / model_cfg.get("deployment_checkpoint", "")
        if model_cfg.get("deployment_checkpoint") and deployment.exists():
            return deployment
        candidates = [Path(item) for item in glob.glob(str(self.root / model_cfg["checkpoint_glob"]))]
        if not candidates:
            raise FileNotFoundError("No PC-MP-PINN checkpoint found. Run the experiment pipeline first.")
        preferred = [path for path in candidates if model_cfg.get("prefer_mode", "full") in path.stem]
        pool = preferred or candidates
        def validation_loss(path: Path) -> float:
            metadata = path.with_suffix(".json")
            if not metadata.exists(): return float("inf")
            return float(json.loads(metadata.read_text(encoding="utf-8")).get("best_validation_loss", float("inf")))
        return min(pool, key=validation_loss)

    def predict(self, elapsed_s: float, c_rate: float, ambient_c: float) -> dict:
        start = time.perf_counter()
        horizon = float(self.paper_config["simulation"]["fixed_horizon_s"])
        tau = float(np.clip(elapsed_s / horizon, 0.0, 1.0))
        x = torch.column_stack([
            self.radius,
            torch.full_like(self.radius, tau),
            torch.full_like(self.radius, c_rate / 2.0),
            torch.full_like(self.radius, ambient_c / 60.0),
        ])
        with torch.inference_mode():
            output = self.model(x)
            concentration = output["c"].reshape(1, -1) * float(self.paper_config["physics"]["max_concentration_mol_m3"])
            radial, tangential = spherical_particle_stress(
                concentration,
                self.radius,
                float(self.paper_config["physics"]["youngs_modulus_Pa"]),
                float(self.paper_config["physics"]["poissons_ratio"]),
                float(self.paper_config["physics"]["partial_molar_volume_m3_mol"]),
            )
            temperature_c = float(output["T"][0].item() * 60.0)
        c_np = concentration.numpy().reshape(-1)
        sr_np = radial.numpy().reshape(-1) / 1e6
        st_np = tangential.numpy().reshape(-1) / 1e6
        peak_index = int(np.argmax(st_np))
        confidence, reasons = self._domain_confidence(elapsed_s, c_rate, ambient_c)
        return {
            "temperature_C": temperature_c,
            "radius_star": self.radius.numpy().round(6).tolist(),
            "concentration_mol_m3": c_np.round(3).tolist(),
            "sigma_radial_MPa": sr_np.round(4).tolist(),
            "sigma_tangential_MPa": st_np.round(4).tolist(),
            "peak_tensile_stress_MPa": float(st_np[peak_index]),
            "peak_stress_radius_star": float(self.radius[peak_index]),
            "surface_concentration_mol_m3": float(c_np[-1]),
            "core_concentration_mol_m3": float(c_np[0]),
            "confidence": confidence,
            "domain_reasons": reasons,
            "inference_ms": (time.perf_counter() - start) * 1000.0,
            "checkpoint": self.checkpoint_path.name,
            "training_mode": self.summary.get("mode", "unknown"),
        }

    @staticmethod
    def _domain_confidence(elapsed_s: float, c_rate: float, ambient_c: float) -> tuple[float, list[str]]:
        score, reasons = 1.0, []
        if not 0.5 <= c_rate <= 2.0:
            score -= min(0.45, abs(c_rate - np.clip(c_rate, 0.5, 2.0)) * 0.35)
            reasons.append("C-rate is outside the trained range 0.5C–2.0C")
        if not 10.0 <= ambient_c <= 35.0:
            score -= min(0.45, abs(ambient_c - np.clip(ambient_c, 10.0, 35.0)) * 0.035)
            reasons.append("Ambient temperature is outside the trained range 10–35 °C")
        if elapsed_s > 7200.0:
            score -= 0.4
            reasons.append("Elapsed time exceeds the trained horizon")
        return round(max(0.05, score), 3), reasons
