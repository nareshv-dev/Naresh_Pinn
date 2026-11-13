"""Deterministic software battery plant used in place of hardware sensors."""
from __future__ import annotations

import math
import random


class SoftwareBatteryPlant:
    def __init__(self, simulation_config: dict, scenario_name: str, scenario: dict, seed: int = 2026):
        self.config = simulation_config
        self.scenario_name = scenario_name
        self.scenario = scenario
        self.random = random.Random(seed)
        self.time_s = 0.0
        self.soc = float(simulation_config["initial_soc"])
        self.temperature_c = float(scenario["ambient_C"])

    def current_c_rate(self) -> float:
        base = float(self.scenario["base_c_rate"])
        if self.scenario_name == "pulsed":
            phase = int(self.time_s // 180.0) % 3
            return [0.65, 1.85, 1.10][phase]
        if self.scenario_name == "aggressive":
            return min(2.2, base + 0.2 * math.sin(self.time_s / 90.0))
        return base

    def step(self, dt_s: float, c_rate_override: float | None = None, cooling_override: float | None = None) -> dict:
        c_rate = float(c_rate_override if c_rate_override is not None else self.current_c_rate())
        ambient = float(self.scenario["ambient_C"])
        cooling = float(cooling_override if cooling_override is not None else self.scenario.get("cooling_factor", 1.0))
        capacity_ah = float(self.config["nominal_capacity_Ah"])
        current_a = c_rate * capacity_ah
        self.soc = max(0.0, self.soc - c_rate * dt_s / 3600.0)
        resistance = float(self.config["internal_resistance_ohm"]) * (1.0 + 0.35 * (1.0 - self.soc))
        heat_w = current_a * current_a * resistance + float(self.scenario.get("extra_heat_W", 0.0))
        thermal_capacity = float(self.config["cell_mass_kg"]) * float(self.config["cell_heat_capacity_J_kg_K"])
        heat_transfer_w_k = 0.16 * cooling
        dtemp_dt = (heat_w - heat_transfer_w_k * (self.temperature_c - ambient)) / thermal_capacity
        self.temperature_c += dtemp_dt * dt_s
        ocv = 3.0 + 1.2 * self.soc - 0.08 * (1.0 - self.soc) ** 2
        voltage = max(2.5, ocv - current_a * resistance)
        self.time_s += dt_s
        return {
            "source": "software_simulator",
            "current_A": round(current_a + self.random.gauss(0.0, 0.015), 4),
            "c_rate": round(c_rate, 4),
            "voltage_V": round(voltage + self.random.gauss(0.0, 0.002), 4),
            "soc_percent": round(self.soc * 100.0, 3),
            "ambient_C": ambient,
            "surface_temperature_C": round(self.temperature_c + self.random.gauss(0.0, 0.025), 4),
            "heat_generation_W": round(heat_w, 4),
            "cooling_factor": round(cooling, 3),
        }


def forecast_plant(
    state: dict,
    simulation_config: dict,
    horizon_s: float,
    c_rate: float,
    cooling_factor: float,
    extra_heat_w: float = 0.0,
) -> float:
    temperature = float(state["surface_temperature_C"])
    ambient = float(state["ambient_C"])
    capacity_ah = float(simulation_config["nominal_capacity_Ah"])
    current_a = c_rate * capacity_ah
    resistance = float(simulation_config["internal_resistance_ohm"])
    thermal_capacity = float(simulation_config["cell_mass_kg"]) * float(simulation_config["cell_heat_capacity_J_kg_K"])
    heat_transfer_w_k = 0.16 * cooling_factor
    for _ in range(max(1, int(horizon_s // 5))):
        heat = current_a * current_a * resistance + extra_heat_w
        temperature += ((heat - heat_transfer_w_k * (temperature - ambient)) / thermal_capacity) * 5.0
    return round(temperature, 3)
