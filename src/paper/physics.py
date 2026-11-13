"""Dimensionless residuals used by the paper-aligned models."""
from __future__ import annotations

import torch


def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True, retain_graph=True)[0]


def diffusion_residual(model, x: torch.Tensor, diffusivity: float, horizon: float, radius_m: float):
    x = x.requires_grad_(True)
    c = model(x)["c"]
    dc = _grad(c, x)
    c_r, c_t = dc[:, 0:1], dc[:, 1:2]
    c_rr = _grad(c_r, x)[:, 0:1]
    rho = torch.clamp(x[:, 0:1], min=1e-3)
    kappa = diffusivity * horizon / (radius_m * radius_m)
    return c_t - kappa * (c_rr + 2.0 * c_r / rho)


def thermal_residual(model, x: torch.Tensor, heat_star: torch.Tensor, cooling_star: float):
    x = x.requires_grad_(True)
    temperature = model(x)["T"]
    dtemperature_dtau = _grad(temperature, x)[:, 1:2]
    return dtemperature_dtau - heat_star + cooling_star * (temperature - x[:, 3:4])


def center_boundary_residual(model, x: torch.Tensor):
    x = x.clone().detach()
    x[:, 0] = 0.0
    x.requires_grad_(True)
    return _grad(model(x)["c"], x)[:, 0:1]


def surface_boundary_residual(model, x: torch.Tensor, target_gradient: torch.Tensor):
    x = x.clone().detach()
    x[:, 0] = 1.0
    x.requires_grad_(True)
    dc_drho = _grad(model(x)["c"], x)[:, 0:1]
    return dc_drho - target_gradient


def mechanical_equilibrium_residual(model, x: torch.Tensor):
    x = x.requires_grad_(True)
    output = model(x)
    radial, tangential = output["sigma_r"], output["sigma_t"]
    dsigma_dr = _grad(radial, x)[:, 0:1]
    rho = torch.clamp(x[:, 0:1], min=1e-3)
    return dsigma_dr + 2.0 * (radial - tangential) / rho
