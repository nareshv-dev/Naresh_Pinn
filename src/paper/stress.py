"""Analytical concentration-to-stress operator for a spherical particle."""
from __future__ import annotations

import torch


def spherical_particle_stress(
    concentration: torch.Tensor,
    radius: torch.Tensor,
    youngs_modulus: float,
    poissons_ratio: float,
    partial_molar_volume: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return radial and tangential diffusion-induced stress.

    Inputs are batches of full radial profiles with shape ``(batch, n_radius)``.
    Radius must be normalized to [0, 1]. The operator uses trapezoidal spherical
    volume averages and satisfies sigma_r(1)=0 by construction.
    """
    if concentration.ndim != 2:
        raise ValueError("concentration must have shape (batch, n_radius)")
    if radius.ndim == 1:
        radius = radius.unsqueeze(0).expand_as(concentration)
    eps = torch.finfo(concentration.dtype).eps
    # Stress is invariant to a uniform concentration offset. Subtracting the
    # centre value before quadrature makes that invariance exact on coarse grids.
    shifted = concentration - concentration[:, :1]
    integral_total = torch.trapz(shifted * radius.square(), radius, dim=1,).unsqueeze(1)
    dr = radius[:, 1:] - radius[:, :-1]
    shells = 0.5 * (
        shifted[:, 1:] * radius[:, 1:].square()
        + shifted[:, :-1] * radius[:, :-1].square()
    ) * dr
    cumulative = torch.cat(
        [torch.zeros_like(concentration[:, :1]), torch.cumsum(shells, dim=1)], dim=1
    )
    local_average = 3.0 * cumulative / torch.clamp(radius.pow(3), min=eps)
    local_average[:, 0] = shifted[:, 0]
    particle_average = 3.0 * integral_total
    factor = youngs_modulus * partial_molar_volume / (3.0 * (1.0 - poissons_ratio))
    sigma_r = 2.0 * factor * (particle_average - local_average)
    sigma_t = factor * (2.0 * particle_average + local_average - 3.0 * shifted)
    sigma_r[:, -1] = 0.0
    return sigma_r, sigma_t
