"""
src/physics/residuals.py

PINN Physics Residuals for Stage I.

DECLARED SCOPE (Stage I):
  - SPM-derived solid diffusion residual (negative electrode only)
  - Lumped thermal energy balance residual
  This is an SPMe-derived PINN, NOT a full DFN PINN.
  Electrolyte transport, positive electrode, and full Butler-Volmer
  coupling are deferred to Stage II.

════════════════════════════════════════════════════════════════════════════════
GOVERNING EQUATIONS (dimensional, SI units)
════════════════════════════════════════════════════════════════════════════════

1. SOLID DIFFUSION (SPM, negative electrode, spherical symmetry):

   ∂c_s/∂t = (D_s / r²) · ∂/∂r(r² · ∂c_s/∂r)        (radial Fickian)

   BC1 (symmetry at r = 0):
     ∂c_s/∂r |_{r=0} = 0                              (no-flux at center)

   BC2 (surface at r = R_s_neg):
     -D_s · ∂c_s/∂r |_{r=R_s_neg} = j_n / F          (intercalation flux)
     where j_n = I / (a_s · L_neg · A_cell · F)       (volumetric flux)

   IC (t = 0):
     c_s(r, 0) = c_s0 = SOC_init · c_s_max_neg        (uniform)

   Reference: Newman & Thomas-Alyea (2004), Eq. 11.22

════════════════════════════════════════════════════════════════════════════════
NONDIMENSIONALIZATION (applied to the above)
════════════════════════════════════════════════════════════════════════════════

Scales (from configs/simulation_v0.yaml, see src/preprocessing/nondim.py):
  t*   = t / t_ref          where t_ref  = R_s_neg² / D_s0
  r*   = r / R_s_neg        so    r*  ∈ [0, 1]
  c*   = c_s / c_s_max_neg  so    c*  ∈ [0, 1] approximately
  I*   = I / I_ref          where I_ref  = nominal 1C current
  T*   = (T - T_ref) / ΔT
  V*   = (V - V_ref) / ΔV

Dimensionless solid diffusion (with D_s = D_s0 = const in Stage I):
  ∂c*/∂t* = (1/r*²) · ∂/∂r*(r*² · ∂c*/∂r*)

Dimensionless surface flux BC (at r* = 1):
  -∂c*/∂r* |_{r*=1} = β · I*
  where β = (R_s_neg · I_ref) / (D_s0 · a_s · L_neg · A_cell · F · c_s_max_neg)
  [β is a dimensionless group computed from Chen2020 parameters]

Dimensionless symmetry BC (at r* = 0):
  ∂c*/∂r* |_{r*=0} = 0

════════════════════════════════════════════════════════════════════════════════

2. LUMPED THERMAL ENERGY BALANCE:

   ρ · c_p · V_cell · dT/dt = Q_gen - h · A_cell · (T - T_amb)

   where:
     Q_gen  = I · (V_ocv - V) + I · T · dU_ocv/dT  (ohmic + entropic)
   Simplified Stage I model (no separate entropic term):
     Q_gen  ≈ |I| · |η|   where η = V - V_ocv is overpotential
   In practice, PyBaMM computes Q_gen internally; at Stage I we use a
   simplified thermal residual based only on the total heat signal.

   Dimensionless lumped thermal:
     dT*/dt* = α · [Q_gen* - γ · (T* - T_amb*)]

   where:
     α = t_ref / (ρ · c_p · V_cell / (h · A_cell))  [dimensionless ratio]
     γ = dimensionless cooling coefficient
   NOTE: α and γ are declared as learned or fixed parameters (beta_thermal,
   gamma_thermal) in the PINN loss config; they are not hardcoded here.

════════════════════════════════════════════════════════════════════════════════

IMPLEMENTATION NOTES:
  - All residuals use torch.autograd.grad for exact automatic differentiation.
  - create_graph=True is required for higher-order derivatives (d/dr, d/dt).
  - Shape assertions are placed around every residual computation.
  - Unit/scale comments label every line.

"""

from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Utility: gradient via autograd
# ══════════════════════════════════════════════════════════════════════════════

def _grad(output: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    """
    Compute ∂output/∂inp using torch.autograd.grad.

    Parameters
    ----------
    output : Tensor, shape (N, 1) or (N,)
    inp    : Tensor, shape (N, 1) or (N,) — must have requires_grad=True

    Returns
    -------
    Tensor, same shape as inp.
    """
    g = torch.autograd.grad(
        outputs=output,
        inputs=inp,
        grad_outputs=torch.ones_like(output),
        create_graph=True,   # allows higher-order gradients
        retain_graph=True,
        allow_unused=True,
    )[0]
    if g is None:
        # No gradient path — return zeros (unusual but safe)
        return torch.zeros_like(inp)
    return g


# ══════════════════════════════════════════════════════════════════════════════
# Solid Diffusion Residual (SPM, negative electrode)
# ══════════════════════════════════════════════════════════════════════════════

class SolidDiffusionResidual:
    """
    Computes the dimensionless SPM solid diffusion residual and BC violations.

    Residual (interior):
      R_diff = ∂c*/∂t* - (1/r*²) · ∂/∂r*(r*² · ∂c*/∂r*)
             = ∂c*/∂t* - (∂²c*/∂r*² + 2/r* · ∂c*/∂r*)    [expanded]

      Should be ≈ 0 for a solution satisfying the PDE.

    BC1 residual (symmetry at r* = 0):
      R_bc_center = ∂c*/∂r* |_{r*=0}   (should be = 0)

    BC2 residual (surface flux at r* = 1):
      R_bc_surface = ∂c*/∂r* |_{r*=1} + β · I*   (should be = 0)
      where β is the dimensionless flux coupling coefficient.

    IC residual (at t* = 0):
      R_ic = c*(r*, 0) - c_s0_star    (should be = 0)

    Parameters
    ----------
    beta_flux : float
        Dimensionless surface flux coupling coefficient.
        β = (R_s_neg · I_ref) / (D_s0 · a_s · L_neg · A_cell · F · c_s_max_neg)
        Declared in YAML config; default approximate value for Chen2020.
    c_s_output_idx : int
        Index into the model output vector for c_star (default: 1).
    """

    def __init__(
        self,
        beta_flux: float = 0.1,   # approximate dimensionless group; override from config
        c_s_output_idx: int = 1,
    ) -> None:
        self.beta = float(beta_flux)
        self.c_idx = c_s_output_idx

    def interior_residual(
        self,
        net: nn.Module,
        t_star: torch.Tensor,   # shape (N, 1), requires_grad=True
        r_star: torch.Tensor,   # shape (N, 1), requires_grad=True
        I_star: torch.Tensor,   # shape (N, 1) — applied current (not differentiated)
        T_amb_star: torch.Tensor,  # shape (N, 1) — ambient temp (not differentiated)
    ) -> torch.Tensor:
        """
        Compute PDE residual at interior collocation points.

        R_diff = ∂c*/∂t* - (∂²c*/∂r*² + (2/r*) · ∂c*/∂r*)

        Returns
        -------
        Tensor shape (N, 1) — residual at each collocation point.
        """
        # Shape assertions
        assert t_star.shape == r_star.shape, (
            f"t_star {t_star.shape} != r_star {r_star.shape}"
        )
        assert t_star.requires_grad, "t_star must have requires_grad=True"
        assert r_star.requires_grad, "r_star must have requires_grad=True"

        N = t_star.shape[0]

        # Build input: [t*, r*, I*, T_amb*]  — shape (N, 4)
        inputs = torch.cat([t_star, r_star, I_star, T_amb_star], dim=1)
        # [dimensionless units throughout]

        # Forward pass — output: [V*, c*, T*]  shape (N, 3)
        outputs = net(inputs)
        assert outputs.shape == (N, 3), f"Net output shape {outputs.shape} ≠ (N, 3)"

        # Extract c* — shape (N, 1)
        # c* [dimensionless, ≈ c_s / c_s_max_neg]
        c_star = outputs[:, self.c_idx : self.c_idx + 1]

        # ∂c*/∂t*  [dimensionless / dimensionless = dimensionless]
        dc_dt = _grad(c_star, t_star)   # shape (N, 1)

        # ∂c*/∂r*  [dimensionless / dimensionless]
        dc_dr = _grad(c_star, r_star)   # shape (N, 1)

        # ∂²c*/∂r*²
        d2c_dr2 = _grad(dc_dr, r_star)  # shape (N, 1)

        # Laplacian in spherical coords (dimensionless):
        #   ∂²c*/∂r*² + (2/r*) · ∂c*/∂r*
        # NOTE: r_star > 0 at interior points (clamped in sampler)
        laplacian = d2c_dr2 + (2.0 / r_star) * dc_dr   # [dimensionless]

        # PDE residual:  ∂c*/∂t* - ∇²c* = 0
        residual = dc_dt - laplacian   # shape (N, 1) [dimensionless]

        assert residual.shape == (N, 1), f"Residual shape {residual.shape} ≠ (N, 1)"
        return residual

    def bc_center_residual(
        self,
        net: nn.Module,
        t_star: torch.Tensor,   # shape (N, 1)
        r_star: torch.Tensor,   # shape (N, 1), all values ≈ 0
        I_star: torch.Tensor,
        T_amb_star: torch.Tensor,
    ) -> torch.Tensor:
        """
        BC1 residual: ∂c*/∂r* |_{r*=0} = 0   (symmetry, no-flux at center)

        Returns
        -------
        Tensor shape (N, 1).
        """
        assert r_star.requires_grad, "r_star must have requires_grad=True"
        N = t_star.shape[0]

        inputs = torch.cat([t_star, r_star, I_star, T_amb_star], dim=1)
        outputs = net(inputs)
        c_star = outputs[:, self.c_idx : self.c_idx + 1]   # [dimensionless]

        # ∂c*/∂r*  at r* ≈ 0  [dimensionless / dimensionless]
        dc_dr = _grad(c_star, r_star)   # shape (N, 1)

        assert dc_dr.shape == (N, 1), f"BC center shape {dc_dr.shape}"
        return dc_dr   # residual: should = 0

    def bc_surface_residual(
        self,
        net: nn.Module,
        t_star: torch.Tensor,   # shape (N, 1)
        r_star: torch.Tensor,   # shape (N, 1), all values ≈ 1
        I_star: torch.Tensor,   # shape (N, 1)  [dimensionless current = I/I_ref]
        T_amb_star: torch.Tensor,
    ) -> torch.Tensor:
        """
        BC2 residual: ∂c*/∂r* |_{r*=1} + β · I* = 0

        The surface flux BC in dimensionless form:
          -D_s · ∂c_s/∂r |_{r=R_s} = j_n / F
        becomes, after nondimensionalization:
          ∂c*/∂r* |_{r*=1} = -β · I*
        →  R_bc_surface = ∂c*/∂r* |_{r*=1} + β · I*   (should = 0)

        β [dimensionless] = R_s_neg · I_ref / (D_s0 · a_s · L_neg · A_cell · F · c_s_max_neg)

        Returns
        -------
        Tensor shape (N, 1).
        """
        assert r_star.requires_grad, "r_star must have requires_grad=True"
        N = t_star.shape[0]

        inputs = torch.cat([t_star, r_star, I_star, T_amb_star], dim=1)
        outputs = net(inputs)
        c_star = outputs[:, self.c_idx : self.c_idx + 1]   # [dimensionless]

        # ∂c*/∂r*  at r* ≈ 1  [dimensionless]
        dc_dr = _grad(c_star, r_star)   # shape (N, 1)

        # β · I*  [dimensionless · dimensionless = dimensionless]
        flux_term = torch.tensor(self.beta, dtype=t_star.dtype, device=t_star.device) * I_star

        # Residual: ∂c*/∂r* + β·I* = 0
        residual = dc_dr + flux_term   # shape (N, 1)

        assert residual.shape == (N, 1), f"BC surface shape {residual.shape}"
        return residual

    def ic_residual(
        self,
        net: nn.Module,
        t_star: torch.Tensor,    # shape (N, 1), all values ≈ 0
        r_star: torch.Tensor,    # shape (N, 1)
        I_star: torch.Tensor,
        T_amb_star: torch.Tensor,
        c_s0_star: torch.Tensor, # shape (N, 1) — reference IC value [dimensionless]
    ) -> torch.Tensor:
        """
        IC residual: c*(r*, 0) - c_s0*  = 0

        c_s0_star = SOC_init · (c_s0_neg / c_s_max_neg)  [dimensionless]

        Returns
        -------
        Tensor shape (N, 1).
        """
        N = t_star.shape[0]

        inputs = torch.cat([t_star, r_star, I_star, T_amb_star], dim=1)
        outputs = net(inputs)
        c_star = outputs[:, self.c_idx : self.c_idx + 1]   # [dimensionless]

        # Residual: c*(t*=0, r*) - c_s0*  [dimensionless]
        residual = c_star - c_s0_star   # shape (N, 1)
        assert residual.shape == (N, 1)
        return residual


# ══════════════════════════════════════════════════════════════════════════════
# Lumped Thermal Residual
# ══════════════════════════════════════════════════════════════════════════════

class LumpedThermalResidual:
    """
    Dimensionless lumped thermal energy balance residual.

    GOVERNING EQUATION (dimensional):
      m · c_p · dT/dt = Q_gen(t) - h · A · (T - T_amb)

    After nondimensionalization (T* = (T - T_ref) / ΔT, t* = t / t_ref):
      dT*/dt* = α · [Q_gen*(t*) - γ · (T* - T_amb*)]

    where:
      α = t_ref · h · A / (m · c_p)   [dimensionless cooling ratio]
      γ = 1 / α  (absorbed into Q_gen* definition below)

    Stage I simplification: We compute the residual in terms of the NN
    outputs (T*) and the prescribed heat source from data (Q_gen* from
    PyBaMM), treating it as a forcing term evaluated at collocation times.
    This avoids requiring Butler-Volmer kinetics at Stage I.

    In simplified form:
      R_thermal = dT*/dt* - alpha_th · (T_amb* - T*)

    This is the free-cooling / linear-cooling ODE without explicit heat
    generation (since Q_gen is embedded in training data implicitly).
    A richer form including Q_gen forcing is Stage II.

    NOTE: Declared explicitly as the simplified Stage I lumped thermal residual.

    Parameters
    ----------
    alpha_thermal : float
        Dimensionless cooling coefficient α = t_ref · h · A / (m · c_p).
        Default approximate value; should be set from config.
    T_output_idx : int
        Index into model output for T* (default: 2).
    """

    def __init__(
        self,
        alpha_thermal: float = 0.05,
        T_output_idx: int = 2,
    ) -> None:
        self.alpha = float(alpha_thermal)
        self.T_idx = T_output_idx

    def interior_residual(
        self,
        net: nn.Module,
        t_star: torch.Tensor,      # shape (N, 1), requires_grad=True
        r_star: torch.Tensor,      # shape (N, 1) — included for unified interface
        I_star: torch.Tensor,      # shape (N, 1)
        T_amb_star: torch.Tensor,  # shape (N, 1)  [dimensionless]
    ) -> torch.Tensor:
        """
        Simplified lumped thermal residual:

          R_th = dT*/dt* - α · (T_amb* - T*)   (should = 0)

        This enforces Newton's law of cooling in dimensionless form.
        T is a function of t only (lumped); r_star is not used but
        kept for API uniformity with SolidDiffusionResidual.

        Returns
        -------
        Tensor shape (N, 1).
        """
        assert t_star.requires_grad, "t_star must have requires_grad=True"
        N = t_star.shape[0]

        inputs = torch.cat([t_star, r_star, I_star, T_amb_star], dim=1)
        outputs = net(inputs)
        assert outputs.shape == (N, 3), f"Output shape {outputs.shape}"

        # T* — shape (N, 1) [dimensionless temperature]
        T_star = outputs[:, self.T_idx : self.T_idx + 1]

        # dT*/dt*  [dimensionless / dimensionless]
        dT_dt = _grad(T_star, t_star)   # shape (N, 1)

        # α · (T_amb* - T*)  [dimensionless]
        cooling = (
            torch.tensor(self.alpha, dtype=t_star.dtype, device=t_star.device)
            * (T_amb_star - T_star)
        )

        # Residual: dT*/dt* - α·(T_amb* - T*) = 0
        residual = dT_dt - cooling   # shape (N, 1) [dimensionless]
        assert residual.shape == (N, 1)
        return residual

    def ic_residual(
        self,
        net: nn.Module,
        t_star: torch.Tensor,      # shape (N, 1), all ≈ 0
        r_star: torch.Tensor,
        I_star: torch.Tensor,
        T_amb_star: torch.Tensor,  # IC: T(0) = T_amb
    ) -> torch.Tensor:
        """
        Thermal IC residual: T*(r*, 0) - T_amb* = 0

        Returns
        -------
        Tensor shape (N, 1).
        """
        N = t_star.shape[0]
        inputs = torch.cat([t_star, r_star, I_star, T_amb_star], dim=1)
        outputs = net(inputs)
        T_star = outputs[:, self.T_idx : self.T_idx + 1]
        residual = T_star - T_amb_star   # shape (N, 1) [dimensionless]
        assert residual.shape == (N, 1)
        return residual
