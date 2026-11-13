"""
src/models/pinn.py

EXP-003: Fixed-weight Physics-Informed Neural Network.

Architecture:
  - Same MLP backbone as EXP-002 (src.models.mlp.MLP)
  - Additional physics loss terms computed via src.physics.residuals

Loss decomposition (all terms computed separately, logged independently):
  L_total = w_data · L_data
          + w_diff · L_diffusion
          + w_therm · L_thermal
          + w_ic · L_ic
          + w_bc · L_bc

  L_data        : MSE on labeled observations [V*, c*, T*]
  L_diffusion   : MSE of solid diffusion PDE residual at interior points
  L_thermal     : MSE of lumped thermal ODE residual at interior points
  L_ic          : MSE of [solid IC + thermal IC] at t* = 0
  L_bc          : MSE of [BC center (symmetry) + BC surface (flux)] at r* = 0, 1

Fixed loss weights (no adaptive weighting at Stage I).

Physics residuals:
  See src/physics/residuals.py for full documentation of the governing equations.

SCOPE REMINDER:
  This PINN enforces:
    ✓ Radial solid diffusion in the negative electrode (SPM-level)
    ✓ Lumped thermal cooling ODE
    ✗ Electrolyte transport (Stage II)
    ✗ Positive electrode (Stage II)
    ✗ Full Butler-Volmer kinetics (Stage II)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .mlp import MLP
from src.physics.residuals import SolidDiffusionResidual, LumpedThermalResidual

logger = logging.getLogger(__name__)


class PINN(nn.Module):
    """
    EXP-003: Physics-Informed Neural Network for Li-Ion cells.

    Wraps an MLP backbone with physics loss computation methods.

    Parameters
    ----------
    input_dim, output_dim, hidden_dims, activation : see MLP
    loss_weights : dict with keys data, diffusion_residual, thermal_residual, ic, bc
    beta_flux : float — dimensionless surface flux coefficient (see residuals.py)
    alpha_thermal : float — dimensionless cooling coefficient (see residuals.py)
    """

    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 3,
        hidden_dims: List[int] = None,
        activation: str = "tanh",
        loss_weights: Optional[Dict[str, float]] = None,
        beta_flux: float = 0.1,
        alpha_thermal: float = 0.05,
    ) -> None:
        super().__init__()

        if loss_weights is None:
            loss_weights = {
                "data": 1.0,
                "diffusion_residual": 0.01,
                "thermal_residual": 0.01,
                "ic": 0.1,
                "bc": 0.1,
            }
        self.loss_weights = loss_weights

        # MLP backbone (shared with EXP-002 architecture)
        self.net = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims or [128, 128, 128, 64],
            activation=activation,
        )

        # Physics residual objects
        self.diff_residual = SolidDiffusionResidual(
            beta_flux=beta_flux,
            c_s_output_idx=1,   # output index for c* in [V*, c*, T*]
        )
        self.therm_residual = LumpedThermalResidual(
            alpha_thermal=alpha_thermal,
            T_output_idx=2,     # output index for T* in [V*, c*, T*]
        )

        n_params = self.net.count_parameters()
        logger.info(
            "PINN created | n_params=%d | b_flux=%.4f | a_th=%.4f | weights=%s",
            n_params, beta_flux, alpha_thermal, loss_weights,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — same as MLP."""
        return self.net(x)

    # ── Loss components ────────────────────────────────────────────────────────

    def data_loss(
        self,
        inputs: torch.Tensor,    # shape (N, 4)
        targets: torch.Tensor,   # shape (N, 3)  [V*, c*, T*]
    ) -> torch.Tensor:
        """
        Supervised data loss: MSE between predictions and labeled observations.

          L_data = (1/N) Σ ||net(x_i) - y_i||²    [dimensionless²]
        """
        preds = self.net(inputs)   # shape (N, 3)
        assert preds.shape == targets.shape, f"{preds.shape} ≠ {targets.shape}"
        return nn.functional.mse_loss(preds, targets)

    def diffusion_residual_loss(
        self,
        t_col: torch.Tensor,      # shape (N, 1) requires_grad=True  [dimensionless]
        r_col: torch.Tensor,      # shape (N, 1) requires_grad=True  [dimensionless]
        I_col: torch.Tensor,      # shape (N, 1)  [dimensionless current]
        T_amb_col: torch.Tensor,  # shape (N, 1)  [dimensionless T_amb]
    ) -> torch.Tensor:
        """
        PDE residual loss for solid diffusion (interior collocation points).

          L_diff = (1/N) Σ R_diff(t*, r*)²    [dimensionless²]

        R_diff = ∂c*/∂t* - (∂²c*/∂r*² + (2/r*) · ∂c*/∂r*)
        """
        res = self.diff_residual.interior_residual(
            self.net, t_col, r_col, I_col, T_amb_col
        )   # shape (N, 1)
        return res.pow(2).mean()

    def thermal_residual_loss(
        self,
        t_col: torch.Tensor,
        r_col: torch.Tensor,
        I_col: torch.Tensor,
        T_amb_col: torch.Tensor,
    ) -> torch.Tensor:
        """
        ODE residual loss for lumped thermal model.

          L_therm = (1/N) Σ R_th(t*)²    [dimensionless²]

        R_th = dT*/dt* - α·(T_amb* - T*)
        """
        res = self.therm_residual.interior_residual(
            self.net, t_col, r_col, I_col, T_amb_col
        )   # shape (N, 1)
        return res.pow(2).mean()

    def ic_loss(
        self,
        t_ic: torch.Tensor,       # shape (N, 1), all ≈ 0
        r_ic: torch.Tensor,       # shape (N, 1)
        I_ic: torch.Tensor,
        T_amb_ic: torch.Tensor,
        c_s0_star: torch.Tensor,  # shape (N, 1)  — known IC concentration
    ) -> torch.Tensor:
        """
        Initial condition loss:
          L_ic = MSE[c*(t*=0, r*) - c_s0*] + MSE[T*(t*=0) - T_amb*]

        Both solid concentration and temperature must match initial conditions.
        """
        # Solid concentration IC: c*(t*=0, r*) = c_s0*
        res_c = self.diff_residual.ic_residual(
            self.net, t_ic, r_ic, I_ic, T_amb_ic, c_s0_star
        )   # shape (N, 1)

        # Thermal IC: T*(t*=0) = T_amb*
        res_T = self.therm_residual.ic_residual(
            self.net, t_ic, r_ic, I_ic, T_amb_ic
        )   # shape (N, 1)

        return res_c.pow(2).mean() + res_T.pow(2).mean()

    def bc_loss(
        self,
        # BC center (r* = 0)
        t_bc_c: torch.Tensor,
        r_bc_c: torch.Tensor,
        I_bc_c: torch.Tensor,
        T_amb_bc_c: torch.Tensor,
        # BC surface (r* = 1)
        t_bc_s: torch.Tensor,
        r_bc_s: torch.Tensor,
        I_bc_s: torch.Tensor,
        T_amb_bc_s: torch.Tensor,
    ) -> torch.Tensor:
        """
        Boundary condition loss:
          L_bc = MSE[∂c*/∂r*|_{r*=0}] + MSE[∂c*/∂r*|_{r*=1} + β·I*]

        Enforces:
          1. No-flux symmetry at particle center
          2. Intercalation flux at particle surface
        """
        # BC1: symmetry (∂c*/∂r* = 0 at r* = 0)
        res_center = self.diff_residual.bc_center_residual(
            self.net, t_bc_c, r_bc_c, I_bc_c, T_amb_bc_c
        )   # shape (N, 1)

        # BC2: surface flux (∂c*/∂r* + β·I* = 0 at r* = 1)
        res_surface = self.diff_residual.bc_surface_residual(
            self.net, t_bc_s, r_bc_s, I_bc_s, T_amb_bc_s
        )   # shape (N, 1)

        return res_center.pow(2).mean() + res_surface.pow(2).mean()

    def total_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        col_batch: dict,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total weighted PINN loss and return all components for logging.

        Parameters
        ----------
        inputs  : shape (N_lab, 4) — labeled inputs
        targets : shape (N_lab, 3) — labeled targets
        col_batch : dict with keys:
          t_int, r_int, I_int, T_amb_int           — interior
          t_ic, r_ic, I_ic, T_amb_ic, c_s0_ic      — IC
          t_bc_c, r_bc_c, I_bc_c, T_amb_bc_c       — BC center
          t_bc_s, r_bc_s, I_bc_s, T_amb_bc_s       — BC surface

        Returns
        -------
        (total_loss, loss_dict)
        """
        w = self.loss_weights

        # ── Data loss ─────────────────────────────────────────────────────
        L_data = self.data_loss(inputs, targets)

        # ── Physics losses ─────────────────────────────────────────────────
        L_diff = self.diffusion_residual_loss(
            col_batch["t_int"], col_batch["r_int"],
            col_batch["I_int"], col_batch["T_amb_int"],
        )
        L_therm = self.thermal_residual_loss(
            col_batch["t_int"], col_batch["r_int"],
            col_batch["I_int"], col_batch["T_amb_int"],
        )
        L_ic = self.ic_loss(
            col_batch["t_ic"], col_batch["r_ic"],
            col_batch["I_ic"], col_batch["T_amb_ic"],
            col_batch["c_s0_ic"],
        )
        L_bc = self.bc_loss(
            col_batch["t_bc_c"], col_batch["r_bc_c"],
            col_batch["I_bc_c"], col_batch["T_amb_bc_c"],
            col_batch["t_bc_s"], col_batch["r_bc_s"],
            col_batch["I_bc_s"], col_batch["T_amb_bc_s"],
        )

        L_total = (
            w["data"] * L_data
            + w["diffusion_residual"] * L_diff
            + w["thermal_residual"] * L_therm
            + w["ic"] * L_ic
            + w["bc"] * L_bc
        )

        loss_dict = {
            "total": float(L_total.item()),
            "data": float(L_data.item()),
            "diffusion": float(L_diff.item()),
            "thermal": float(L_therm.item()),
            "ic": float(L_ic.item()),
            "bc": float(L_bc.item()),
        }
        return L_total, loss_dict

    def count_parameters(self) -> int:
        return self.net.count_parameters()

    def save(self, path: str) -> None:
        torch.save({
            "state_dict": self.state_dict(),
            "net_config": {
                "input_dim": self.net.input_dim,
                "output_dim": self.net.output_dim,
                "hidden_dims": self.net.hidden_dims,
                "activation": self.net.activation_name,
            },
            "pinn_config": {
                "loss_weights": self.loss_weights,
                "beta_flux": self.diff_residual.beta,
                "alpha_thermal": self.therm_residual.alpha,
            },
        }, path)
        logger.info("PINN saved to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "PINN":
        ckpt = torch.load(path, map_location=device)
        nc = ckpt["net_config"]
        pc = ckpt["pinn_config"]
        model = cls(
            input_dim=nc["input_dim"],
            output_dim=nc["output_dim"],
            hidden_dims=nc["hidden_dims"],
            activation=nc["activation"],
            loss_weights=pc["loss_weights"],
            beta_flux=pc["beta_flux"],
            alpha_thermal=pc["alpha_thermal"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        logger.info("PINN loaded from %s", path)
        return model
