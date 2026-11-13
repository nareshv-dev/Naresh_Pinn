"""
tests/test_residuals.py

Unit tests for PINN physics residuals.
Tests: output shapes, gradient flow, zero-initialization behavior.
Does NOT require PyBaMM.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.mlp import MLP
from src.models.pinn import PINN
from src.physics.residuals import SolidDiffusionResidual, LumpedThermalResidual


def make_test_tensors(N: int = 16, device: str = "cpu"):
    """Create small (N, 1) tensors with requires_grad=True for t and r."""
    t = torch.rand(N, 1, device=device, requires_grad=True)
    r = (torch.rand(N, 1, device=device) * 0.9 + 0.05).requires_grad_(True)  # avoid r=0
    I = torch.ones(N, 1, device=device) * 0.5
    T_amb = torch.zeros(N, 1, device=device)  # dimensionless T_amb
    return t, r, I, T_amb


class TestSolidDiffusionResidual:
    """Tests for the SPM solid diffusion residual."""

    @pytest.fixture
    def small_net(self):
        """Tiny MLP for shape tests."""
        return MLP(input_dim=4, output_dim=3, hidden_dims=[8, 8], activation="tanh")

    @pytest.fixture
    def diff_res(self):
        return SolidDiffusionResidual(beta_flux=0.1, c_s_output_idx=1)

    def test_interior_residual_shape(self, small_net, diff_res):
        N = 8
        t, r, I, T_amb = make_test_tensors(N)
        res = diff_res.interior_residual(small_net, t, r, I, T_amb)
        assert res.shape == (N, 1), f"Expected (N, 1), got {res.shape}"

    def test_interior_residual_is_tensor(self, small_net, diff_res):
        t, r, I, T_amb = make_test_tensors(4)
        res = diff_res.interior_residual(small_net, t, r, I, T_amb)
        assert isinstance(res, torch.Tensor)

    def test_interior_residual_has_grad(self, small_net, diff_res):
        """Residual must be differentiable (for backprop)."""
        t, r, I, T_amb = make_test_tensors(4)
        res = diff_res.interior_residual(small_net, t, r, I, T_amb)
        loss = res.pow(2).mean()
        loss.backward()
        # Check that at least one parameter has a gradient
        has_grad = any(
            p.grad is not None for p in small_net.parameters()
        )
        assert has_grad, "No gradients flowed to model parameters from diffusion residual"

    def test_bc_center_shape(self, small_net, diff_res):
        N = 6
        t = torch.rand(N, 1, requires_grad=True)
        r = torch.full((N, 1), 1e-4, requires_grad=True)  # r* ≈ 0
        I = torch.ones(N, 1)
        T_amb = torch.zeros(N, 1)
        res = diff_res.bc_center_residual(small_net, t, r, I, T_amb)
        assert res.shape == (N, 1), f"BC center shape {res.shape}"

    def test_bc_surface_shape(self, small_net, diff_res):
        N = 6
        t = torch.rand(N, 1, requires_grad=True)
        r = torch.full((N, 1), 1.0 - 1e-4, requires_grad=True)  # r* ≈ 1
        I = torch.ones(N, 1) * 0.5
        T_amb = torch.zeros(N, 1)
        res = diff_res.bc_surface_residual(small_net, t, r, I, T_amb)
        assert res.shape == (N, 1), f"BC surface shape {res.shape}"

    def test_ic_residual_shape(self, small_net, diff_res):
        N = 5
        t = torch.zeros(N, 1, requires_grad=True)
        r = torch.rand(N, 1)
        I = torch.ones(N, 1)
        T_amb = torch.zeros(N, 1)
        c_s0 = torch.full((N, 1), 0.866)
        res = diff_res.ic_residual(small_net, t, r, I, T_amb, c_s0)
        assert res.shape == (N, 1)

    def test_requires_grad_assertion(self, small_net, diff_res):
        """Must raise if r_star doesn't have requires_grad=True."""
        N = 4
        t = torch.rand(N, 1, requires_grad=True)
        r = torch.rand(N, 1)  # NO requires_grad — should fail
        I = torch.ones(N, 1)
        T_amb = torch.zeros(N, 1)
        with pytest.raises(AssertionError):
            diff_res.bc_center_residual(small_net, t, r, I, T_amb)


class TestLumpedThermalResidual:
    """Tests for the lumped thermal ODE residual."""

    @pytest.fixture
    def small_net(self):
        return MLP(input_dim=4, output_dim=3, hidden_dims=[8, 8], activation="tanh")

    @pytest.fixture
    def therm_res(self):
        return LumpedThermalResidual(alpha_thermal=0.05, T_output_idx=2)

    def test_interior_shape(self, small_net, therm_res):
        N = 10
        t, r, I, T_amb = make_test_tensors(N)
        res = therm_res.interior_residual(small_net, t, r, I, T_amb)
        assert res.shape == (N, 1)

    def test_thermal_ic_shape(self, small_net, therm_res):
        N = 5
        t = torch.zeros(N, 1, requires_grad=True)
        r = torch.rand(N, 1)
        I = torch.ones(N, 1)
        T_amb = torch.zeros(N, 1)
        res = therm_res.ic_residual(small_net, t, r, I, T_amb)
        assert res.shape == (N, 1)

    def test_thermal_grad_flows(self, small_net, therm_res):
        t, r, I, T_amb = make_test_tensors(8)
        res = therm_res.interior_residual(small_net, t, r, I, T_amb)
        loss = res.pow(2).mean()
        loss.backward()
        has_grad = any(p.grad is not None for p in small_net.parameters())
        assert has_grad, "Thermal residual does not produce gradients"


class TestPINNTotalLoss:
    """Integration tests for PINN total loss computation."""

    @pytest.fixture
    def small_pinn(self):
        return PINN(
            input_dim=4,
            output_dim=3,
            hidden_dims=[8, 8],
            activation="tanh",
            loss_weights={
                "data": 1.0,
                "diffusion_residual": 0.01,
                "thermal_residual": 0.01,
                "ic": 0.1,
                "bc": 0.1,
            },
            beta_flux=0.1,
            alpha_thermal=0.05,
        )

    def _make_col_batch(self, N: int = 8, c_s0: float = 0.866):
        """Make a minimal collocation batch dict."""
        def _coords():
            t = torch.rand(N, 1, requires_grad=True)
            r = (torch.rand(N, 1) * 0.8 + 0.05).requires_grad_(True)
            I = torch.ones(N, 1) * 0.5
            T_amb = torch.zeros(N, 1)
            return t, r, I, T_amb

        t_int, r_int, I_int, T_int = _coords()
        t_ic = torch.zeros(N, 1, requires_grad=True)
        r_ic = torch.rand(N, 1)
        I_ic = torch.ones(N, 1)
        T_ic = torch.zeros(N, 1)
        t_bcc, r_bcc_dummy, I_bcc, T_bcc = _coords()
        r_bcc = torch.full((N, 1), 1e-4, requires_grad=True)
        t_bcs, r_bcs_dummy, I_bcs, T_bcs = _coords()
        r_bcs = torch.full((N, 1), 1.0 - 1e-4, requires_grad=True)

        c_s0_batch = torch.full((N, 1), c_s0)

        return {
            "t_int": t_int, "r_int": r_int, "I_int": I_int, "T_amb_int": T_int,
            "t_ic": t_ic, "r_ic": r_ic, "I_ic": I_ic, "T_amb_ic": T_ic,
            "c_s0_ic": c_s0_batch,
            "t_bc_c": t_bcc, "r_bc_c": r_bcc, "I_bc_c": I_bcc, "T_amb_bc_c": T_bcc,
            "t_bc_s": t_bcs, "r_bc_s": r_bcs, "I_bc_s": I_bcs, "T_amb_bc_s": T_bcs,
        }

    def test_total_loss_returns_scalar(self, small_pinn):
        inputs = torch.rand(16, 4)
        targets = torch.rand(16, 3)
        col = self._make_col_batch(8)
        L_total, loss_dict = small_pinn.total_loss(inputs, targets, col)
        assert L_total.shape == torch.Size([]), "Total loss must be a scalar"

    def test_loss_dict_keys(self, small_pinn):
        inputs = torch.rand(16, 4)
        targets = torch.rand(16, 3)
        col = self._make_col_batch(8)
        _, loss_dict = small_pinn.total_loss(inputs, targets, col)
        expected_keys = {"total", "data", "diffusion", "thermal", "ic", "bc"}
        assert expected_keys == set(loss_dict.keys())

    def test_total_loss_positive(self, small_pinn):
        inputs = torch.rand(16, 4)
        targets = torch.rand(16, 3)
        col = self._make_col_batch(8)
        L_total, _ = small_pinn.total_loss(inputs, targets, col)
        assert float(L_total.item()) >= 0.0

    def test_backward_from_total_loss(self, small_pinn):
        inputs = torch.rand(8, 4)
        targets = torch.rand(8, 3)
        col = self._make_col_batch(4)
        small_pinn.zero_grad()
        L_total, _ = small_pinn.total_loss(inputs, targets, col)
        L_total.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().max() > 0
            for p in small_pinn.parameters()
        )
        assert has_grad, "Backward pass did not produce nonzero gradients"

    def test_data_loss_decreases_with_perfect_targets(self, small_pinn):
        """Data loss should be 0 when predictions perfectly match targets."""
        inputs = torch.rand(16, 4)
        # Targets set to current model output → data loss = 0
        with torch.no_grad():
            targets = small_pinn(inputs).clone()
        L_data = small_pinn.data_loss(inputs, targets)
        assert float(L_data.item()) < 1e-10


class TestNondimensionalization:
    """Test nondim round-trip accuracy."""

    def test_round_trip_time(self):
        from src.preprocessing.nondim import Nondimensionalizer
        cfg = {
            "t_ref_s": 1041.0, "R_s_neg_m": 5.86e-6,
            "c_s_max_neg_mol_m3": 33133.0, "T_ref_K": 298.15,
            "delta_T_K": 50.0, "V_ref_V": 3.7, "delta_V_V": 1.0, "I_ref_A": 5.0,
        }
        nd = Nondimensionalizer(cfg)
        t_s = np.array([0.0, 500.0, 1041.0, 3600.0])
        t_star = nd.time_to_star(t_s)
        t_back = nd.time_from_star(t_star)
        np.testing.assert_allclose(t_back, t_s, rtol=1e-10)

    def test_round_trip_concentration(self):
        from src.preprocessing.nondim import Nondimensionalizer
        cfg = {
            "t_ref_s": 1041.0, "R_s_neg_m": 5.86e-6,
            "c_s_max_neg_mol_m3": 33133.0, "T_ref_K": 298.15,
            "delta_T_K": 50.0, "V_ref_V": 3.7, "delta_V_V": 1.0, "I_ref_A": 5.0,
        }
        nd = Nondimensionalizer(cfg)
        c_mol_m3 = np.array([28000.0, 20000.0, 10000.0])
        c_star = nd.concentration_to_star(c_mol_m3)
        c_back = nd.concentration_from_star(c_star)
        np.testing.assert_allclose(c_back, c_mol_m3, rtol=1e-10)
