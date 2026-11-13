"""Three paper comparison models with matched four-layer tanh backbones."""
from __future__ import annotations

import torch
from torch import nn


class TanhMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, width: int = 96, depth: int = 4):
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(depth):
            linear = nn.Linear(dim, width)
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.extend([linear, nn.Tanh()])
            dim = width
        layers.append(nn.Linear(dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class _HardICMixin:
    c0_star: float

    def hard_concentration(self, raw: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return self.c0_star + tau * raw

    @staticmethod
    def hard_temperature(raw: torch.Tensor, tau: torch.Tensor, ambient_star: torch.Tensor) -> torch.Tensor:
        return ambient_star + tau * raw


class DataOnlyModel(nn.Module, _HardICMixin):
    """Data-only baseline with independent concentration, temperature and stress outputs."""
    def __init__(self, c0_star: float, width: int = 96, depth: int = 4):
        super().__init__()
        self.c0_star = c0_star
        self.net = TanhMLP(4, 4, width, depth)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.net(x)
        tau, ambient = x[:, 1:2], x[:, 3:4]
        return {
            "c": self.hard_concentration(raw[:, 0:1], tau),
            "T": self.hard_temperature(raw[:, 1:2], tau, ambient),
            "sigma_r": tau * raw[:, 2:3],
            "sigma_t": tau * raw[:, 3:4],
        }


class PriorPINN(DataOnlyModel):
    """Prior-style PINN: physics residuals but independent learned stress heads."""


class PCMPPINN(nn.Module, _HardICMixin):
    """Proposed model: separate fields and no independent stress output heads."""
    def __init__(self, c0_star: float, width: int = 96, depth: int = 4):
        super().__init__()
        self.c0_star = c0_star
        self.concentration_net = TanhMLP(4, 1, width, depth)
        self.temperature_net = TanhMLP(3, 1, width, depth)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        rho, tau, c_rate, ambient = x.split(1, dim=1)
        c_raw = self.concentration_net(x)
        t_raw = self.temperature_net(torch.cat([tau, c_rate, ambient], dim=1))
        return {
            "c": self.hard_concentration(c_raw, tau),
            "T": self.hard_temperature(t_raw, tau, ambient),
        }
