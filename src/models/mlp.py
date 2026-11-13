"""
src/models/mlp.py

EXP-002: Data-only Multi-Layer Perceptron baseline.

Inputs  (normalized, dimensionless):
  [t_star, r_star, I_star, T_amb_star]   shape (..., 4)

Outputs (normalized, dimensionless):
  [V_star, c_star, T_star]               shape (..., 3)

Architecture:
  - Fully connected layers with configurable hidden_dims
  - Tanh activation (smooth, bounded — beneficial for physics problems)
  - Optional He initialization for weight init

This model has NO physics constraints in its loss. It serves as the
data-only baseline (EXP-002) against which the PINN (EXP-003) is compared.
"""

from __future__ import annotations

import logging
from typing import List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """
    Fully-connected MLP for battery state prediction.

    Parameters
    ----------
    input_dim : int
        Number of input features (default 4: t*, r*, I*, T_amb*).
    output_dim : int
        Number of output targets (default 3: V*, c*, T*).
    hidden_dims : list[int]
        Width of each hidden layer (e.g., [128, 128, 128, 64]).
    activation : str
        Activation function: "tanh" (default) or "relu" or "silu".
    """

    INPUT_NAMES = ["t_star", "r_star", "I_star", "T_amb_star"]
    OUTPUT_NAMES = ["V_star", "c_star", "T_star"]

    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 3,
        hidden_dims: List[int] = None,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128, 128, 64]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation_name = activation

        # Build layer list
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        layers: list[nn.Module] = []
        act = self._make_activation(activation)

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                # Hidden layer: add activation
                layers.append(act)

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "MLP created: %s -> %s | activation=%s | n_params=%d",
            hidden_dims, output_dim, activation, n_params,
        )

    def _make_activation(self, name: str) -> nn.Module:
        mapping = {"tanh": nn.Tanh(), "relu": nn.ReLU(), "silu": nn.SiLU()}
        if name not in mapping:
            raise ValueError(f"Unknown activation '{name}'. Choose from {list(mapping)}")
        return mapping[name]

    def _init_weights(self) -> None:
        """Xavier uniform init for Linear layers (suitable for Tanh)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor, shape (N, 4)
            Normalized inputs [t_star, r_star, I_star, T_amb_star].

        Returns
        -------
        Tensor, shape (N, 3) — [V_star, c_star, T_star].
        """
        assert x.shape[-1] == self.input_dim, (
            f"Expected input dim {self.input_dim}, got {x.shape[-1]}"
        )
        return self.network(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str) -> None:
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "hidden_dims": self.hidden_dims,
                "activation": self.activation_name,
            },
        }, path)
        logger.info("MLP saved to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "MLP":
        ckpt = torch.load(path, map_location=device)
        cfg = ckpt["config"]
        model = cls(**cfg)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        logger.info("MLP loaded from %s", path)
        return model
