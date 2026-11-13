import numpy as np
import pandas as pd
import torch

from src.paper.dataset import load_config, paper_conditions
from src.paper.models import PCMPPINN
from src.paper.stress import spherical_particle_stress


def test_paper_condition_split():
    config = load_config("configs/paper_experiment.yaml")
    conditions = paper_conditions(config)
    assert len(conditions) == 18
    assert sum(c.split == "train" for c in conditions) == 12
    assert sum(c.split == "validation" for c in conditions) == 4
    assert sum(c.split == "test" for c in conditions) == 2
    assert len({(c.c_rate, c.ambient_c) for c in conditions}) == 18


def test_proposed_model_hard_initial_conditions():
    model = PCMPPINN(c0_star=0.91)
    x = torch.tensor([[0.3, 0.0, 0.5, 0.4]], dtype=torch.float32)
    output = model(x)
    assert torch.allclose(output["c"], torch.tensor([[0.91]]))
    assert torch.allclose(output["T"], torch.tensor([[0.4]]))
    assert "sigma_t" not in output


def test_stress_operator_surface_boundary_and_uniform_zero():
    radius = torch.linspace(0, 1, 25)
    concentration = torch.full((2, 25), 28000.0)
    radial, tangential = spherical_particle_stress(concentration, radius, 15e9, 0.3, 3.1e-6)
    assert torch.allclose(radial[:, -1], torch.zeros(2))
    assert float(radial.abs().max()) < 1e4
    assert float(tangential.abs().max()) < 1e4


def test_generated_dataset_shape_when_present():
    path = "data/processed/paper_v1/dataset.parquet"
    frame = pd.read_parquet(path)
    assert frame.shape[0] == 45000
    assert frame.condition_id.nunique() == 18
    assert np.isfinite(frame.select_dtypes("number").to_numpy()).all()
