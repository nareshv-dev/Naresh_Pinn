"""
src/evaluation/__init__.py
"""
from .metrics import compute_metrics, compute_residual_norms
from .figures import FigureGenerator

__all__ = ["compute_metrics", "compute_residual_norms", "FigureGenerator"]
