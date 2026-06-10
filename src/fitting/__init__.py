"""Memristor compact model parameter fitting package.

This package exposes objective loss evaluation functions and the ModelFitter
interface for fitting compact models to device characterization data.
"""

from fitting.fitter import ModelFitter
from fitting.loss import calculate_linear_nmse, calculate_log_nmse, hybrid_loss

__all__ = [
    "hybrid_loss",
    "calculate_linear_nmse",
    "calculate_log_nmse",
    "ModelFitter",
]
