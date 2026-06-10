"""Objective loss functions for fitting memristor compact models.

This module provides normalized mean-squared-error (NMSE) functions in both
linear and logarithmic spaces to capture broad-range switching dynamics.
"""

import numpy as np


def calculate_linear_nmse(i_sim: np.ndarray, i_target: np.ndarray, eps: float = 1e-20) -> float:
    """Calculates Normalized Mean Squared Error in linear current space.

    NMSE = Mean((i_sim - i_target)^2) / (Var(i_target) + eps)

    Args:
        i_sim: Simulated current array (A).
        i_target: Target current array (A).
        eps: Small regularization factor to prevent division by zero.

    Returns:
        float: Linear normalized MSE.
    """
    mse = np.mean((i_sim - i_target) ** 2)
    var = np.var(i_target)
    return float(mse / (var + eps))


def calculate_log_nmse(
    i_sim: np.ndarray, i_target: np.ndarray, floor: float = 1.0e-10, eps: float = 1e-20
) -> float:
    """Calculates Normalized Mean Squared Error in logarithmic current space.

    Converts currents to log10 space with a measurement floor before evaluating NMSE.

    Args:
        i_sim: Simulated current array (A).
        i_target: Target current array (A).
        floor: Noise floor below which currents are clipped (A).
        eps: Regularization factor.

    Returns:
        float: Logarithmic normalized MSE.
    """
    log_sim = np.log10(np.abs(i_sim) + floor)
    log_target = np.log10(np.abs(i_target) + floor)

    mse = np.mean((log_sim - log_target) ** 2)
    var = np.var(log_target)
    return float(mse / (var + eps))


def hybrid_loss(
    i_sim: np.ndarray,
    i_target: np.ndarray,
    gamma: float = 0.3,
    floor: float = 1.0e-10,
    eps: float = 1e-20,
) -> float:
    """Calculates a hybrid loss combining normalized linear and log losses.

    Loss = gamma * NMSE_linear + (1 - gamma) * NMSE_log

    Args:
        i_sim: Simulated current array (A).
        i_target: Target current array (A).
        gamma: Weight factor for linear loss (0.0 to 1.0).
        floor: Noise floor for log conversion (A).
        eps: Regularization factor.

    Returns:
        float: Total loss value.
    """
    linear_loss = calculate_linear_nmse(i_sim, i_target, eps)
    log_loss = calculate_log_nmse(i_sim, i_target, floor, eps)

    return float(gamma * linear_loss + (1.0 - gamma) * log_loss)
