"""Window functions for memristor compact models.

These functions are used to restrict the state variable to a normalized range
[0, 1] and model non-linear boundaries at the physical limits.
"""

import numpy as np


def joglekar_window(w: float | np.ndarray, p: int = 4) -> float | np.ndarray:
    """Calculates the Joglekar window function.

    f(w) = 1 - (2w - 1)^(2p)

    Args:
        w: Normalized state variable in [0, 1].
        p: Exponent parameter controlling the shape.

    Returns:
        float | np.ndarray: Window value.
    """
    w_clipped = np.clip(w, 0.0, 1.0)
    return 1.0 - (2.0 * w_clipped - 1.0) ** (2 * p)


def biolek_window(w: float | np.ndarray, v: float | np.ndarray, p: int = 4) -> float | np.ndarray:
    """Calculates the Biolek window function.

    f(w, v) = 1 - (w - step(-v))^(2p)
    Solves the boundary lock problem by using an asymmetric voltage dependence.

    Args:
        w: Normalized state variable in [0, 1].
        v: Applied voltage.
        p: Exponent parameter.

    Returns:
        float | np.ndarray: Window value.
    """
    w_clipped = np.clip(w, 0.0, 1.0)
    # step_v represents the target boundary towards which the state moves:
    # If v < 0, state moves towards 1 (w_off).
    # If v > 0, state moves towards 0 (w_on).
    step_v = np.where(v > 0, 1.0, 0.0) if isinstance(v, np.ndarray) else 1.0 if v > 0 else 0.0

    return 1.0 - (w_clipped - step_v) ** (2 * p)


def prodromakis_window(w: float | np.ndarray, p: int = 4, j: float = 1.0) -> float | np.ndarray:
    """Calculates the Prodromakis window function.

    f(w) = j * (1 - [(w - 0.5)^2 + 0.75]^p)
    Enforces stricter boundary controls while allowing scaling.

    Args:
        w: Normalized state variable in [0, 1].
        p: Exponent parameter.
        j: Scaling parameter.

    Returns:
        float | np.ndarray: Window value.
    """
    w_clipped = np.clip(w, 0.0, 1.0)
    return j * (1.0 - ((w_clipped - 0.5) ** 2 + 0.75) ** p)
