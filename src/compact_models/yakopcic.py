"""Yakopcic memristor compact model.

This model fits asymmetric switching behaviors using exponential thresholds
and hyperbolic sine (sinh) current formulations.
"""

from typing import Any

import numpy as np

from compact_models.base import MemristorCompactModel


class YakopcicModel(MemristorCompactModel):
    """Yakopcic Memristor Compact Model."""

    def __init__(self, params: dict[str, Any]) -> None:
        """Initializes the YakopcicModel.

        Args:
            params: Parameters dictionary containing:
                    - a1: positive bias current scaling (A)
                    - a2: negative bias current scaling (A)
                    - b: current voltage exponent (1/V)
                    - xp: positive threshold state cutoff
                    - xn: negative threshold state cutoff
                    - gp: positive rate scaling coefficient
                    - gn: negative rate scaling coefficient
                    - ap: positive exponential rate factor
                    - an: negative exponential rate factor
                    - eta: state variable direction multiplier (+1 or -1)
                    - vp: positive set threshold voltage (V)
                    - vn: negative reset threshold voltage (V)
        """
        # Yakopcic state variable x is normalized between 0 and 1
        super().__init__(params, w_min=0.0, w_max=1.0)

    def deriv(self, v: float | np.ndarray, w: float | np.ndarray) -> float | np.ndarray:
        """Calculates the state variable derivative (dx/dt).

        Args:
            v: The applied voltage (V).
            w: The state variable value.

        Returns:
            float | np.ndarray: Rate of change of the state variable.
        """
        eta = self.params.get("eta", 1.0)
        vp = self.params.get("vp", 1.0)
        vn = self.params.get("vn", 1.0)
        gp = self.params.get("gp", 1.0)
        gn = self.params.get("gn", 1.0)
        ap = self.params.get("ap", 1.0)
        an = self.params.get("an", 1.0)
        xp = self.params.get("xp", 0.5)
        xn = self.params.get("xn", 0.5)

        # Support array inputs
        if isinstance(v, np.ndarray):
            g_v = np.zeros_like(v)
            f_w = np.zeros_like(v)

            # Calculate g(V)
            pos_mask = v > vp
            neg_mask = v < -vn

            if np.any(pos_mask):
                g_v[pos_mask] = gp * (np.exp(ap * v[pos_mask]) - np.exp(ap * vp))
            if np.any(neg_mask):
                g_v[neg_mask] = -gn * (np.exp(-an * v[neg_mask]) - np.exp(an * vn))

            # Calculate f(w, V) - Yakopcic state limitations
            v_pos = v > 0
            v_neg = v <= 0

            # For positive voltage sweeps (Set)
            idx_pos_high = v_pos & (w >= xp)
            idx_pos_low = v_pos & (w < xp)
            if np.any(idx_pos_high):
                f_w[idx_pos_high] = (xp - w[idx_pos_high]) / (1.0 - xp)
            if np.any(idx_pos_low):
                f_w[idx_pos_low] = 1.0

            # For negative voltage sweeps (Reset)
            idx_neg_low = v_neg & (w <= xn)
            idx_neg_high = v_neg & (w > xn)
            if np.any(idx_neg_low):
                f_w[idx_neg_low] = w[idx_neg_low] / (1.0 - xn)
            if np.any(idx_neg_high):
                f_w[idx_neg_high] = 1.0

            return eta * g_v * f_w
        else:
            # Scalar branch
            if v > vp:
                g_v = gp * (np.exp(ap * v) - np.exp(ap * vp))
            elif v < -vn:
                g_v = -gn * (np.exp(-an * v) - np.exp(an * vn))
            else:
                g_v = 0.0

            if v > 0:
                f_w = (xp - w) / (1.0 - xp) if w >= xp else 1.0
            else:
                f_w = w / (1.0 - xn) if w <= xn else 1.0

            return eta * g_v * f_w

    def current(self, v: float | np.ndarray, w: float | np.ndarray) -> float | np.ndarray:
        """Calculates the current (I) through the memristor.

        I(V, w) = a1 * w * sinh(b * V) for V >= 0
        I(V, w) = a2 * w * sinh(b * V) for V < 0

        Args:
            v: The applied voltage (V).
            w: The state variable value.

        Returns:
            float | np.ndarray: The device current (A).
        """
        a1 = self.params["a1"]
        a2 = self.params["a2"]
        b = self.params["b"]

        if isinstance(v, np.ndarray):
            i = np.zeros_like(v)
            pos = v >= 0
            neg = v < 0
            if np.any(pos):
                i[pos] = a1 * w[pos] * np.sinh(b * v[pos])
            if np.any(neg):
                i[neg] = a2 * w[neg] * np.sinh(b * v[neg])
            return i
        else:
            a = a1 if v >= 0 else a2
            return a * w * np.sinh(b * v)
