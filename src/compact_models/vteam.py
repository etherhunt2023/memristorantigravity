"""VTEAM (Voltage Threshold Adaptive Memristor) compact model.

This model is a threshold-based switching model for bipolar memristive devices,
where state changes occur only when the applied voltage exceeds thresholds.
"""

from typing import Any

import numpy as np

from compact_models.base import MemristorCompactModel
from compact_models.windows import biolek_window


class VTEAMModel(MemristorCompactModel):
    """VTEAM Memristor Compact Model."""

    def __init__(self, params: dict[str, Any]) -> None:
        """Initializes the VTEAMModel.

        Args:
            params: Parameters dictionary containing:
                    - w_on: physical ON-state limit (m or normalized)
                    - w_off: physical OFF-state limit (m or normalized)
                    - v_on: threshold voltage for SET (V)
                    - v_off: threshold voltage for RESET (V)
                    - k_on: SET rate constant (1/s)
                    - k_off: RESET rate constant (1/s)
                    - alpha_on: SET voltage exponent
                    - alpha_off: RESET voltage exponent
                    - r_on: LRS resistance (Ohms)
                    - r_off: HRS resistance (Ohms)
                    - p: Biolek window parameter
        """
        w_min = min(params["w_on"], params["w_off"])
        w_max = max(params["w_on"], params["w_off"])
        super().__init__(params, w_min, w_max)

    def deriv(self, v: float | np.ndarray, w: float | np.ndarray) -> float | np.ndarray:
        """Calculates the state variable derivative (dw/dt).

        Args:
            v: The applied voltage (V).
            w: The state variable value.

        Returns:
            float | np.ndarray: Rate of change of the state variable.
        """
        # Normalize w for the Biolek window
        w_norm = (w - self.w_min) / (self.w_max - self.w_min + 1e-20)

        # Exponent for window function
        p = self.params.get("p", 4)
        f_win = biolek_window(w_norm, v, p)

        v_on = self.params["v_on"]
        v_off = self.params["v_off"]
        k_on = self.params["k_on"]
        k_off = self.params["k_off"]
        alpha_on = self.params["alpha_on"]
        alpha_off = self.params["alpha_off"]

        # Support array inputs
        if isinstance(v, np.ndarray):
            dw = np.zeros_like(v)

            # SET transition (V > v_on)
            set_mask = v > v_on
            if np.any(set_mask):
                dw[set_mask] = k_on * ((v[set_mask] / v_on) - 1.0) ** alpha_on * f_win[set_mask]

            # RESET transition (V < v_off)
            reset_mask = v < v_off
            if np.any(reset_mask):
                dw[reset_mask] = (
                    k_off * ((v[reset_mask] / v_off) - 1.0) ** alpha_off * f_win[reset_mask]
                )

            return dw
        else:
            if v > v_on:
                return k_on * ((v / v_on) - 1.0) ** alpha_on * f_win
            elif v < v_off:
                return k_off * ((v / v_off) - 1.0) ** alpha_off * f_win
            else:
                return 0.0

    def current(self, v: float | np.ndarray, w: float | np.ndarray) -> float | np.ndarray:
        """Calculates the current (I) through the memristor.

        R(w) = R_on + (R_off - R_on) * (w - w_on) / (w_off - w_on)
        I(V, w) = V / R(w)

        Args:
            v: The applied voltage (V).
            w: The state variable value.

        Returns:
            float | np.ndarray: The device current (A).
        """
        w_on = self.params["w_on"]
        w_off = self.params["w_off"]
        r_on = self.params["r_on"]
        r_off = self.params["r_off"]

        # Calculate resistance by linear interpolation
        r_w = r_on + (r_off - r_on) * (w - w_on) / (w_off - w_on + 1e-20)

        # Safety check to avoid division by zero
        r_w = np.clip(r_w, min(r_on, r_off), max(r_on, r_off))

        return v / r_w
