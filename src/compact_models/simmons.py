"""Simmons Tunneling Barrier compact model.

This model uses Simmons' tunneling equations to represent transport through a
nanoscale insulator gap and models gap change using ionic migration equations.
"""

from typing import Any

import numpy as np

from compact_models.base import MemristorCompactModel
from compact_models.windows import biolek_window


class SimmonsModel(MemristorCompactModel):
    """Simmons Tunneling Barrier Memristor Compact Model."""

    def __init__(self, params: dict[str, Any]) -> None:
        """Initializes the SimmonsModel.

        Args:
            params: Parameters dictionary containing:
                    - w_on: physical gap limit at SET (m, e.g. 1e-10)
                    - w_off: physical gap limit at RESET (m, e.g. 1.5e-9)
                    - tox: total oxide thickness (m, e.g. 3e-9)
                    - phi_0: barrier height (eV, e.g. 0.95)
                    - area: contact area (m^2, e.g. 1e-12)
                    - k_on: SET ionic migration velocity factor (m/s)
                    - k_off: RESET ionic migration velocity factor (m/s)
                    - v_on: threshold voltage for SET (V)
                    - v_off: threshold voltage for RESET (V)
                    - p: Biolek window parameter
        """
        w_min = min(params["w_on"], params["w_off"])
        w_max = max(params["w_on"], params["w_off"])
        super().__init__(params, w_min, w_max)

    def deriv(self, v: float | np.ndarray, w: float | np.ndarray) -> float | np.ndarray:
        """Calculates the state variable derivative (dw/dt).

        Args:
            v: The applied voltage (V).
            w: The state variable value (m).

        Returns:
            float | np.ndarray: Rate of change of the state variable (m/s).
        """
        w_norm = (w - self.w_min) / (self.w_max - self.w_min + 1e-20)
        p = self.params.get("p", 4)
        f_win = biolek_window(w_norm, v, p)

        v_on = self.params["v_on"]
        v_off = self.params["v_off"]
        k_on = self.params["k_on"]
        k_off = self.params["k_off"]

        # Support array inputs
        if isinstance(v, np.ndarray):
            dw = np.zeros_like(v)

            # SET transition (V > v_on, w decreases towards w_on)
            set_mask = v > v_on
            if np.any(set_mask):
                dw[set_mask] = k_on * (np.exp(v[set_mask] / v_on) - 1.0) * f_win[set_mask]

            # RESET transition (V < v_off, w increases towards w_off)
            reset_mask = v < v_off
            if np.any(reset_mask):
                dw[reset_mask] = k_off * (np.exp(v[reset_mask] / v_off) - 1.0) * f_win[reset_mask]

            return dw
        else:
            if v > v_on:
                return k_on * (np.exp(v / v_on) - 1.0) * f_win
            elif v < v_off:
                return k_off * (np.exp(v / v_off) - 1.0) * f_win
            else:
                return 0.0

    def current(self, v: float | np.ndarray, w: float | np.ndarray) -> float | np.ndarray:
        """Calculates the tunneling current using Simmons' equation.

        Args:
            v: The applied voltage (V).
            w: The tunneling barrier width (m).

        Returns:
            float | np.ndarray: The device current (A).
        """
        # Fundamental physical constants
        e_charge = 1.602176634e-19  # C
        h_planck = 6.62607015e-34  # J*s
        m_electron = 9.1093837e-31 * 0.2  # effective mass (0.2 * m0)

        phi_0 = self.params["phi_0"]  # eV
        area = self.params["area"]  # m^2

        # Convert phi_0 to Joules
        phi_j = phi_0 * e_charge

        v_abs = np.abs(v)
        # Term inside square root: E = phi_j - e * V_abs / 2
        term1_j = phi_j - 0.5 * e_charge * v_abs
        term2_j = phi_j + 0.5 * e_charge * v_abs

        # Bound term1 to avoid negative values
        if isinstance(v_abs, np.ndarray):
            term1_j = np.clip(term1_j, 1e-25, None)
        else:
            term1_j = max(term1_j, 1e-25)

        # Simmons coefficients
        beta = (4.0 * np.pi * np.sqrt(2.0 * m_electron)) / h_planck
        coeff = e_charge / (2.0 * np.pi * h_planck * (w**2))

        # Calculate current density J (A/m^2)
        j_tunnel = coeff * (
            term1_j * np.exp(-beta * w * np.sqrt(term1_j))
            - term2_j * np.exp(-beta * w * np.sqrt(term2_j))
        )

        i_tunnel = j_tunnel * area

        # Restore sign of voltage
        if isinstance(v, np.ndarray):
            return np.where(v >= 0, i_tunnel, -i_tunnel)
        else:
            return i_tunnel if v >= 0 else -i_tunnel
