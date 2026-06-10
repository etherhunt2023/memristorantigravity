"""Abstract base class for memristor physics-based compact models.

This module defines the MemristorCompactModel class, providing common interfaces
and numerical integration solvers for dynamic state variable simulations.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class MemristorCompactModel(ABC):
    """Abstract base class for memristor compact models."""

    def __init__(self, params: dict[str, Any], w_min: float, w_max: float) -> None:
        """Initializes the MemristorCompactModel.

        Args:
            params: Dictionary of model parameters.
            w_min: Minimum physical boundary for state variable.
            w_max: Maximum physical boundary for state variable.
        """
        self.params = params
        self.w_min = w_min
        self.w_max = w_max

    @abstractmethod
    def deriv(self, v: float | np.ndarray, w: float | np.ndarray) -> float | np.ndarray:
        """Calculates the derivative of the state variable (dw/dt).

        Args:
            v: The applied voltage (V).
            w: The state variable value.

        Returns:
            float | np.ndarray: Rate of change of the state variable.
        """
        pass

    @abstractmethod
    def current(self, v: float | np.ndarray, w: float | np.ndarray) -> float | np.ndarray:
        """Calculates the current (I) through the memristor.

        Args:
            v: The applied voltage (V).
            w: The state variable value.

        Returns:
            float | np.ndarray: The device current (A).
        """
        pass

    def solve_sweep(
        self,
        voltage_sweep: np.ndarray,
        time_points: np.ndarray,
        w_init: float,
        solver_type: str = "rk4",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulates the dynamic state variable response to a voltage sweep.

        Args:
            voltage_sweep: Array of applied voltage values over time.
            time_points: Array of time points corresponding to the voltage sweep.
            w_init: Initial value of the state variable.
            solver_type: Ordinary Differential Equation solver ('euler' or 'rk4').

        Returns:
            Tuple[np.ndarray, np.ndarray]: (state_history, current_history)
        """
        num_steps = len(time_points)
        w_hist = np.zeros(num_steps)
        i_hist = np.zeros(num_steps)

        # Initialize
        w_hist[0] = np.clip(w_init, self.w_min, self.w_max)
        i_hist[0] = self.current(voltage_sweep[0], w_hist[0])

        for k in range(num_steps - 1):
            t_curr, t_next = time_points[k], time_points[k + 1]
            dt = t_next - t_curr

            if dt <= 0:
                # No time step, keep state constant
                w_hist[k + 1] = w_hist[k]
                i_hist[k + 1] = self.current(voltage_sweep[k + 1], w_hist[k + 1])
                continue

            v_curr, v_next = voltage_sweep[k], voltage_sweep[k + 1]
            w_curr = w_hist[k]

            if solver_type.lower() == "euler":
                # Euler Forward integration step
                dw = self.deriv(v_curr, w_curr)
                w_next = w_curr + dt * dw

            elif solver_type.lower() == "rk4":
                # Runge-Kutta 4th Order integration step
                v_half = 0.5 * (v_curr + v_next)

                k1 = self.deriv(v_curr, w_curr)
                k2 = self.deriv(v_half, w_curr + 0.5 * dt * k1)
                k3 = self.deriv(v_half, w_curr + 0.5 * dt * k2)
                k4 = self.deriv(v_next, w_curr + dt * k3)

                w_next = w_curr + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            else:
                raise ValueError(f"Unknown solver type: {solver_type}")

            # Enforce hard boundaries
            w_hist[k + 1] = np.clip(w_next, self.w_min, self.w_max)
            i_hist[k + 1] = self.current(voltage_sweep[k + 1], w_hist[k + 1])

        return w_hist, i_hist
