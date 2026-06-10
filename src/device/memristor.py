"""Hardware-aware memristor device model with non-idealities.

This module implements the MemristorDevice class, which wraps a physics-based
compact model and introduces device-to-device (D2D) variation, cycle-to-cycle (C2C)
variation, conductance/resistance drift, and thermal/shot noise.
"""

from typing import Any

import numpy as np

from compact_models.base import MemristorCompactModel
from utils.config_loader import Config, load_config
from utils.logger import get_logger

logger = get_logger("device")


class MemristorDevice:
    """Represents a hardware-aware memristor device with physical non-idealities."""

    # Physical constants
    K_B = 1.380649e-23  # Boltzmann constant (J/K)
    Q = 1.60217663e-19  # Electron charge (C)

    def __init__(
        self,
        model_class: type[MemristorCompactModel],
        base_params: dict[str, Any],
        device_config: dict[str, Any] | Config | None = None,
        w_init: float | None = None,
        temperature: float | None = None,
    ) -> None:
        """Initializes the MemristorDevice.

        Args:
            model_class: The compact model class (e.g., VTEAMModel).
            base_params: Nominal parameter dictionary for the compact model.
            device_config: Config dictionary or object representing device non-idealities.
            w_init: Initial state variable value. If None, uses model default.
            temperature: Operating temperature (K). If None, reads from config.
        """
        # Resolve configurations
        if device_config is None:
            try:
                full_config = load_config()
                self.config = Config(full_config.raw.get("device", {}))
            except Exception:
                self.config = Config({})
        elif isinstance(device_config, Config):
            self.config = device_config
        else:
            self.config = Config(device_config)

        # Parse D2D config
        d2d_config = self.config.get("d2d", {})
        self.params = self._apply_d2d_variation(base_params, d2d_config)

        # Instantiate compact model with D2D-perturbed parameters
        self.model = model_class(self.params)

        # Set initial state
        w_on = self.params.get("w_on", self.params.get("w_min", 0.0))
        w_off = self.params.get("w_off", self.params.get("w_max", 1.0))
        if w_init is not None:
            self.w = np.clip(w_init, self.model.w_min, self.model.w_max)
        else:
            self.w = w_on  # Default to ON state

        self.w_init_val = self.w
        self.w_programmed = self.w

        # Noise & Temperature parameters
        self.temperature = temperature or self.config.get("noise.temperature", 300.0)
        self.bandwidth = self.config.get("noise.bandwidth", 1.0e6)
        self.thermal_noise_enabled = self.config.get("noise.thermal", True)
        self.shot_noise_enabled = self.config.get("noise.shot", True)
        self.generic_noise_std = self.config.get("noise.generic_std", 0.0)

        # C2C parameters
        self.c2c_enabled = self.config.get("c2c.enabled", True)
        self.c2c_state_noise_std = self.config.get("c2c.state_noise_std", 0.02)
        self.c2c_param_noise_std = self.config.get("c2c.parameter_noise_std", 0.0)

        # Drift parameters
        self.drift_enabled = self.config.get("drift.enabled", True)
        self.drift_coeff = self.config.get("drift.coeff", 0.05)
        self.drift_t_zero = self.config.get("drift.t_zero", 1.0)
        self.drift_type = self.config.get("drift.type", "resistance").lower()
        self.programming_threshold = self.config.get("drift.programming_threshold", 0.5)

        # State drift relaxed state (typically HRS)
        self.w_relaxed = self.config.get("drift.w_relaxed", w_off)

        # Tracking variables
        self.time_since_programming = 0.0
        self.last_v = 0.0

    def _apply_d2d_variation(
        self, base_params: dict[str, Any], d2d_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Applies spatial device-to-device variation to base parameters.

        Args:
            base_params: Nominal parameter dictionary.
            d2d_config: Device-to-Device variation configuration dictionary.

        Returns:
            Dict[str, Any]: Perturbed parameters dictionary.
        """
        if not d2d_config.get("enabled", True):
            return base_params.copy()

        params = base_params.copy()
        parameters_to_perturb = d2d_config.get("parameters", {})

        for param_name, opt in parameters_to_perturb.items():
            if param_name not in params:
                continue

            dist = opt.get("dist", "gaussian").lower()
            std = opt.get("std", 0.0)

            if std <= 0.0:
                continue

            val = params[param_name]

            if dist == "lognormal":
                # Lognormal multiplier with mean=1.0: E[exp(z)] = exp(mu + 0.5*sigma^2)
                # Setting mu = -0.5 * std^2 makes the expected value exactly 1.0
                multiplier = np.exp(np.random.normal(0.0, std) - 0.5 * (std**2))
                val_perturbed = val * multiplier
            else:  # gaussian
                multiplier = 1.0 + np.random.normal(0.0, std)
                val_perturbed = val * multiplier

            params[param_name] = val_perturbed

        # Enforce physical bounds for parameters
        for r_key in ["r_on", "r_off", "R_on", "R_off", "ron", "roff"]:
            if r_key in params:
                params[r_key] = max(params[r_key], 1.0)  # at least 1 Ohm

        # Vset/Vreset thresholds
        for v_key in ["v_on", "V_on", "von"]:
            if v_key in params:
                params[v_key] = max(params[v_key], 1.0e-3)
        for v_key in ["v_off", "V_off", "voff"]:
            if v_key in params:
                params[v_key] = min(params[v_key], -1.0e-3)

        # Rate constants polarity
        if "k_on" in params:
            if base_params["k_on"] < 0:
                params["k_on"] = min(params["k_on"], -1.0e-10)
            else:
                params["k_on"] = max(params["k_on"], 1.0e-10)
        if "k_off" in params:
            if base_params["k_off"] < 0:
                params["k_off"] = min(params["k_off"], -1.0e-10)
            else:
                params["k_off"] = max(params["k_off"], 1.0e-10)

        return params

    def reset(self) -> None:
        """Resets the device to its initial state."""
        self.w = self.w_init_val
        self.w_programmed = self.w
        self.time_since_programming = 0.0
        self.last_v = 0.0

    def step(self, v: float, dt: float) -> float:
        """Steps the device state forward by dt under applied voltage v.

        Args:
            v: The applied voltage (V).
            dt: The time step (s).

        Returns:
            float: The noisy device current (A).
        """
        self.last_v = v

        if dt <= 0:
            return self.current(v)

        # Check if voltage is programming voltage (above threshold)
        is_programming = abs(v) >= self.programming_threshold

        if is_programming:
            self.time_since_programming = 0.0

            # Calculate derivative from physics compact model
            dw = self.model.deriv(v, self.w)

            # Cycle-to-cycle state diffusion (Langevin diffusion term)
            c2c_state_noise = 0.0
            if self.c2c_enabled and self.c2c_state_noise_std > 0.0:
                # Scale noise standard deviation by sqrt(dt) for diffusion process
                c2c_state_noise = np.random.normal(0.0, self.c2c_state_noise_std) * np.sqrt(dt)

            self.w = self.w + dw * dt + c2c_state_noise
            self.w = np.clip(self.w, self.model.w_min, self.model.w_max)
            self.w_programmed = self.w
        else:
            self.time_since_programming += dt

            # If state drift is active, drift the state variable itself
            if self.drift_enabled and self.drift_type == "state":
                # Relax state w towards the relaxed state (typically HRS)
                decay = (1.0 + self.time_since_programming / self.drift_t_zero) ** (
                    -self.drift_coeff
                )
                self.w = self.w_relaxed + (self.w_programmed - self.w_relaxed) * decay
                self.w = np.clip(self.w, self.model.w_min, self.model.w_max)

        # Calculate current
        return self.current(v)

    def current(self, v: float) -> float:
        """Calculates current through the device at current state self.w, including noise.

        Args:
            v: Applied voltage (V).

        Returns:
            float: Device current (A).
        """
        # Clean current from compact model
        i_clean = self.model.current(v, self.w)
        i_active = i_clean

        # Apply resistance drift
        if (
            self.drift_enabled
            and self.drift_type == "resistance"
            and self.time_since_programming > 0.0
        ):
            drift_mult = (1.0 + self.time_since_programming / self.drift_t_zero) ** self.drift_coeff
            i_active = i_clean / drift_mult

        # Apply electrical noise
        # Compute effective resistance for Johnson noise calculations
        r_on = self.params.get("r_on", self.params.get("R_on", 1.0e3))
        r_off = self.params.get("r_off", self.params.get("R_off", 1.0e5))

        # Linearly interpolate resistance to estimate effective resistance at state w
        w_norm = (self.w - self.model.w_min) / (self.model.w_max - self.model.w_min + 1e-20)
        r_eff = r_on + (r_off - r_on) * w_norm
        r_eff = max(r_eff, 1.0)  # Safe guard against <= 0

        # Noise variance in current (A^2)
        variance = 0.0

        if self.thermal_noise_enabled:
            # S_I = 4 * k_B * T / R
            variance += (4.0 * self.K_B * self.temperature * self.bandwidth) / r_eff

        if self.shot_noise_enabled:
            # S_I = 2 * q * |I|
            variance += 2.0 * self.Q * abs(i_active) * self.bandwidth

        # Add generic noise std (standard deviation in A, squared to get variance)
        variance += self.generic_noise_std**2

        # Sample noise
        if variance > 0.0:
            noise = np.random.normal(0.0, np.sqrt(variance))
            return float(i_active + noise)

        return float(i_active)

    def solve_sweep(
        self,
        voltage_sweep: np.ndarray,
        time_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulates the device response to a transient voltage sweep.

        Args:
            voltage_sweep: Array of applied voltage values over time.
            time_points: Array of time points corresponding to the voltage sweep.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (state_history, current_history)
        """
        num_steps = len(time_points)
        w_hist = np.zeros(num_steps)
        i_hist = np.zeros(num_steps)

        # Initialize
        self.reset()
        w_hist[0] = self.w
        i_hist[0] = self.current(voltage_sweep[0])

        for k in range(num_steps - 1):
            t_curr, t_next = time_points[k], time_points[k + 1]
            dt = t_next - t_curr
            v_next = voltage_sweep[k + 1]

            # Step device forward
            i_hist[k + 1] = self.step(v_next, dt)
            w_hist[k + 1] = self.w

        return w_hist, i_hist
