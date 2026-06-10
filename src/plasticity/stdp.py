"""Spike-Timing-Dependent Plasticity (STDP) models for memristor synapses.

This module provides phenomenological (analytical) and pulse-based physical
STDP weight update rules, allowing conversion of spike timing differences
into memristor conductance/state changes.
"""

import numpy as np

from device.memristor import MemristorDevice
from utils.config_loader import Config, load_config
from utils.logger import get_logger

logger = get_logger("plasticity")


class PhenomenologicalSTDP:
    """Analytical exponential STDP learning model."""

    def __init__(
        self,
        a_plus: float | None = None,
        a_minus: float | None = None,
        tau_plus: float | None = None,
        tau_minus: float | None = None,
        update_type: str | None = None,
    ) -> None:
        """Initializes the PhenomenologicalSTDP model.

        Args:
            a_plus: Amplitude of potentiation. If None, reads from config.
            a_minus: Amplitude of depression. If None, reads from config.
            tau_plus: Potentiation time constant (s). If None, reads from config.
            tau_minus: Depression time constant (s). If None, reads from config.
            update_type: Weight update scheme ("additive" or "multiplicative").
        """
        try:
            full_config = load_config()
            stdp_config = Config(full_config.raw.get("plasticity", {}).get("stdp", {}))
        except Exception:
            stdp_config = Config({})

        self.a_plus = a_plus if a_plus is not None else float(stdp_config.get("a_plus", 0.05))
        self.a_minus = a_minus if a_minus is not None else float(stdp_config.get("a_minus", 0.03))
        self.tau_plus = (
            tau_plus if tau_plus is not None else float(stdp_config.get("tau_plus", 20.0e-3))
        )
        self.tau_minus = (
            tau_minus if tau_minus is not None else float(stdp_config.get("tau_minus", 20.0e-3))
        )
        self.update_type = (
            update_type
            if update_type is not None
            else stdp_config.get("update_type", "multiplicative")
        ).lower()

        logger.info(
            f"Initialized PhenomenologicalSTDP (update_type={self.update_type}): "
            f"A+={self.a_plus}, A-={self.a_minus}, tau+={self.tau_plus}s, tau-={self.tau_minus}s"
        )

    def calculate_delta_w(self, delta_t: float | np.ndarray) -> float | np.ndarray:
        """Calculates the analytical change in synaptic weight (dw).

        dw > 0 indicates potentiation (increase in conductance, i.e., decrease in state w).
        dw < 0 indicates depression (decrease in conductance, i.e., increase in state w).

        Args:
            delta_t: Spike timing difference t_post - t_pre (s).

        Returns:
            float | np.ndarray: Analytical weight change.
        """
        if isinstance(delta_t, np.ndarray):
            dw = np.zeros_like(delta_t)
            pos_mask = delta_t > 0.0
            neg_mask = delta_t < 0.0

            # Potentiation (t_post > t_pre) -> positive weight change
            dw[pos_mask] = self.a_plus * np.exp(-delta_t[pos_mask] / self.tau_plus)
            # Depression (t_post < t_pre) -> negative weight change
            dw[neg_mask] = -self.a_minus * np.exp(delta_t[neg_mask] / self.tau_minus)
            return dw
        else:
            if delta_t > 0.0:
                return self.a_plus * np.exp(-delta_t / self.tau_plus)
            elif delta_t < 0.0:
                return -self.a_minus * np.exp(delta_t / self.tau_minus)
            return 0.0

    def apply_stdp(self, device: MemristorDevice, delta_t: float) -> float:
        """Applies STDP update to a MemristorDevice state.

        Args:
            device: The MemristorDevice to update.
            delta_t: Spike timing difference t_post - t_pre (s).

        Returns:
            float: New device state variable value.
        """
        dw = self.calculate_delta_w(delta_t)

        w_old = device.w
        w_min = device.model.w_min
        w_max = device.model.w_max

        # Potentiation (dw > 0) should decrease w (towards w_min/LRS)
        # Depression (dw < 0) should increase w (towards w_max/HRS)
        if self.update_type == "additive":
            # Additive: dw is subtracted directly from state
            w_new = w_old - dw
        elif self.update_type == "multiplicative":
            # Multiplicative (state-dependent):
            if dw > 0.0:  # Potentiation
                w_new = w_old - dw * (w_old - w_min)
            elif dw < 0.0:  # Depression
                w_new = w_old - dw * (w_max - w_old)
            else:
                w_new = w_old
        else:
            raise ValueError(f"Unknown update type: {self.update_type}")

        # Apply state update and hard clipping
        device.w = np.clip(w_new, w_min, w_max)
        device.w_programmed = device.w
        device.time_since_programming = 0.0

        return float(device.w)


class PulseBasedSTDP:
    """Pulse-based physical STDP model using overlapping voltage waveforms."""

    def __init__(
        self,
        v_pre_amp: float | None = None,
        v_post_amp: float | None = None,
        tau_pulse: float | None = None,
        pulse_duration: float | None = None,
    ) -> None:
        """Initializes the PulseBasedSTDP model.

        Args:
            v_pre_amp: Amplitude of pre-synaptic pulse. If None, reads from config.
            v_post_amp: Amplitude of post-synaptic pulse. If None, reads from config.
            tau_pulse: Time constant for exponential decay. If None, reads from config.
            pulse_duration: Total duration simulated (s). If None, reads from config.
        """
        try:
            full_config = load_config()
            pb_config = Config(
                full_config.raw.get("plasticity", {}).get("stdp", {}).get("pulse_based", {})
            )
        except Exception:
            pb_config = Config({})

        self.v_pre_amp = (
            v_pre_amp if v_pre_amp is not None else float(pb_config.get("v_pre_amp", 1.2))
        )
        self.v_post_amp = (
            v_post_amp if v_post_amp is not None else float(pb_config.get("v_post_amp", 1.2))
        )
        self.tau_pulse = (
            tau_pulse if tau_pulse is not None else float(pb_config.get("tau_pulse", 15.0e-3))
        )
        self.pulse_duration = (
            pulse_duration
            if pulse_duration is not None
            else float(pb_config.get("pulse_duration", 50.0e-3))
        )

        logger.info(
            f"Initialized PulseBasedSTDP: V_pre={self.v_pre_amp}V, V_post={self.v_post_amp}V, "
            f"tau={self.tau_pulse}s, duration={self.pulse_duration}s"
        )

    def generate_waveform(self, delta_t: float, time_points: np.ndarray) -> np.ndarray:
        """Generates the overlapping net voltage waveform V_pre - V_post.

        Pre-synaptic spike occurs at the center of the pulse duration window.
        Post-synaptic spike occurs at t_pre + delta_t.

        Args:
            delta_t: Spike timing difference t_post - t_pre (s).
            time_points: Array of time points corresponding to simulation.

        Returns:
            np.ndarray: Waveform array of net voltage drop over time.
        """
        t_pre = self.pulse_duration / 2.0
        t_post = t_pre + delta_t

        # Pre-synaptic voltage pulse (positive pulse on row)
        v_pre = np.zeros_like(time_points)
        mask_pre = time_points >= t_pre
        v_pre[mask_pre] = self.v_pre_amp * np.exp(-(time_points[mask_pre] - t_pre) / self.tau_pulse)

        # Post-synaptic voltage pulse (positive pulse on column, subtracted from row)
        v_post = np.zeros_like(time_points)
        mask_post = time_points >= t_post
        v_post[mask_post] = self.v_post_amp * np.exp(
            -(time_points[mask_post] - t_post) / self.tau_pulse
        )

        # Net voltage drop V_pre - V_post
        return v_pre - v_post

    def apply_stdp(self, device: MemristorDevice, delta_t: float, num_steps: int = 200) -> float:
        """Applies STDP update by solving memristor dynamics under overlapping waveform.

        Args:
            device: The MemristorDevice to simulate and update.
            delta_t: Spike timing difference t_post - t_pre (s).
            num_steps: Number of simulation time steps.

        Returns:
            float: New device state variable value.
        """
        # Generate time sweep
        time_points = np.linspace(0.0, self.pulse_duration, num_steps)
        v_net = self.generate_waveform(delta_t, time_points)

        # Solve device transient dynamics (automatically updates device.w)
        device.solve_sweep(v_net, time_points)

        return float(device.w)
