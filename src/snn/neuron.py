"""Vectorized Leaky Integrate-and-Fire (LIF) neuron model for SNN simulations.

This module provides the LIFNeurons class, which represents a population of
spiking neurons with leaky membrane dynamics, threshold firing, reset potentials,
and refractory periods.
"""

import numpy as np

from utils.config_loader import Config, load_config
from utils.logger import get_logger

logger = get_logger("snn")


class LIFNeurons:
    """Represents a population of Leaky Integrate-and-Fire (LIF) spiking neurons."""

    def __init__(
        self,
        num_neurons: int,
        v_thresh: float | None = None,
        v_rest: float | None = None,
        v_reset: float | None = None,
        tau_m: float | None = None,
        leak: float | None = None,
        r_membrane: float | None = None,
        t_refractory: float | None = None,
    ) -> None:
        """Initializes the LIFNeurons population.

        Args:
            num_neurons: Number of neurons in this population.
            v_thresh: Action potential threshold (V). If None, reads from config.
            v_rest: Resting membrane potential (V). If None, reads from config.
            v_reset: Reset potential after firing (V). If None, reads from config.
            tau_m: Membrane time constant (s). If None, computed from leak or config.
            leak: Direct membrane leak multiplier (0 to 1). If None, reads from config.
            r_membrane: Membrane resistance (Ohms). Defaults to 10k Ohms.
            t_refractory: Refractory period duration (s). If None, reads from config.
        """
        self.num_neurons = num_neurons

        try:
            full_config = load_config()
            snn_config = Config(full_config.raw.get("snn", {}))
        except Exception:
            snn_config = Config({})

        self.v_thresh = (
            v_thresh if v_thresh is not None else float(snn_config.get("threshold", 1.0))
        )
        self.v_rest = v_rest if v_rest is not None else 0.0
        self.v_reset = v_reset if v_reset is not None else 0.0
        self.r_membrane = r_membrane if r_membrane is not None else 1.0e4  # 10k Ohms
        self.t_refractory = (
            t_refractory
            if t_refractory is not None
            else float(snn_config.get("refractory_period", 2.0e-3))
        )

        # Resolve leak decay constant
        self.tau_m = tau_m
        self.leak = leak if leak is not None else snn_config.get("leak_constant", 0.95)

        # State variables
        self.v = np.full(self.num_neurons, self.v_rest)
        self.refractory_timers = np.zeros(self.num_neurons)

        logger.info(
            f"Initialized LIFNeurons population of size {num_neurons}: "
            f"Vth={self.v_thresh}V, Vrest={self.v_rest}V, Vreset={self.v_reset}V, "
            f"R_mem={self.r_membrane} Ohms, t_ref={self.t_refractory}s"
        )

    def reset(self) -> None:
        """Resets all membrane potentials and refractory timers to baseline values."""
        self.v = np.full(self.num_neurons, self.v_rest)
        self.refractory_timers = np.zeros(self.num_neurons)

    def step(self, i_in: np.ndarray, dt: float) -> np.ndarray:
        """Advances the membrane dynamics by time step dt under input current i_in.

        Args:
            i_in: 1D array of input currents for each neuron (shape: num_neurons).
            dt: Time step duration (s).

        Returns:
            np.ndarray: Boolean mask array of shape (num_neurons,) indicating spiking neurons.
        """
        # Ensure correct input current shape
        if len(i_in) != self.num_neurons:
            raise ValueError(
                f"Input current shape {i_in.shape} does not match neuron count {self.num_neurons}."
            )

        # Update refractory timers
        self.refractory_timers = np.clip(self.refractory_timers - dt, 0.0, None)

        # Select non-refractory neurons to integrate currents
        non_ref = self.refractory_timers == 0.0

        # Calculate voltage decay coefficient
        decay = np.exp(-dt / self.tau_m) if self.tau_m is not None else self.leak

        # Membrane integration:
        # V(t+dt) = V_rest + (V(t) - V_rest) * decay + I_in * R_mem * (1 - decay)
        self.v[non_ref] = (
            self.v_rest
            + (self.v[non_ref] - self.v_rest) * decay
            + i_in[non_ref] * self.r_membrane * (1.0 - decay)
        )

        # Check for threshold crossings (firing action potentials)
        spikes = self.v >= self.v_thresh

        # Reset firing neurons and place them into refractory state
        self.v[spikes] = self.v_reset
        self.refractory_timers[spikes] = self.t_refractory

        return spikes
