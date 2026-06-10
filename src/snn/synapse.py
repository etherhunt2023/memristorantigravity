"""Crossbar-based synaptic layers connecting SNN neuron populations.

This module provides the CrossbarSynapse class, which uses a CrossbarArray
instance to perform synaptic weight multiplication (current summation) under
ideal or parasitic hardware-aware conditions.
"""

import numpy as np

from crossbar.array import CrossbarArray
from utils.logger import get_logger

logger = get_logger("snn")


class CrossbarSynapse:
    """Connects pre- and post-synaptic neurons via a memristive crossbar array."""

    def __init__(self, crossbar: CrossbarArray, v_pulse: float = 1.0) -> None:
        """Initializes the CrossbarSynapse.

        Args:
            crossbar: The CrossbarArray simulator representing the synaptic weights.
            v_pulse: The voltage amplitude representing an active spike pulse (V).
        """
        self.crossbar = crossbar
        self.v_pulse = v_pulse

        logger.info(
            f"Initialized CrossbarSynapse: size {crossbar.rows}x{crossbar.cols}, "
            f"spike pulse amplitude = {v_pulse}V"
        )

    def reset(self) -> None:
        """Resets the state of all devices in the underlying crossbar array."""
        self.crossbar.reset()

    def forward(
        self, pre_spikes: np.ndarray, use_mna: bool = True, dt: float = 1.0e-3
    ) -> np.ndarray:
        """Propagates pre-synaptic spikes to produce post-synaptic currents.

        Converts pre-synaptic spikes into row voltages and computes column currents
        using either idealized or parasitic MNA solver. Stepps device states.

        Args:
            pre_spikes: 1D boolean array indicating active row spikes (shape: rows).
            use_mna: If True, uses the Modified Nodal Analysis solver with wire parasitics.
                     If False, uses idealized virtual ground vector-matrix multiplication.
            dt: Time step duration (s).

        Returns:
            np.ndarray: 1D array of output currents entering the post-synaptic nodes (shape: cols).
        """
        # Convert pre-synaptic spike events to input voltages (V)
        row_voltages = np.array(pre_spikes, dtype=float) * self.v_pulse

        if use_mna:
            # Solve using full Modified Nodal Analysis with wire resistances,
            # which steps all the devices dynamically.
            i_out = self.crossbar.step(row_voltages, col_voltages=None, dt=dt)
        else:
            # Idealized mode: assume columns are held at virtual ground (0V)
            # 1. Retrieve current conductance matrix
            g_matrix = np.zeros((self.crossbar.rows, self.crossbar.cols))
            for i in range(self.crossbar.rows):
                for j in range(self.crossbar.cols):
                    dev = self.crossbar.devices[i, j]
                    r_on = dev.params.get("r_on", dev.params.get("R_on", 1.0e3))
                    r_off = dev.params.get("r_off", dev.params.get("R_off", 1.0e5))
                    w_norm = (dev.w - dev.model.w_min) / (dev.model.w_max - dev.model.w_min + 1e-20)
                    r_eff = r_on + (r_off - r_on) * w_norm
                    g_matrix[i, j] = 1.0 / max(r_eff, 1.0)

            # 2. Compute post-synaptic currents: I_out = G_matrix.T @ V_row
            i_out = g_matrix.T @ row_voltages

            # 3. Step all devices under local idealized voltages (V_drop = V_row_input)
            for i in range(self.crossbar.rows):
                for j in range(self.crossbar.cols):
                    self.crossbar.devices[i, j].step(row_voltages[i], dt)

        return i_out
