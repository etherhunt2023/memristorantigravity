"""Unit tests for the Spiking Neural Network (SNN) simulator.

Verifies LIFNeuron integration, spiking thresholds, refractory periods,
CrossbarSynapse layer operations, and network spike propagation.
"""

import numpy as np
import pytest

from compact_models.vteam import VTEAMModel
from crossbar.array import CrossbarArray
from snn.neuron import LIFNeurons
from snn.synapse import CrossbarSynapse


@pytest.fixture
def base_vteam_params() -> dict[str, float]:
    """Provides base VTEAM parameters for testing."""
    return {
        "w_on": 0.0,
        "w_off": 1.0,
        "v_on": 0.8,
        "v_off": -0.8,
        "k_on": -10.0,
        "k_off": 10.0,
        "alpha_on": 3.0,
        "alpha_off": 3.0,
        "r_on": 1000.0,
        "r_off": 100000.0,
        "d": 3.0e-9,
        "p": 4.0,
    }


def test_lif_neuron_integration_and_spike() -> None:
    """Tests membrane integration, leak decay, threshold firing, and resets."""
    # 1. Integration and leak
    neurons = LIFNeurons(
        num_neurons=2, v_thresh=1.0, leak=0.9, r_membrane=1000.0, t_refractory=0.01
    )

    # Apply input current to neuron 0 only
    i_in = np.array([1.5e-3, 0.0])  # 1.5 mA to neuron 0, 0 to neuron 1

    # Step forward by 1ms
    spikes = neurons.step(i_in, dt=1.0e-3)
    assert not spikes[0]
    assert not spikes[1]

    # Membrane voltage of neuron 0 should rise, neuron 1 should remain 0
    # expected: v_rest + (v_init - v_rest)*leak + i_in * r_membrane * (1 - leak)
    # v0: 0 + 0 + 1.5e-3 * 1000 * (1 - 0.9) = 1.5 * 0.1 = 0.15 V
    assert np.allclose(neurons.v[0], 0.15)
    assert np.allclose(neurons.v[1], 0.0)

    # Let it leak with zero current
    neurons.step(np.zeros(2), dt=1.0e-3)
    # expected v0: 0.15 * 0.9 = 0.135 V
    assert np.allclose(neurons.v[0], 0.135)

    # 2. Trigger spike
    # Apply large current to trigger immediate spike
    i_spike = np.array([5.0e-3, 0.0])  # 5 mA
    spikes = neurons.step(i_spike, dt=1.0e-3)
    # expected v0 before check: 0.135 * 0.9 + 5.0 * 0.1 = 0.1215 + 0.5 = 0.6215 V
    # Next step:
    spikes = neurons.step(i_spike, dt=1.0e-3)
    # expected v0 before check: 0.6215 * 0.9 + 5.0 * 0.1
    # = 0.55935 + 0.5 = 1.05935 V >= 1.0 -> spikes!
    assert spikes[0]
    assert not spikes[1]

    # Membrane should reset to reset potential (0.0) after spike
    assert np.allclose(neurons.v[0], 0.0)
    assert neurons.refractory_timers[0] > 0.0


def test_lif_neuron_refractory_period() -> None:
    """Tests that neurons remain unresponsive to input currents during refractory period."""
    neurons = LIFNeurons(
        num_neurons=1, v_thresh=1.0, leak=0.9, r_membrane=1000.0, t_refractory=0.01
    )

    # Force refractory state manually
    neurons.refractory_timers[0] = 0.005  # 5ms refractory period

    # Apply input current
    i_in = np.array([2.0e-3])
    neurons.step(i_in, dt=1.0e-3)

    # Voltage should remain at reset/rest potential (0.0) because neuron is refractory
    assert np.allclose(neurons.v[0], 0.0)
    assert np.allclose(neurons.refractory_timers[0], 0.004)


def test_crossbar_synapse_forward(base_vteam_params: dict[str, float]) -> None:
    """Tests forward propagation of spikes in CrossbarSynapse."""
    cb = CrossbarArray(
        rows=4,
        cols=4,
        model_class=VTEAMModel,
        base_params=base_vteam_params,
        device_config={"d2d": {"enabled": False}},
    )

    # Program cell (0, 0) to LRS (high weight) and others to HRS (low weight)
    for i in range(4):
        for j in range(4):
            cb.devices[i, j].w = 1.0
    cb.devices[0, 0].w = 0.0

    synapse = CrossbarSynapse(cb, v_pulse=0.5)

    # Spike on row 0 only
    pre_spikes = np.array([True, False, False, False])

    # 1. Test idealized mode
    i_out_ideal = synapse.forward(pre_spikes, use_mna=False, dt=1.0e-3)
    assert i_out_ideal.shape == (4,)
    assert i_out_ideal[0] > i_out_ideal[1]  # LRS column output should be higher

    # 2. Test MNA mode
    i_out_mna = synapse.forward(pre_spikes, use_mna=True, dt=1.0e-3)
    assert i_out_mna.shape == (4,)
    assert i_out_mna[0] > i_out_mna[1]


def test_transient_spike_propagation(base_vteam_params: dict[str, float]) -> None:
    """Verifies end-to-end spike propagation from input spikes to post-synaptic firing."""
    cb = CrossbarArray(
        rows=2,
        cols=2,
        model_class=VTEAMModel,
        base_params=base_vteam_params,
        device_config={"d2d": {"enabled": False}},
    )

    # Set all weights to LRS (w = 0.0, R = 1k, G = 1mS)
    for i in range(2):
        for j in range(2):
            cb.devices[i, j].w = 0.0

    synapse = CrossbarSynapse(cb, v_pulse=1.0)
    neurons = LIFNeurons(
        num_neurons=2, v_thresh=0.2, leak=0.9, r_membrane=1000.0, t_refractory=0.01
    )

    # Apply continuous pre-synaptic spikes at each step
    pre_spikes = np.array([True, True])

    spike_detected = False
    for _ in range(5):
        # 1. Forward pass through synapse
        i_post = synapse.forward(pre_spikes, use_mna=False, dt=1.0e-3)

        # 2. Step neurons
        post_spikes = neurons.step(i_post, dt=1.0e-3)
        if np.any(post_spikes):
            spike_detected = True
            break

    # Spikes should propagate and eventually fire the post-synaptic neurons
    assert spike_detected
