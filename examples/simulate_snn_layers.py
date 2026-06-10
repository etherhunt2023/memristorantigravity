"""Example script demonstrating a Spiking Neural Network (SNN) layer.

Connects input spikes through a memristive crossbar synapse to a population of
LIF neurons, simulates transient spike propagation, and plots the spike rasters
and membrane potential dynamics.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compact_models.vteam import VTEAMModel
from crossbar.array import CrossbarArray
from snn.neuron import LIFNeurons
from snn.synapse import CrossbarSynapse


def generate_poisson_spikes(
    rate_hz: float, duration_s: float, dt_s: float, shape: int
) -> np.ndarray:
    """Generates a Poisson spike train.

    Args:
        rate_hz: Average firing rate in Hz.
        duration_s: Total duration of the spike train (s).
        dt_s: Time step size (s).
        shape: Firing pattern shape (number of spike channels).

    Returns:
        np.ndarray: 2D boolean array of shape (timesteps x shape) indicating spike events.
    """
    timesteps = int(duration_s / dt_s)
    # Probability of spike in time step dt is rate * dt
    prob = rate_hz * dt_s
    spikes = np.random.rand(timesteps, shape) < prob
    return spikes


def run_snn_simulation() -> None:
    """Simulates SNN layer and plots inputs, membrane voltages, and output spikes."""
    print("==================================================")
    print("SIMULATING MEMRISTOR SNN LAYER DYNAMICS")
    print("==================================================")

    # Base parameters for VTEAM
    base_params = {
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

    # Simulation time configuration
    dt = 1.0e-3  # 1 ms time step
    duration_phase = 0.05  # 50 ms per phase
    total_duration = duration_phase * 2.0
    timesteps = int(total_duration / dt)

    # Network topology: 8 inputs -> 4 output neurons
    num_inputs = 8
    num_outputs = 4

    # 1. Generate Poisson input spikes
    np.random.seed(42)  # for reproducible plots
    print("Generating Poisson spike trains...")

    # Phase 1 (0-50ms): Rows 0-3 spike high (70Hz), Rows 4-7 spike low (5Hz)
    spikes_p1_high = generate_poisson_spikes(70.0, duration_phase, dt, shape=4)
    spikes_p1_low = generate_poisson_spikes(5.0, duration_phase, dt, shape=4)
    spikes_p1 = np.hstack([spikes_p1_high, spikes_p1_low])

    # Phase 2 (50-100ms): Rows 0-3 spike low (5Hz), Rows 4-7 spike high (70Hz)
    spikes_p2_low = generate_poisson_spikes(5.0, duration_phase, dt, shape=4)
    spikes_p2_high = generate_poisson_spikes(70.0, duration_phase, dt, shape=4)
    spikes_p2 = np.hstack([spikes_p2_low, spikes_p2_high])

    # Concatenate phases
    input_spikes = np.vstack([spikes_p1, spikes_p2])

    # 2. Instantiate SNN Components
    crossbar_config = {
        "line_resistance": 1.5,
        "source_resistance": 50.0,
        "load_resistance": 50.0,
    }

    # Create Crossbar
    cb = CrossbarArray(
        rows=num_inputs,
        cols=num_outputs,
        model_class=VTEAMModel,
        base_params=base_params,
        device_config={"d2d": {"enabled": False}},
        crossbar_config=crossbar_config,
    )

    # Set up Synapse and LIF Neurons
    synapse = CrossbarSynapse(cb, v_pulse=0.8)  # 0.8V pulses
    neurons = LIFNeurons(
        num_neurons=num_outputs,
        v_thresh=0.8,  # 0.8 V threshold
        leak=0.95,  # leak constant
        r_membrane=2.0e3,  # 2k Ohms membrane resistance
        t_refractory=3.0e-3,  # 3ms refractory period
    )

    # Program weights to establish selective connectivity:
    # Outputs 0-1 connect to Rows 0-3 (LRS w=0.0);
    # Outputs 2-3 connect to Rows 4-7 (LRS w=0.0).
    for i in range(num_inputs):
        for j in range(num_outputs):
            cb.devices[i, j].w = 1.0  # nominal HRS

    # Connect Rows 0-3 to Outputs 0 & 1
    for i in range(4):
        cb.devices[i, 0].w = 0.0
        cb.devices[i, 1].w = 0.0
    # Connect Rows 4-7 to Outputs 2 & 3
    for i in range(4, 8):
        cb.devices[i, 2].w = 0.0
        cb.devices[i, 3].w = 0.0

    # 3. SNN Transient Simulation Loop
    print("Running transient simulation loop...")
    v_history = np.zeros((timesteps, num_outputs))
    output_spikes = np.zeros((timesteps, num_outputs), dtype=bool)

    neurons.reset()
    synapse.reset()

    for t_step in range(timesteps):
        pre_s = input_spikes[t_step]

        # Forward pass: propagate spikes to get post-synaptic currents
        i_post = synapse.forward(pre_s, use_mna=True, dt=dt)

        # Step neurons membrane potential
        post_s = neurons.step(i_post, dt=dt)

        # Record histories
        v_history[t_step] = neurons.v
        output_spikes[t_step] = post_s

    # 4. Plot Results
    temp_dir = Path(__file__).resolve().parent / "temp_data"
    temp_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    time_ms = np.arange(timesteps) * dt * 1000.0

    # Top: Input spikes raster plot
    for neuron_idx in range(num_inputs):
        spike_times = time_ms[input_spikes[:, neuron_idx]]
        axes[0].scatter(
            spike_times,
            np.full_like(spike_times, neuron_idx),
            marker="|",
            color="royalblue",
            s=40,
        )
    axes[0].axvline(50, color="gray", linestyle="--", alpha=0.7)
    axes[0].set_ylabel("Input Index", fontsize=11, fontweight="bold")
    axes[0].set_title("Input Spike Rasters (Poisson)", fontsize=13, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # Middle: Membrane potential traces
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd"]
    for neuron_idx in range(num_outputs):
        axes[1].plot(
            time_ms,
            v_history[:, neuron_idx],
            color=colors[neuron_idx],
            label=f"Neuron {neuron_idx}",
            linewidth=1.8,
        )
    axes[1].axvline(50, color="gray", linestyle="--", alpha=0.7)
    axes[1].axhline(0.8, color="black", linestyle=":", alpha=0.8, label="Vthreshold")
    axes[1].set_ylabel("Membrane Voltage (V)", fontsize=11, fontweight="bold")
    axes[1].set_title("Post-synaptic Neuron Membrane Potentials", fontsize=13, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="upper right")

    # Bottom: Output spikes raster plot
    for neuron_idx in range(num_outputs):
        spike_times = time_ms[output_spikes[:, neuron_idx]]
        axes[2].scatter(
            spike_times,
            np.full_like(spike_times, neuron_idx),
            marker="|",
            color=colors[neuron_idx],
            s=50,
            linewidth=2.0,
        )
    axes[2].axvline(50, color="gray", linestyle="--", alpha=0.7)
    axes[2].set_ylabel("Neuron Index", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Time (ms)", fontsize=11, fontweight="bold")
    axes[2].set_title("Output Spike Rasters (LIF Firing)", fontsize=13, fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    output_path = temp_dir / "snn_simulation.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(
        f"\nSimulation complete! SNN dynamics plot saved to:\n[snn_simulation.png](file:///{output_path.as_posix()})"
    )
    print("==================================================\n")


if __name__ == "__main__":
    run_snn_simulation()
