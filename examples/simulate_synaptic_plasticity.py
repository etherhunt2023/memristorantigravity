"""Example script simulating Spike-Timing-Dependent Plasticity (STDP).

Sweeps relative spike timings (t_post - t_pre) and evaluates both
phenomenological (analytical) and pulse-based physical STDP updates,
plotting the learning curves and overlapping pulse waveforms.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compact_models.vteam import VTEAMModel
from device.memristor import MemristorDevice
from plasticity.stdp import PhenomenologicalSTDP, PulseBasedSTDP


def run_plasticity_simulation() -> None:
    """Runs STDP sweeps and plots comparison curves."""
    print("==================================================")
    print("SIMULATING SPIKE-TIMING-DEPENDENT PLASTICITY (STDP)")
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

    # Spike timing differences to sweep (-80ms to +80ms)
    delta_ts = np.linspace(-0.08, 0.08, 100)

    # -------------------------------------------------------------
    # 1. Phenomenological STDP Curve
    # -------------------------------------------------------------
    print("Part 1: Simulating Phenomenological (Analytical) STDP...")
    stdp_ana = PhenomenologicalSTDP(
        a_plus=0.08, a_minus=0.06, tau_plus=0.02, tau_minus=0.02, update_type="multiplicative"
    )

    dw_ana = stdp_ana.calculate_delta_w(delta_ts)

    # -------------------------------------------------------------
    # 2. Pulse-Based Physical STDP Curve
    # -------------------------------------------------------------
    print("Part 2: Simulating Pulse-Based Physical STDP...")
    pb = PulseBasedSTDP(v_pre_amp=1.2, v_post_amp=1.2, tau_pulse=0.015, pulse_duration=0.06)

    # We instantiate a device for the physical sweep
    # w_init = 0.5 (middle state)
    device = MemristorDevice(
        VTEAMModel,
        base_params,
        device_config={
            "d2d": {"enabled": False},
            "c2c": {"enabled": False},
            "drift": {"enabled": False},
        },
        w_init=0.5,
    )

    dw_physical = []
    for dt in delta_ts:
        device.reset()
        pb.apply_stdp(device, dt, num_steps=200)
        # Weight change is proportional to conductance change
        # delta_w = w_initial - w_final (since decreasing w increases conductance/weight)
        dw_phys = 0.5 - device.w
        dw_physical.append(dw_phys)

    # -------------------------------------------------------------
    # 3. Waveform Overlap Visualization
    # -------------------------------------------------------------
    print("Part 3: Generating waveform overlap examples...")
    time_pts = np.linspace(0.0, 0.06, 300)
    v_overlap_pos = pb.generate_waveform(0.005, time_pts)  # dt = +5ms (potentiation)
    v_overlap_neg = pb.generate_waveform(-0.005, time_pts)  # dt = -5ms (depression)
    v_overlap_far = pb.generate_waveform(0.04, time_pts)  # dt = +40ms (no switching)

    # Save results
    temp_dir = Path(__file__).resolve().parent / "temp_data"
    temp_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Phenomenological STDP
    axes[0].plot(delta_ts * 1e3, dw_ana, color="blue", linewidth=2.5, label="Analytical Model")
    axes[0].axvline(0, color="black", linestyle="--", alpha=0.5)
    axes[0].axhline(0, color="black", linestyle="--", alpha=0.5)
    axes[0].set_title("Phenomenological STDP Curve", fontsize=12, fontweight="bold")
    axes[0].set_xlabel(r"Spike Timing Difference $\Delta t$ (ms)", fontsize=10)
    axes[0].set_ylabel(r"Weight Change $\Delta w$", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Plot 2: Pulse-based STDP
    axes[1].plot(
        delta_ts * 1e3, dw_physical, color="green", linewidth=2.5, label="Physical Overlap"
    )
    axes[1].axvline(0, color="black", linestyle="--", alpha=0.5)
    axes[1].axhline(0, color="black", linestyle="--", alpha=0.5)
    axes[1].set_title("Pulse-Based Physical STDP", fontsize=12, fontweight="bold")
    axes[1].set_xlabel(r"Spike Timing Difference $\Delta t$ (ms)", fontsize=10)
    axes[1].set_ylabel(r"State Change ($w_{init} - w_{final}$)", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    # Plot 3: Overlapping waveforms
    axes[2].plot(time_pts * 1e3, v_overlap_pos, color="red", label=r"$\Delta t = +5$ ms (Set)")
    axes[2].plot(time_pts * 1e3, v_overlap_neg, color="purple", label=r"$\Delta t = -5$ ms (Reset)")
    axes[2].plot(time_pts * 1e3, v_overlap_far, color="gray", linestyle="--", label="No Overlap")
    # Draw switching thresholds
    axes[2].axhline(0.8, color="black", linestyle=":", alpha=0.7, label="Vset Threshold")
    axes[2].axhline(-0.8, color="black", linestyle=":", alpha=0.7, label="Vreset Threshold")
    axes[2].set_title("Net Voltage Waveforms $V_{pre} - V_{post}$", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Time (ms)", fontsize=10)
    axes[2].set_ylabel("Voltage (V)", fontsize=10)
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend()

    plt.tight_layout()
    output_path = temp_dir / "stdp_curves.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(
        f"\nSimulation complete! STDP plots saved to:\n[stdp_curves.png](file:///{output_path.as_posix()})"
    )
    print("==================================================\n")


if __name__ == "__main__":
    run_plasticity_simulation()
