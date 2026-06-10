"""Example script demonstrating hardware-aware memristor device non-idealities.

Simulates device-to-device (D2D) spatial variation, cycle-to-cycle (C2C)
stochastic switching, and resistance power-law drift over time.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compact_models.vteam import VTEAMModel
from device.memristor import MemristorDevice


def run_device_nonideality_simulation() -> None:
    """Runs the simulations and plots the non-idealities."""
    print("==================================================")
    print("SIMULATING HARDWARE-AWARE MEMRISTOR NON-IDEALITIES")
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

    # Create figure directory
    temp_dir = Path(__file__).resolve().parent / "temp_data"
    temp_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # -------------------------------------------------------------
    # 1. Device-to-Device (D2D) Spatial Variation
    # -------------------------------------------------------------
    print("Part 1: Simulating Device-to-Device (D2D) spatial variation...")
    d2d_config = {
        "d2d": {
            "enabled": True,
            "parameters": {
                "r_on": {"dist": "lognormal", "std": 0.15},
                "r_off": {"dist": "lognormal", "std": 0.15},
                "v_on": {"dist": "gaussian", "std": 0.05},
                "v_off": {"dist": "gaussian", "std": 0.05},
            },
        },
        "c2c": {"enabled": False},
        "drift": {"enabled": False},
        "noise": {"thermal": False, "shot": False, "generic_std": 0.0},
    }

    # Triangular voltage sweep
    t = np.linspace(0, 4, 400)
    # 0V -> 1.5V -> -1.5V -> 0V
    v = np.zeros_like(t)
    v[t < 1] = 1.5 * t[t < 1]
    v[(t >= 1) & (t < 3)] = 1.5 - 1.5 * (t[(t >= 1) & (t < 3)] - 1)
    v[t >= 3] = -1.5 + 1.5 * (t[t >= 3] - 3)

    num_devices = 40
    ax_d2d = axes[0]
    for _ in range(num_devices):
        dev = MemristorDevice(VTEAMModel, base_params, device_config=d2d_config, w_init=1.0)
        _, i_hist = dev.solve_sweep(v, t)
        ax_d2d.plot(v, i_hist * 1e3, color="teal", alpha=0.25)

    # Plot nominal device for reference
    nom_config = {
        "d2d": {"enabled": False},
        "c2c": {"enabled": False},
        "drift": {"enabled": False},
        "noise": {"thermal": False, "shot": False, "generic_std": 0.0},
    }
    dev_nom = MemristorDevice(VTEAMModel, base_params, device_config=nom_config, w_init=1.0)
    _, i_nom = dev_nom.solve_sweep(v, t)
    ax_d2d.plot(v, i_nom * 1e3, color="red", linewidth=2.0, label="Nominal Device")

    ax_d2d.set_title("Device-to-Device (D2D) Variation", fontsize=12, fontweight="bold")
    ax_d2d.set_xlabel("Applied Voltage (V)", fontsize=10)
    ax_d2d.set_ylabel("Current (mA)", fontsize=10)
    ax_d2d.grid(True, linestyle="--", alpha=0.5)
    ax_d2d.legend()

    # -------------------------------------------------------------
    # 2. Cycle-to-Cycle (C2C) Temporal Variation
    # -------------------------------------------------------------
    print("Part 2: Simulating Cycle-to-Cycle (C2C) stochastic switching...")
    c2c_config = {
        "d2d": {"enabled": False},
        "c2c": {
            "enabled": True,
            # Artificially high state noise to show clear stochastic cycles
            "state_noise_std": 0.15,
        },
        "drift": {"enabled": False},
        "noise": {
            "thermal": True,
            "shot": True,
            "bandwidth": 1.0e6,
            "temperature": 300.0,
            "generic_std": 1.0e-6,
        },
    }

    # Let's run a single device through 5 consecutive sweeps
    dev_c2c = MemristorDevice(VTEAMModel, base_params, device_config=c2c_config, w_init=1.0)
    ax_c2c = axes[1]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for cycle in range(5):
        _, i_hist = dev_c2c.solve_sweep(v, t)
        ax_c2c.plot(v, i_hist * 1e3, color=colors[cycle], alpha=0.7, label=f"Cycle {cycle + 1}")

    ax_c2c.set_title("Cycle-to-Cycle (C2C) & Noise", fontsize=12, fontweight="bold")
    ax_c2c.set_xlabel("Applied Voltage (V)", fontsize=10)
    ax_c2c.set_ylabel("Current (mA)", fontsize=10)
    ax_c2c.grid(True, linestyle="--", alpha=0.5)
    ax_c2c.legend()

    # -------------------------------------------------------------
    # 3. Conductance & Resistance Drift (Power-law relaxation)
    # -------------------------------------------------------------
    print("Part 3: Simulating resistance power-law drift...")
    drift_config = {
        "d2d": {"enabled": False},
        "c2c": {"enabled": False},
        "drift": {
            "enabled": True,
            "coeff": 0.08,  # Drift exponent nu
            "t_zero": 1.0,
            "type": "resistance",
            "programming_threshold": 0.5,
        },
        "noise": {"thermal": False, "shot": False, "generic_std": 0.0},
    }

    # Setup device, program to LRS (w = 0.0, base resistance = 1000 ohms)
    dev_drift = MemristorDevice(VTEAMModel, base_params, device_config=drift_config, w_init=0.0)

    # Let's measure resistance over log-spaced time points (1s to 1000s)
    times = np.logspace(0, 3, 100)
    resistances = []

    # Reset device drift state
    dev_drift.reset()
    # Step at time = 0 to program it
    dev_drift.step(0.1, 0.0)

    # Measure resistance at subsequent time steps
    last_t = 0.0
    for t_pt in times:
        dt = t_pt - last_t
        # Measure current at small read voltage (0.1V) below programming threshold
        i_read = dev_drift.step(0.1, dt)
        r_measured = 0.1 / i_read
        resistances.append(r_measured)
        last_t = t_pt

    ax_drift = axes[2]
    ax_drift.loglog(
        times, resistances, color="darkorange", linewidth=2.5, label="Drifted Resistance"
    )

    # Plot expected analytical drift line R(t) = R0 * (t/t0)^nu
    expected_r = base_params["r_on"] * ((times / 1.0) ** 0.08)
    ax_drift.loglog(
        times,
        expected_r,
        color="black",
        linestyle="--",
        alpha=0.7,
        label=r"Theoretical ($t^{0.08}$)",
    )

    ax_drift.set_title("Resistance Drift (Power Law)", fontsize=12, fontweight="bold")
    ax_drift.set_xlabel("Time (s)", fontsize=10)
    ax_drift.set_ylabel("Resistance (Ohms)", fontsize=10)
    ax_drift.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_drift.legend()

    plt.tight_layout()
    output_path = temp_dir / "device_nonidealities.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(
        f"\nSimulation complete! Non-ideality plots saved to:\n[device_nonidealities.png](file:///{output_path.as_posix()})"
    )
    print("==================================================\n")


if __name__ == "__main__":
    run_device_nonideality_simulation()
