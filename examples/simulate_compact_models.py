"""Example script simulating physics-based compact models.

This script loads default configurations, instantiates VTEAM, Yakopcic, and
Simmons models, simulates their dynamic state and current responses under a
triangular I-V sweep, and saves the publication-quality visualization.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compact_models.simmons import SimmonsModel
from compact_models.vteam import VTEAMModel
from compact_models.yakopcic import YakopcicModel
from utils.config_loader import load_config
from utils.logger import setup_logger

# Setup logging
logger = setup_logger(log_level="INFO")


def run_example() -> None:
    """Runs the compact models simulation example."""
    logger.info("Starting compact models simulation example...")

    # Load configuration
    config = load_config()

    # Create triangular voltage sweep: 0 -> 2V -> -2V -> 0V
    t_points = np.linspace(0, 2e-3, 2001)  # 2 ms duration
    v_sweep = np.zeros_like(t_points)

    # Segment the sweep
    idx1 = int(0.25 * len(t_points))
    idx2 = int(0.75 * len(t_points))

    v_sweep[:idx1] = np.linspace(0.0, 2.0, idx1)
    v_sweep[idx1:idx2] = np.linspace(2.0, -2.0, idx2 - idx1)
    v_sweep[idx2:] = np.linspace(-2.0, 0.0, len(t_points) - idx2)

    # 1. Simulate VTEAM Model
    logger.info("Simulating VTEAM model...")
    vteam_params = config.get("compact_model.vteam")
    vteam = VTEAMModel(vteam_params)
    w_vteam, i_vteam = vteam.solve_sweep(
        v_sweep, t_points, w_init=vteam_params["w_off"], solver_type="rk4"
    )

    # 2. Simulate Yakopcic Model
    logger.info("Simulating Yakopcic model...")
    yakopcic_params = config.get("compact_model.yakopcic")
    # Add thresholds for SET/RESET in Yakopcic
    yakopcic_params["vp"] = 1.0
    yakopcic_params["vn"] = 1.0
    yakopcic = YakopcicModel(yakopcic_params)
    w_yakopcic, i_yakopcic = yakopcic.solve_sweep(v_sweep, t_points, w_init=0.1, solver_type="rk4")

    # 3. Simulate Simmons Model
    logger.info("Simulating Simmons Tunneling Barrier model...")
    simmons_params = config.get("compact_model.simmons")
    simmons = SimmonsModel(simmons_params)
    w_simmons, i_simmons = simmons.solve_sweep(
        v_sweep, t_points, w_init=simmons_params["w_off"], solver_type="rk4"
    )

    # Visualize results
    logger.info("Plotting simulation sweeps...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

    # VTEAM plots
    axes[0, 0].plot(t_points * 1e3, v_sweep, "r-", label="Voltage")
    axes[0, 0].set_ylabel("Voltage (V)")
    axes[0, 0].set_title("VTEAM applied sweep")
    axes[0, 0].grid(True)

    tw1 = axes[0, 0].twinx()
    tw1.plot(t_points * 1e3, w_vteam, "b--", label="State w")
    tw1.set_ylabel("State variable w", color="b")

    axes[1, 0].plot(v_sweep, i_vteam * 1e6, "k-")
    axes[1, 0].set_xlabel("Voltage (V)")
    axes[1, 0].set_ylabel("Current (µA)")
    axes[1, 0].set_title("VTEAM hysteretic I-V loop")
    axes[1, 0].grid(True)

    # Yakopcic plots
    axes[0, 1].plot(t_points * 1e3, v_sweep, "r-")
    axes[0, 1].set_title("Yakopcic applied sweep")
    axes[0, 1].grid(True)

    tw2 = axes[0, 1].twinx()
    tw2.plot(t_points * 1e3, w_yakopcic, "g--")
    tw2.set_ylabel("State variable w", color="g")

    axes[1, 1].plot(v_sweep, i_yakopcic * 1e3, "k-")
    axes[1, 1].set_xlabel("Voltage (V)")
    axes[1, 1].set_ylabel("Current (mA)")
    axes[1, 1].set_title("Yakopcic hysteretic I-V loop")
    axes[1, 1].grid(True)

    # Simmons plots
    axes[0, 2].plot(t_points * 1e3, v_sweep, "r-")
    axes[0, 2].set_title("Simmons applied sweep")
    axes[0, 2].grid(True)

    tw3 = axes[0, 2].twinx()
    tw3.plot(t_points * 1e3, w_simmons * 1e9, "c--")
    tw3.set_ylabel("Gap width w (nm)", color="c")

    axes[1, 2].plot(v_sweep, i_simmons * 1e6, "k-")
    axes[1, 2].set_xlabel("Voltage (V)")
    axes[1, 2].set_ylabel("Current (µA)")
    axes[1, 2].set_title("Simmons hysteretic I-V loop")
    axes[1, 2].grid(True)

    # Save output plot
    output_dir = Path(__file__).resolve().parent / "temp_data"
    output_dir.mkdir(exist_ok=True)
    fig_path = output_dir / "compact_model_hysteresis.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    logger.info(f"Dynamic I-V hysteresis figure saved to {fig_path}")


if __name__ == "__main__":
    run_example()
