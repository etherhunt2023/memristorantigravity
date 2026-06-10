"""Example script demonstrating the parameter fitting engine.

This script generates a synthetic noisy memristor I-V sweep, fits the VTEAM model
to it using the two-stage ModelFitter, prints a comparison of the target and
optimized parameters, and saves the comparison plot.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compact_models.vteam import VTEAMModel
from comsol.dataset import COMSOLDataset
from fitting.fitter import ModelFitter
from utils.config_loader import Config, load_config
from utils.logger import setup_logger

# Setup logging
logger = setup_logger(log_level="INFO")


def generate_noisy_target_data(config: Config) -> COMSOLDataset:
    """Generates synthetic VTEAM target data with added noise.

    Args:
        config: Config object representing parameters.

    Returns:
        COMSOLDataset: Synthetic dataset.
    """
    v_fwd = np.linspace(0.0, 1.5, 51)
    v_bwd = np.linspace(1.5, -1.5, 101)
    v_ret = np.linspace(-1.5, 0.0, 51)
    v_sweep = np.concatenate([v_fwd, v_bwd, v_ret])
    t_points = np.linspace(0, 1e-3, len(v_sweep))

    # Target parameters
    vteam_params = config.get("compact_model.vteam").copy()
    # Let's specify exact targets
    vteam_params["k_on"] = -25.0
    vteam_params["k_off"] = 25.0
    vteam_params["r_on"] = 1200.0
    vteam_params["r_off"] = 80000.0

    target_model = VTEAMModel(vteam_params)
    _, i_target = target_model.solve_sweep(
        v_sweep, t_points, w_init=vteam_params["w_off"], solver_type="rk4"
    )

    # Add Gaussian noise (log-space noise to represent fluctuations)
    log_i = np.log10(np.abs(i_target) + 1e-12)
    noise = np.random.normal(0, 0.05, len(log_i))
    i_noisy = np.sign(i_target) * (10 ** (log_i + noise))

    df = pd.DataFrame({"time": t_points, "voltage": v_sweep, "current": i_noisy})
    return COMSOLDataset(df, {"target_k_on": -25.0, "target_k_off": 25.0})


def run_example() -> None:
    """Runs the parameter fitting example."""
    logger.info("Starting compact model parameter fitting example...")

    # Load config
    config = load_config()

    # Generate synthetic target I-V sweep data
    logger.info("Generating target simulation data...")
    dataset = generate_noisy_target_data(config)

    # Instantiate fitter
    fitter = ModelFitter(config)

    # Read bounds from config
    bounds = config.get("compact_model.fitting.vteam_bounds")

    # Define fixed parameters
    vteam_defaults = config.get("compact_model.vteam")
    fixed_params = {
        "w_on": vteam_defaults["w_on"],
        "w_off": vteam_defaults["w_off"],
        "v_on": vteam_defaults["v_on"],
        "v_off": vteam_defaults["v_off"],
        "alpha_on": vteam_defaults["alpha_on"],
        "alpha_off": vteam_defaults["alpha_off"],
        "d": vteam_defaults["d"],
        "p": vteam_defaults["p"],
    }

    # Parameters to optimize
    fit_keys = ["k_on", "k_off", "r_on", "r_off"]

    # Run fitting
    logger.info("Running optimization...")
    fitted_params = fitter.fit(
        dataset=dataset,
        model_class=VTEAMModel,
        fit_param_keys=fit_keys,
        bounds_dict=bounds,
        fixed_params=fixed_params,
        w_init=vteam_defaults["w_off"],
        solver_type="rk4",
    )

    # Print results comparison
    print("\n================ Parameter Fitting Results ================")
    print(f"Target k_on:  -25.00        | Fitted k_on:  {fitted_params['k_on']:.4f}")
    print(f"Target k_off:  25.00        | Fitted k_off: {fitted_params['k_off']:.4f}")
    print(f"Target r_on:   1200.00 Ohms | Fitted r_on:  {fitted_params['r_on']:.2f} Ohms")
    print(f"Target r_off:  80000.00 Ohms| Fitted r_off: {fitted_params['r_off']:.2f} Ohms")
    print("===========================================================\n")

    # Simulate with fitted parameters for comparison
    fitted_model = VTEAMModel(fitted_params)
    v_sweep = dataset.data["voltage"].to_numpy()
    t_points = dataset.data["time"].to_numpy()
    _, i_fitted = fitted_model.solve_sweep(
        v_sweep, t_points, w_init=fitted_params["w_off"], solver_type="rk4"
    )

    # Plot comparisons
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(v_sweep, dataset.data["current"] * 1e6, "ro", alpha=0.3, label="Noisy Target (COMSOL)")
    ax.plot(v_sweep, i_fitted * 1e6, "b-", linewidth=2.5, label="Fitted VTEAM Model")
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (µA)")
    ax.set_title("Memristor I-V Loop: Target vs. Fitted Compact Model")
    ax.legend()
    ax.grid(True)

    # Save comparison figure
    temp_dir = Path(__file__).resolve().parent / "temp_data"
    temp_dir.mkdir(exist_ok=True)
    fig_path = temp_dir / "fitted_model_comparison.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    logger.info(f"Fitted model comparison figure saved to {fig_path}")


if __name__ == "__main__":
    run_example()
