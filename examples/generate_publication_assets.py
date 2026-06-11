"""Example script demonstrating the generation of publication-quality figures and tables.

Applies IEEE styling configurations, simulates representative memristor and crossbar
data, generates publication-ready vector (PDF/SVG) and raster (PNG) assets, and
formats LaTeX/Markdown tables ready for integration into research manuscripts.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from visualization.plotter import (
    apply_journal_style,
    plot_crossbar_heatmaps,
    plot_fitting_results,
    plot_iv_sweep,
    plot_snn_activity,
)
from visualization.tables import format_fitting_metrics_table, format_parameter_table


def generate_assets() -> None:
    """Generates all publication assets (plots and tables)."""
    print("==================================================")
    print("GENERATING PUBLICATION-QUALITY ASSETS")
    print("==================================================")

    # 1. Apply journal plotting style
    print("Applying IEEE journal visual style configurations...")
    apply_journal_style("ieee")

    # Set up output directories
    temp_dir = Path(__file__).resolve().parent / "temp_data"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 2. Asset: Memristor I-V Sweep Curves (Linear & Log scale)
    print("Simulating and plotting I-V hysteresis loop...")
    voltages = np.concatenate(
        [
            np.linspace(0.0, 1.2, 50),
            np.linspace(1.2, -1.2, 100),
            np.linspace(-1.2, 0.0, 50),
        ]
    )
    # Hysteresis loop model: higher conductance on backward path
    conductance = np.zeros_like(voltages)
    g_lrs = 1.0e-3
    g_hrs = 1.0e-5

    # Simple threshold-based switching for mock plot
    state = 0.0  # 0: HRS, 1: LRS
    for i, v in enumerate(voltages):
        if v > 0.8:
            state = 1.0
        elif v < -0.8:
            state = 0.0

        # Soft transition
        conductance[i] = g_hrs + (g_lrs - g_hrs) * state

    currents = voltages * conductance

    # Plot Linear I-V Hysteresis
    fig_iv_linear = plot_iv_sweep(
        voltages, currents, title="Memristor I-V Loop (Linear Scale)", log_scale=False
    )
    for ext in ["png", "pdf", "svg"]:
        fig_iv_linear.savefig(temp_dir / f"publication_iv_linear.{ext}", dpi=300)
    plt.close(fig_iv_linear)

    # Plot Log I-V Hysteresis
    fig_iv_log = plot_iv_sweep(
        voltages, currents, title="Memristor I-V Loop (Logarithmic Scale)", log_scale=True
    )
    for ext in ["png", "pdf", "svg"]:
        fig_iv_log.savefig(temp_dir / f"publication_iv_log.{ext}", dpi=300)
    plt.close(fig_iv_log)

    # 3. Asset: Parameter Fitting residual analysis
    print("Plotting parameter fitting comparison and residuals...")
    # Add noise to simulate raw experimental data
    noise = np.random.normal(0, 2.0e-5, len(currents))
    experimental_currents = currents + noise
    fitted_currents = currents + np.random.normal(0, 5.0e-6, len(currents))

    fig_fitting = plot_fitting_results(voltages, experimental_currents, fitted_currents)
    for ext in ["png", "pdf", "svg"]:
        fig_fitting.savefig(temp_dir / f"publication_fitting.{ext}", dpi=300)
    plt.close(fig_fitting)

    # 4. Asset: Crossbar Voltage drop heatmap (8x8)
    print("Plotting crossbar junction voltage drop heatmaps...")
    # Simulate parasitic voltage drop along rows and columns
    crossbar_data = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            # Degradation from top-left (1.0V) to bottom-right (0.4V)
            crossbar_data[i, j] = 1.0 - 0.04 * i - 0.035 * j

    fig_crossbar = plot_crossbar_heatmaps(
        crossbar_data,
        title="Junction Voltages with Line Parasitics",
        label="Junction Voltage V_j (V)",
    )
    for ext in ["png", "pdf", "svg"]:
        fig_crossbar.savefig(temp_dir / f"publication_crossbar_voltages.{ext}", dpi=300)
    plt.close(fig_crossbar)

    # 5. Asset: SNN Transient Spiking activity
    print("Plotting SNN multi-panel spike rasters and membrane potential traces...")
    time_ms = np.linspace(0, 100, 200)
    # Mock spiking activity
    input_spikes = np.zeros((200, 8), dtype=bool)
    # poisson spikes
    input_spikes[np.random.rand(200, 8) < 0.15] = True

    # Output LIF membrane potentials
    mem_voltages = np.zeros((200, 3))
    # Integrate and fire
    v = np.zeros(3)
    output_spikes = np.zeros((200, 3), dtype=bool)
    v_thresh = 0.8
    for t_idx in range(200):
        # Accumulate current from inputs
        active_inputs = input_spikes[t_idx].sum()
        v += 0.08 * active_inputs
        # Leakage
        v *= 0.95
        # Threshold crossings
        spikes = v >= v_thresh
        output_spikes[t_idx] = spikes
        mem_voltages[t_idx] = v
        # Reset
        v[spikes] = 0.0

    fig_snn = plot_snn_activity(
        time_ms, input_spikes, mem_voltages, output_spikes, v_thresh=v_thresh
    )
    for ext in ["png", "pdf", "svg"]:
        fig_snn.savefig(temp_dir / f"publication_snn_activity.{ext}", dpi=300)
    plt.close(fig_snn)

    # 6. Asset: Parameter Tables (LaTeX booktabs and Markdown formats)
    print("Formatting and exporting LaTeX and Markdown tables...")
    parameters = {
        "w_on": 0.0,
        "w_off": 1.0e-9,
        "v_on": 0.8015,
        "v_off": -0.7981,
        "k_on": -10.245,
        "k_off": 9.8732,
        "r_on": 1050.2,
        "r_off": 98450.0,
    }
    descriptions = {
        "w_on": "Minimum state variable boundary",
        "w_off": "Maximum state variable boundary",
        "v_on": "Positive set voltage threshold (V)",
        "v_off": "Negative reset voltage threshold (V)",
        "k_on": "ON-state switching rate constant",
        "k_off": "OFF-state switching rate constant",
        "r_on": "Low-Resistance State (LRS) resistance (Ohms)",
        "r_off": "High-Resistance State (HRS) resistance (Ohms)",
    }
    errors = {
        "w_on": 0.0,
        "w_off": 0.0,
        "v_on": 0.0042,
        "v_off": 0.0051,
        "k_on": 0.124,
        "k_off": 0.098,
        "r_on": 12.5,
        "r_off": 945.0,
    }

    # LaTeX parameter table
    latex_param_table = format_parameter_table(
        parameters, descriptions=descriptions, errors=errors, table_format="latex"
    )
    with (temp_dir / "parameter_table.tex").open("w", encoding="utf-8") as f:
        f.write(latex_param_table)

    # Markdown parameter table
    md_param_table = format_parameter_table(
        parameters, descriptions=descriptions, errors=errors, table_format="markdown"
    )
    with (temp_dir / "parameter_table.md").open("w", encoding="utf-8") as f:
        f.write(md_param_table)

    # LaTeX Fitting evaluation metrics table
    metrics = {
        "Mean Squared Error (MSE)": 2.4589e-9,
        "Root Mean Squared Error (RMSE)": 4.9587e-5,
        "R-squared (R2)": 0.998471,
        "Mean Absolute Error (MAE)": 3.1248e-5,
    }
    latex_metrics_table = format_fitting_metrics_table(metrics, table_format="latex")
    with (temp_dir / "fitting_metrics.tex").open("w", encoding="utf-8") as f:
        f.write(latex_metrics_table)

    print("\nAssets generated successfully!")
    print(f"All assets saved to: {temp_dir.as_posix()}")
    print("\n--- PNG/PDF/SVG Plots ---")
    print(f"- [Linear I-V loop](file:///{temp_dir.as_posix()}/publication_iv_linear.png)")
    print(f"- [Log I-V loop](file:///{temp_dir.as_posix()}/publication_iv_log.png)")
    print(f"- [Fitting results](file:///{temp_dir.as_posix()}/publication_fitting.png)")
    print(f"- [Crossbar heatmap](file:///{temp_dir.as_posix()}/publication_crossbar_voltages.png)")
    print(f"- [SNN activity](file:///{temp_dir.as_posix()}/publication_snn_activity.png)")
    print("\n--- LaTeX/Markdown Tables ---")
    print(f"- [LaTeX parameters (.tex)](file:///{temp_dir.as_posix()}/parameter_table.tex)")
    print(f"- [Markdown parameters (.md)](file:///{temp_dir.as_posix()}/parameter_table.md)")
    print(f"- [LaTeX metrics (.tex)](file:///{temp_dir.as_posix()}/fitting_metrics.tex)")
    print("==================================================\n")


if __name__ == "__main__":
    generate_assets()
