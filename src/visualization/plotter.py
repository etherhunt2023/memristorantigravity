"""Matplotlib-based publication-quality plotting utilities for memristive systems.

Configures figure styles to meet requirements for IEEE/Nature journals and provides
functions to plot I-V sweeps, fitting residuals, crossbar heatmaps, and SNN rasters.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def apply_journal_style(style_name: str = "ieee") -> None:
    """Configures matplotlib rcParams for publication-quality figures.

    Args:
        style_name: Name of style template ("ieee" or "nature").
    """
    # Color cycles: Colorblind-friendly, publication standard palettes
    ieee_colors = [
        "#0072B2",  # Blue
        "#D55E00",  # Vermilion (Red-Orange)
        "#009E73",  # Bluish Green
        "#CC79A7",  # Reddish Purple
        "#E69F00",  # Orange
        "#56B4E9",  # Sky Blue
        "#F0E442",  # Yellow
    ]

    nature_colors = [
        "#E64B35FF",  # Red
        "#4DBBD5FF",  # Light Blue
        "#00A087FF",  # Teal
        "#3C5488FF",  # Dark Blue
        "#F39B7FFF",  # Peach
        "#8491B4FF",  # Slate
        "#91D1C2FF",  # Mint
    ]

    colors = ieee_colors if style_name.lower() == "ieee" else nature_colors

    # Configure rcParams
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["text.usetex"] = False  # Avoid dependency on external LaTeX compiler
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.titlesize"] = 10
    plt.rcParams["axes.labelsize"] = 9
    plt.rcParams["xtick.labelsize"] = 8
    plt.rcParams["ytick.labelsize"] = 8
    plt.rcParams["legend.fontsize"] = 8
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.5
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors)
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["axes.linewidth"] = 0.75
    plt.rcParams["xtick.major.width"] = 0.75
    plt.rcParams["ytick.major.width"] = 0.75


def plot_iv_sweep(
    voltages: np.ndarray,
    currents: np.ndarray,
    title: str = "Memristor I-V Hysteresis Loop",
    log_scale: bool = False,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plots a clean, publication-ready I-V hysteresis sweep.

    Uses a color gradient or arrows to indicate the sweep direction.

    Args:
        voltages: Array of sweep voltages (V).
        currents: Array of measured currents (A).
        title: Title of the plot.
        log_scale: If True, plots absolute current on log y-axis.
        ax: Optional matplotlib axes to draw on.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
    else:
        fig = ax.get_figure()

    # Draw color gradient to indicate direction
    t = np.linspace(0, 1, len(voltages))

    if log_scale:
        abs_currents = np.abs(currents)
        # Avoid zero values for log scale
        abs_currents = np.clip(abs_currents, 1.0e-12, None)

        # Plot hysteresis path with color mapping
        scatter = ax.scatter(voltages, abs_currents, c=t, cmap="viridis", s=4, zorder=3)
        ax.plot(voltages, abs_currents, alpha=0.5, color="gray", linewidth=1)
        ax.set_yscale("log")
        ax.set_ylabel("Absolute Current |I| (A)", fontsize=9, fontweight="bold")
    else:
        scatter = ax.scatter(voltages, currents, c=t, cmap="viridis", s=4, zorder=3)
        ax.plot(voltages, currents, alpha=0.5, color="gray", linewidth=1)
        ax.set_ylabel("Current I (A)", fontsize=9, fontweight="bold")

        # Add horizontal/vertical line through zero
        ax.axhline(0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)
        ax.axvline(0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)

    ax.set_xlabel("Applied Voltage V (V)", fontsize=9, fontweight="bold")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)

    # Add colorbar inside/outside to represent time progression
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalize Sweep Time (t/T)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    return fig


def plot_fitting_results(
    voltages: np.ndarray,
    exp_currents: np.ndarray,
    fit_currents: np.ndarray,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plots a dual-axis/subplot comparison of experimental vs fitted parameters.

    Args:
        voltages: Voltages array.
        exp_currents: Experimental currents array.
        fit_currents: Compact model fitted currents array.
        ax: Optional matplotlib axes to draw on.

    Returns:
        plt.Figure: Figure object.
    """
    if ax is None:
        fig, (ax_main, ax_res) = plt.subplots(
            2, 1, figsize=(5.0, 5.0), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
    else:
        fig = ax.get_figure()
        # If single axis passed, we draw on it without residuals
        ax_main = ax
        ax_res = None

    # Main comparison plot
    ax_main.plot(voltages, exp_currents, "o", label="Experimental Data", markersize=3, alpha=0.7)
    ax_main.plot(
        voltages, fit_currents, "-", label="Compact Model Fit", linewidth=2.0, color="#D55E00"
    )

    ax_main.set_ylabel("Current I (A)", fontsize=9, fontweight="bold")
    ax_main.set_title("Experimental vs. compact model fit", fontsize=10, fontweight="bold")
    ax_main.legend(loc="upper left")
    ax_main.grid(True)

    # Residuals plot
    if ax_res is not None:
        residuals = exp_currents - fit_currents
        ax_res.plot(voltages, residuals, "x", color="gray", markersize=3, alpha=0.7)
        ax_res.axhline(0, color="red", linestyle="--", linewidth=1.0)
        ax_res.set_xlabel("Applied Voltage V (V)", fontsize=9, fontweight="bold")
        ax_res.set_ylabel("Residual (A)", fontsize=9, fontweight="bold")
        ax_res.grid(True)

    plt.tight_layout()
    return fig


def plot_crossbar_heatmaps(
    matrix_data: np.ndarray,
    title: str = "Crossbar Junction Analysis",
    label: str = "Voltage Drop (V)",
    log_scale: bool = False,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plots a 2D spatial heatmap representing crossbar junction states, voltages, or currents.

    Args:
        matrix_data: 2D numpy array of values.
        title: Title of the heatmap.
        label: Colorbar label.
        log_scale: If True, uses logarithmic scaling for colorbar.
        ax: Optional axes.

    Returns:
        plt.Figure: Figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
    else:
        fig = ax.get_figure()

    # Determine scaling
    if log_scale:
        # Avoid zero or negative values
        matrix_clipped = np.clip(matrix_data, 1.0e-12, None)
        im = ax.imshow(matrix_clipped, cmap="plasma", norm=LogNorm())
    else:
        im = ax.imshow(matrix_data, cmap="plasma")

    ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
    ax.set_xlabel("Bitline (Column) Index", fontsize=9)
    ax.set_ylabel("Wordline (Row) Index", fontsize=9)

    # Tick adjustments
    ax.set_xticks(np.arange(matrix_data.shape[1]))
    ax.set_yticks(np.arange(matrix_data.shape[0]))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label, fontsize=9, fontweight="bold")
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    return fig


def plot_snn_activity(
    time_ms: np.ndarray,
    input_spikes: np.ndarray,
    mem_voltages: np.ndarray,
    output_spikes: np.ndarray,
    v_thresh: float = 0.8,
) -> plt.Figure:
    """Generates a publication-quality 3-panel SNN raster and trace plot.

    Args:
        time_ms: 1D array representing time points (ms).
        input_spikes: 2D boolean array of shape (timesteps, in_features).
        mem_voltages: 2D array of shape (timesteps, out_features).
        output_spikes: 2D boolean array of shape (timesteps, out_features).
        v_thresh: Threshold voltage for membrane tracing.

    Returns:
        plt.Figure: Figure object.
    """
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 6.0), sharex=True)

    # 1. Input spikes raster
    in_features = input_spikes.shape[1]
    for idx in range(in_features):
        spike_times = time_ms[input_spikes[:, idx]]
        axes[0].scatter(
            spike_times,
            np.full_like(spike_times, idx),
            marker="|",
            color="black",
            s=25,
            linewidths=0.75,
        )
    axes[0].set_ylabel("Input Row Index", fontsize=9, fontweight="bold")
    axes[0].set_title("Input Spike Raster (Poisson)", fontsize=10, fontweight="bold")
    axes[0].set_ylim(-0.5, in_features - 0.5)

    # 2. Membrane potential traces
    out_features = mem_voltages.shape[1]
    for idx in range(out_features):
        axes[1].plot(time_ms, mem_voltages[:, idx], label=f"Neuron {idx}", linewidth=1.2)
    axes[1].axhline(v_thresh, color="red", linestyle=":", label="Threshold", linewidth=1.0)
    axes[1].set_ylabel("Membrane Voltage (V)", fontsize=9, fontweight="bold")
    axes[1].set_title("Output Neuron Membrane Voltages", fontsize=10, fontweight="bold")
    axes[1].legend(
        loc="upper right", frameon=True, facecolor="white", edgecolor="none", framealpha=0.8
    )

    # 3. Output spikes raster
    for idx in range(out_features):
        spike_times = time_ms[output_spikes[:, idx]]
        axes[2].scatter(
            spike_times,
            np.full_like(spike_times, idx),
            marker="|",
            color="blue",
            s=35,
            linewidths=1.2,
        )
    axes[2].set_ylabel("Neuron Index", fontsize=9, fontweight="bold")
    axes[2].set_xlabel("Time (ms)", fontsize=9, fontweight="bold")
    axes[2].set_title("Output Spikes (LIF Firing)", fontsize=10, fontweight="bold")
    axes[2].set_ylim(-0.5, out_features - 0.5)

    # Grid and style consistency
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(direction="in", top=True, right=True)

    plt.tight_layout()
    return fig
