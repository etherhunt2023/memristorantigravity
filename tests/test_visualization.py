"""Unit tests for the publication-quality visualization and tables package.

Verifies matplotlib style templates, figure generation functions (I-V, fitting comparison,
heatmaps, SNN raster), and LaTeX booktabs / Markdown table exports.
"""

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


def test_apply_journal_style() -> None:
    """Verifies that apply_journal_style changes matplotlib parameters correctly."""
    # Test IEEE style
    apply_journal_style("ieee")
    assert plt.rcParams["font.size"] == 8
    assert plt.rcParams["axes.grid"] is True
    assert plt.rcParams["grid.linestyle"] == "--"
    assert plt.rcParams["savefig.dpi"] == 300

    # Test Nature style
    apply_journal_style("nature")
    assert plt.rcParams["font.family"] == ["sans-serif"]


def test_plotting_functions() -> None:
    """Verifies that plotting functions execute and return valid figures."""
    apply_journal_style("ieee")

    # Mock data
    voltages = np.linspace(-1.5, 1.5, 50)
    currents = 1.0e-3 * np.sinh(voltages)
    fit_currents = 1.05e-3 * np.sinh(voltages)

    # 1. Test I-V Sweep plot
    fig_iv = plot_iv_sweep(voltages, currents, log_scale=False)
    assert isinstance(fig_iv, plt.Figure)
    plt.close(fig_iv)

    fig_iv_log = plot_iv_sweep(voltages, currents, log_scale=True)
    assert isinstance(fig_iv_log, plt.Figure)
    plt.close(fig_iv_log)

    # 2. Test Fitting comparison plot
    fig_fit = plot_fitting_results(voltages, currents, fit_currents)
    assert isinstance(fig_fit, plt.Figure)
    plt.close(fig_fit)

    # 3. Test Crossbar Spatial heatmap
    matrix_data = np.random.rand(8, 8)
    fig_heat = plot_crossbar_heatmaps(matrix_data)
    assert isinstance(fig_heat, plt.Figure)
    plt.close(fig_heat)

    # 4. Test SNN Activity plot
    time_ms = np.arange(100)
    input_spikes = np.random.rand(100, 4) > 0.8
    mem_voltages = np.random.rand(100, 2) * 0.7
    output_spikes = np.random.rand(100, 2) > 0.9
    fig_snn = plot_snn_activity(time_ms, input_spikes, mem_voltages, output_spikes)
    assert isinstance(fig_snn, plt.Figure)
    plt.close(fig_snn)


def test_table_formatting() -> None:
    """Verifies that parameter and metric tables are generated with valid formats."""
    parameters = {"r_on": 1.2e3, "v_thresh": 0.8, "alpha": 3.5}
    descriptions = {"r_on": "LRS resistance", "v_thresh": "Threshold", "alpha": "Fitting parameter"}
    errors = {"r_on": 15.2, "v_thresh": 0.02, "alpha": 0.12}

    # 1. Test LaTeX parameter table
    latex_params = format_parameter_table(
        parameters, descriptions=descriptions, errors=errors, table_format="latex"
    )
    assert r"\begin{table}" in latex_params
    assert r"\toprule" in latex_params
    assert r"\midrule" in latex_params
    assert r"\bottomrule" in latex_params
    assert r"r\_on & 1200.0000 \pm 15.2000 & LRS resistance \\" in latex_params
    assert "LRS resistance" in latex_params

    # 2. Test Markdown parameter table
    md_params = format_parameter_table(
        parameters, descriptions=descriptions, errors=errors, table_format="markdown"
    )
    assert "| Parameter | Fitted Value | Description |" in md_params
    assert "| r_on | 1200.0000 ± 15.2000 | LRS resistance |" in md_params

    # 3. Test LaTeX metrics table
    metrics = {"MSE": 1.25e-8, "RMSE": 0.0001118, "R2": 0.9982}
    latex_metrics = format_fitting_metrics_table(metrics, table_format="latex")
    assert r"\begin{table}" in latex_metrics
    assert "Evaluation Metric & Value" in latex_metrics
    assert "MSE & 1.2500e-08" in latex_metrics

    # 4. Test Markdown metrics table
    md_metrics = format_fitting_metrics_table(metrics, table_format="markdown")
    assert "| Evaluation Metric | Value |" in md_metrics
    assert "| MSE | 1.2500e-08 |" in md_metrics
