"""Publication-quality visualization and tables package.

Provides high-impact styling templates and formatting utilities for journal papers.
"""

from visualization.plotter import (
    apply_journal_style,
    plot_crossbar_heatmaps,
    plot_fitting_results,
    plot_iv_sweep,
    plot_snn_activity,
)
from visualization.tables import format_fitting_metrics_table, format_parameter_table

__all__ = [
    "apply_journal_style",
    "plot_iv_sweep",
    "plot_fitting_results",
    "plot_crossbar_heatmaps",
    "plot_snn_activity",
    "format_parameter_table",
    "format_fitting_metrics_table",
]
