"""Formatting utilities to convert memristor parameters and metrics into publication-ready tables.

Supports exporting to LaTeX (with booktabs style) and Markdown.
"""


def format_parameter_table(
    parameters: dict[str, float],
    descriptions: dict[str, str] | None = None,
    errors: dict[str, float] | None = None,
    table_format: str = "latex",
    caption: str = "Extracted and Fitted Memristor Device Parameters",
    label: str = "tab:memristor_params",
) -> str:
    """Formats memristor parameters into a publication-quality LaTeX or Markdown table.

    LaTeX output adheres to the clean booktabs package style guidelines (no vertical lines).

    Args:
        parameters: Dictionary of parameter names and numerical values.
        descriptions: Optional dictionary matching parameter names with descriptive descriptions.
        errors: Optional dictionary containing fitting errors (+/-) for each parameter.
        table_format: Target format, either "latex" or "markdown".
        caption: Caption for LaTeX table environment.
        label: Cross-reference label for LaTeX.

    Returns:
        str: String containing the formatted table.
    """
    desc = descriptions or {}
    errs = errors or {}

    if table_format.lower() == "latex":
        lines = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(rf"\caption{{{caption}}}")
        lines.append(rf"\label{{{label}}}")
        lines.append(r"\begin{tabular}{lll}")
        lines.append(r"\toprule")
        lines.append(r"Parameter & Fitted Value & Description \\")
        lines.append(r"\midrule")

        for key, val in parameters.items():
            # Formatting values in scientific notation if very small/large
            val_str = f"{val:.4e}" if abs(val) < 0.001 or abs(val) > 10000.0 else f"{val:.4f}"

            if key in errs:
                err_val = errs[key]
                if abs(err_val) < 1.0e-3 or abs(err_val) > 1.0e4:
                    val_str += rf" \pm {err_val:.4e}"
                else:
                    val_str += rf" \pm {err_val:.4f}"

            # Format name with LaTeX underscores or math mode
            tex_key = key.replace("_", r"\_")

            description = desc.get(key, "--")
            lines.append(f"{tex_key} & {val_str} & {description} \\\\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        return "\n".join(lines)

    else:  # markdown
        lines = []
        lines.append("| Parameter | Fitted Value | Description |")
        lines.append("| :--- | :--- | :--- |")
        for key, val in parameters.items():
            val_str = f"{val:.4e}" if abs(val) < 0.001 or abs(val) > 10000.0 else f"{val:.4f}"

            if key in errs:
                err_val = errs[key]
                if abs(err_val) < 1.0e-3 or abs(err_val) > 1.0e4:
                    val_str += f" ± {err_val:.4e}"
                else:
                    val_str += f" ± {err_val:.4f}"

            description = desc.get(key, "--")
            lines.append(f"| {key} | {val_str} | {description} |")
        return "\n".join(lines)


def format_fitting_metrics_table(
    metrics: dict[str, float],
    table_format: str = "latex",
    caption: str = "Compact Model Parameter Fitting Evaluation Metrics",
    label: str = "tab:fitting_metrics",
) -> str:
    """Formats model optimization and fit error metrics.

    Args:
        metrics: Dictionary of metric names (e.g. "MSE", "R2") and values.
        table_format: Target format, either "latex" or "markdown".
        caption: Caption for LaTeX table environment.
        label: Cross-reference label for LaTeX.

    Returns:
        str: Table string.
    """
    if table_format.lower() == "latex":
        lines = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(rf"\caption{{{caption}}}")
        lines.append(rf"\label{{{label}}}")
        lines.append(r"\begin{tabular}{ll}")
        lines.append(r"\toprule")
        lines.append(r"Evaluation Metric & Value \\")
        lines.append(r"\midrule")

        for key, val in metrics.items():
            val_str = f"{val:.4e}" if abs(val) < 0.001 or abs(val) > 10000.0 else f"{val:.6f}"

            tex_key = key.replace("_", r"\_")
            lines.append(f"{tex_key} & {val_str} \\\\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        return "\n".join(lines)

    else:  # markdown
        lines = []
        lines.append("| Evaluation Metric | Value |")
        lines.append("| :--- | :--- |")
        for key, val in metrics.items():
            val_str = f"{val:.4e}" if abs(val) < 0.001 or abs(val) > 10000.0 else f"{val:.6f}"
            lines.append(f"| {key} | {val_str} |")
        return "\n".join(lines)
