"""Example script demonstrating crossbar array simulation with Modified Nodal Analysis.

Sets up a 16x16 memristor crossbar array, programs a diagonal pattern,
applies input voltages, solves node voltages under wire parasitics (IR drop),
and plots the resulting 2D voltage distribution and cell currents.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compact_models.vteam import VTEAMModel
from crossbar.array import CrossbarArray


def run_crossbar_simulation() -> None:
    """Simulates a 16x16 crossbar array and plots node voltage distributions."""
    print("==================================================")
    print("SIMULATING MEMRISTOR CROSSBAR ARRAY PARASITICS")
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

    # Crossbar dimensions
    m, n = 16, 16

    # Configure high parasitic wire resistances to make IR drops visually striking
    crossbar_config = {
        "line_resistance": 5.0,  # 5 Ohms per segment (standard crossbars range from 0.5 to 5 Ohms)
        "source_resistance": 50.0,
        "load_resistance": 50.0,
    }

    # Instantiate the crossbar
    cb = CrossbarArray(
        rows=m,
        cols=n,
        model_class=VTEAMModel,
        base_params=base_params,
        device_config={"d2d": {"enabled": False}},  # disable D2D for clean visual patterns
        crossbar_config=crossbar_config,
    )

    # Program a diagonal pattern of devices to LRS (w = 0.0), others remain at HRS (w = 1.0)
    print("Programming a diagonal pattern to Low Resistance State (LRS)...")
    for i in range(m):
        cb.devices[i, i].w = 0.0  # Main diagonal
        if i > 0:
            cb.devices[i, i - 1].w = 0.0  # Subdiagonal

    # Apply input voltages: 1.0V to first 8 rows, 0.0V to remaining rows
    row_voltages = np.zeros(m)
    row_voltages[:8] = 1.0

    # Solve MNA
    print("Solving Modified Nodal Analysis (MNA) using Newton-Raphson...")
    v_row, v_col = cb.solve_mna(row_voltages, use_nonlinear=True)
    v_drop = v_row - v_col

    # Plot results
    temp_dir = Path(__file__).resolve().parent / "temp_data"
    temp_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Row line voltages heatmap
    im0 = axes[0].imshow(v_row, cmap="plasma", aspect="equal", vmin=0, vmax=1.0)
    axes[0].set_title("Wordline (Row) Node Voltages", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Column Index", fontsize=10)
    axes[0].set_ylabel("Row Index", fontsize=10)
    fig.colorbar(im0, ax=axes[0], label="Voltage (V)")

    # 2. Column line voltages heatmap
    im1 = axes[1].imshow(v_col, cmap="plasma", aspect="equal", vmin=0, vmax=1.0)
    axes[1].set_title("Bitline (Column) Node Voltages", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Column Index", fontsize=10)
    axes[1].set_ylabel("Row Index", fontsize=10)
    fig.colorbar(im1, ax=axes[1], label="Voltage (V)")

    # 3. Voltage drop (|V_row - V_col|) heatmap
    im2 = axes[2].imshow(np.abs(v_drop), cmap="inferno", aspect="equal", vmin=0, vmax=1.0)
    axes[2].set_title("Net Voltage Drop Across Memristors", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Column Index", fontsize=10)
    axes[2].set_ylabel("Row Index", fontsize=10)
    fig.colorbar(im2, ax=axes[2], label="Voltage drop |V_drop| (V)")

    plt.tight_layout()
    output_path = temp_dir / "crossbar_voltage_distribution.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    # -------------------------------------------------------------
    # Sneak Path Diagnostic Analysis
    # -------------------------------------------------------------
    print("\nPart 2: Running Sneak Path Diagnostic Analysis...")
    # Reset all devices to HRS
    for i in range(m):
        for j in range(n):
            cb.devices[i, j].w = 1.0  # HRS

    # 1. Analyze sneak path efficiency when target cell (0, 0) is in LRS (surrounded by HRS)
    cb.devices[0, 0].w = 0.0  # set target to LRS
    res_lrs = cb.analyze_sneak_paths(
        target_row=0, target_col=0, read_voltage=0.2, scheme="grounded"
    )

    # 2. Analyze sneak path efficiency when target cell (0, 0) is in HRS (surrounded by LRS)
    # This is the worst-case sneak path scenario
    for i in range(m):
        for j in range(n):
            cb.devices[i, j].w = 0.0  # set all to LRS
    cb.devices[0, 0].w = 1.0  # target to HRS
    res_hrs = cb.analyze_sneak_paths(
        target_row=0, target_col=0, read_voltage=0.2, scheme="grounded"
    )

    print("\nSneak Path Diagnostic Report (Grounded Scheme):")
    print("  Target cell (0,0) in LRS (Clean Read):")
    print(f"    - Target Cell Current:  {res_lrs['target_current']*1e6:8.2f} uA")
    print(f"    - Total Column Current:  {res_lrs['total_column_current']*1e6:8.2f} uA")
    print(f"    - Sneak Path Leakage:    {res_lrs['sneak_current']*1e6:8.2f} uA")
    print(f"    - Sneak Path Efficiency: {res_lrs['sneak_efficiency']*100:8.2f} %")

    print("\n  Target cell (0,0) in HRS (Worst-Case Leakage):")
    print(f"    - Target Cell Current:  {res_hrs['target_current']*1e6:8.2f} uA")
    print(f"    - Total Column Current:  {res_hrs['total_column_current']*1e6:8.2f} uA")
    print(f"    - Sneak Path Leakage:    {res_hrs['sneak_current']*1e6:8.2f} uA")
    print(f"    - Sneak Path Efficiency: {res_hrs['sneak_efficiency']*100:8.2f} %")

    print(
        f"\nSimulation complete! Voltage heatmaps saved to:\n[crossbar_voltage_distribution.png](file:///{output_path.as_posix()})"
    )
    print("==================================================\n")


if __name__ == "__main__":
    run_crossbar_simulation()
