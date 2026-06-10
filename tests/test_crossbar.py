"""Unit tests for the CrossbarArray simulator.

Verifies Modified Nodal Analysis (MNA), Newton-Raphson non-linear solver,
parasitic IR drops, and sneak path efficiency calculations.
"""

import numpy as np
import pytest

from compact_models.vteam import VTEAMModel
from crossbar.array import CrossbarArray


@pytest.fixture
def base_vteam_params() -> dict[str, float]:
    """Provides base VTEAM parameters for testing."""
    return {
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


def test_crossbar_initialization(base_vteam_params: dict[str, float]) -> None:
    """Tests dimensions and device sampling in CrossbarArray."""
    m, n = 4, 8
    crossbar_config = {
        "line_resistance": 2.0,
        "source_resistance": 50.0,
        "load_resistance": 50.0,
    }

    # Disable D2D to check deterministic nominal initialization
    device_config = {"d2d": {"enabled": False}}

    cb = CrossbarArray(
        rows=m,
        cols=n,
        model_class=VTEAMModel,
        base_params=base_vteam_params,
        device_config=device_config,
        crossbar_config=crossbar_config,
    )

    assert cb.rows == m
    assert cb.cols == n
    assert cb.r_line_r == 2.0
    assert cb.r_line_c == 2.0
    assert cb.r_source == 50.0
    assert cb.r_load == 50.0

    # Ensure all devices are VTEAMModel wrappers and w is set to w_off
    assert cb.devices.shape == (m, n)
    for i in range(m):
        for j in range(n):
            assert isinstance(cb.devices[i, j].model, VTEAMModel)
            assert np.allclose(cb.devices[i, j].w, 1.0)  # w_off


def test_linear_vs_nonlinear_mna(base_vteam_params: dict[str, float]) -> None:
    """Verifies that linear and non-linear solvers return similar voltages for low voltage."""
    cb = CrossbarArray(
        rows=4,
        cols=4,
        model_class=VTEAMModel,
        base_params=base_vteam_params,
        device_config={"d2d": {"enabled": False}, "c2c": {"enabled": False}},
    )

    # Apply small voltage sweep (0.1V to row 0)
    row_v = np.zeros(4)
    row_v[0] = 0.1

    # Solve MNA using both linear and non-linear (Newton-Raphson) solvers
    v_row_lin, v_col_lin = cb.solve_mna(row_v, use_nonlinear=False)
    v_row_nonlin, v_col_nonlin = cb.solve_mna(row_v, use_nonlinear=True)

    # At 0.1V, VTEAM is fully linear (below threshold), so linear and non-linear results must match
    assert np.allclose(v_row_lin, v_row_nonlin, rtol=1.0e-3)
    assert np.allclose(v_col_lin, v_col_nonlin, rtol=1.0e-3)


def test_ir_drop_propagation(base_vteam_params: dict[str, float]) -> None:
    """Tests that parasitic line resistances cause a measurable voltage degradation."""
    # Create array with high wire resistance to make IR drop visible
    crossbar_config = {
        "line_resistance": 50.0,  # 50 ohms per segment
        "source_resistance": 10.0,
        "load_resistance": 10.0,
    }

    # Set all memristors to LRS (w = 0.0, r = 1000 ohms) to draw significant line currents
    cb = CrossbarArray(
        rows=1,
        cols=5,
        model_class=VTEAMModel,
        base_params=base_vteam_params,
        device_config={"d2d": {"enabled": False}, "c2c": {"enabled": False}},
        crossbar_config=crossbar_config,
    )
    for j in range(5):
        cb.devices[0, j].w = 0.0

    # Apply 1.0V to row 0 terminal
    row_v = np.array([1.0])
    v_row, _ = cb.solve_mna(row_v, use_nonlinear=False)

    # Node voltages along the row should strictly decrease from left to right (IR drop)
    voltages = v_row[0, :]
    assert voltages[0] < 1.0  # degraded by source resistance
    assert voltages[0] > voltages[1]
    assert voltages[1] > voltages[2]
    assert voltages[2] > voltages[3]
    assert voltages[3] > voltages[4]


def test_sneak_path_diagnostics(base_vteam_params: dict[str, float]) -> None:
    """Tests sneak path analysis under LRS and HRS target device states."""
    cb = CrossbarArray(
        rows=3,
        cols=3,
        model_class=VTEAMModel,
        base_params=base_vteam_params,
        device_config={"d2d": {"enabled": False}},
    )

    # Case 1: Target cell (0, 0) is in HRS (w = 1.0, R = 100k)
    # while others are in LRS (w = 0.0, R = 1k). This represents the
    # worst-case sneak path scenario (reading HRS surrounded by LRS).
    for i in range(3):
        for j in range(3):
            cb.devices[i, j].w = 0.0
    cb.devices[0, 0].w = 1.0

    res_hrs = cb.analyze_sneak_paths(
        target_row=0, target_col=0, read_voltage=0.1, scheme="grounded"
    )

    # Case 2: Target cell (0, 0) is in LRS (w = 0.0, R = 1k)
    # while others are in HRS (w = 1.0, R = 100k). This is a clean read.
    for i in range(3):
        for j in range(3):
            cb.devices[i, j].w = 1.0
    cb.devices[0, 0].w = 0.0

    res_lrs = cb.analyze_sneak_paths(
        target_row=0, target_col=0, read_voltage=0.1, scheme="grounded"
    )

    # Sneak efficiency should be much higher (closer to 1.0) when reading LRS than when reading HRS
    # because when target is HRS, the leakage through other LRS cells dominates the column output
    assert res_lrs["sneak_efficiency"] > res_hrs["sneak_efficiency"]
    assert res_hrs["sneak_efficiency"] < 0.5  # high leakage, low efficiency
    assert res_lrs["sneak_efficiency"] > 0.8  # low leakage, high efficiency
