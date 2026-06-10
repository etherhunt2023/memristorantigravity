"""Unit tests for physics-based compact models and window functions."""

import numpy as np

from compact_models.simmons import SimmonsModel
from compact_models.vteam import VTEAMModel
from compact_models.windows import biolek_window, joglekar_window, prodromakis_window
from compact_models.yakopcic import YakopcicModel


def test_window_functions() -> None:
    """Tests the values and limits of Joglekar, Biolek, and Prodromakis windows."""
    # Joglekar window limits
    assert np.isclose(joglekar_window(0.0, p=1), 0.0)
    assert np.isclose(joglekar_window(1.0, p=1), 0.0)
    assert np.isclose(joglekar_window(0.5, p=1), 1.0)

    # Biolek window (dependent on voltage direction)
    # Moving towards w=0 (positive voltage)
    assert np.isclose(biolek_window(0.0, v=1.0, p=2), 0.0)
    assert biolek_window(0.5, v=1.0, p=2) > 0.0
    # Moving towards w=1 (negative voltage)
    assert np.isclose(biolek_window(1.0, v=-1.0, p=2), 0.0)

    # Prodromakis window
    assert prodromakis_window(0.5, p=1, j=1.0) > 0.0
    assert prodromakis_window(0.0, p=1, j=1.0) >= 0.0


def test_vteam_model() -> None:
    """Tests VTEAM model derivative and current calculations."""
    params = {
        "w_on": 0.0,
        "w_off": 1.0,
        "v_on": 1.0,
        "v_off": -1.0,
        "k_on": -2.0,
        "k_off": 2.0,
        "alpha_on": 1,
        "alpha_off": 1,
        "r_on": 100.0,
        "r_off": 10000.0,
        "p": 2,
    }
    model = VTEAMModel(params)

    # Within thresholds, derivative should be zero
    assert np.isclose(model.deriv(v=0.5, w=0.5), 0.0)
    assert np.isclose(model.deriv(v=-0.5, w=0.5), 0.0)

    # Above threshold, derivative is non-zero
    # For V > v_on, w moves towards w_on (0.0), so dw/dt should be negative
    dw = model.deriv(v=1.5, w=0.5)
    assert dw < 0.0

    # Below threshold, derivative is non-zero
    # For V < v_off, w moves towards w_off (1.0), so dw/dt should be positive
    dw = model.deriv(v=-1.5, w=0.5)
    assert dw > 0.0

    # Vector inputs
    voltages = np.array([0.0, 1.5, -1.5])
    states = np.array([0.5, 0.5, 0.5])
    dws = model.deriv(voltages, states)
    assert len(dws) == 3
    assert np.isclose(dws[0], 0.0)
    assert dws[1] < 0.0
    assert dws[2] > 0.0

    # Current checks
    # I = V / R(w)
    # w = 0 -> R = r_on = 100.
    assert np.isclose(model.current(v=1.0, w=0.0), 1.0 / 100.0)
    # w = 1 -> R = r_off = 10000.
    assert np.isclose(model.current(v=1.0, w=1.0), 1.0 / 10000.0)


def test_yakopcic_model() -> None:
    """Tests Yakopcic model current and derivative."""
    params = {
        "a1": 0.01,
        "a2": 0.01,
        "b": 1.0,
        "vp": 1.0,
        "vn": 1.0,
        "gp": 5.0,
        "gn": 5.0,
        "ap": 0.5,
        "an": 0.5,
        "xp": 0.4,
        "xn": 0.4,
        "eta": 1.0,
    }
    model = YakopcicModel(params)

    # Zero derivative within thresholds
    assert np.isclose(model.deriv(v=0.5, w=0.5), 0.0)

    # Positive derivative above threshold
    assert model.deriv(v=1.5, w=0.3) > 0.0

    # Vector current
    v = np.array([1.0, -1.0])
    w = np.array([0.5, 0.5])
    i = model.current(v, w)
    assert len(i) == 2
    assert i[0] > 0.0
    assert i[1] < 0.0


def test_simmons_model() -> None:
    """Tests Simmons tunneling barrier model."""
    params = {
        "w_on": 0.2e-9,
        "w_off": 1.2e-9,
        "tox": 3.0e-9,
        "phi_0": 0.95,
        "area": 1e-12,
        "k_on": -1e-9,
        "k_off": 1e-9,
        "v_on": 0.5,
        "v_off": -0.5,
        "p": 2,
    }
    model = SimmonsModel(params)

    # Derivative checks
    assert np.isclose(model.deriv(v=0.1, w=0.5e-9), 0.0)
    assert model.deriv(v=1.0, w=0.5e-9) < 0.0
    assert model.deriv(v=-1.0, w=0.5e-9) > 0.0

    # Current calculations
    i_val = model.current(v=1.0, w=0.5e-9)
    assert abs(i_val) > 0.0
    # Current should increase as gap width w decreases (tunneling exponential)
    i_small_gap = model.current(v=1.0, w=0.3e-9)
    assert abs(i_small_gap) > abs(i_val)


def test_solve_sweep() -> None:
    """Tests ODE integration solver routines (Euler & RK4) in base class."""
    params = {
        "w_on": 0.0,
        "w_off": 1.0,
        "v_on": 1.0,
        "v_off": -1.0,
        "k_on": -10.0,
        "k_off": 10.0,
        "alpha_on": 1,
        "alpha_off": 1,
        "r_on": 100.0,
        "r_off": 10000.0,
        "p": 2,
    }
    model = VTEAMModel(params)

    time_points = np.linspace(0, 1.0, 101)
    # Voltage sweep: positive pulse above threshold
    voltage_sweep = np.zeros_like(time_points)
    voltage_sweep[(time_points > 0.2) & (time_points < 0.8)] = 2.0

    # 1. Test RK4 solver (default)
    w_rk4, i_rk4 = model.solve_sweep(voltage_sweep, time_points, w_init=1.0, solver_type="rk4")
    assert len(w_rk4) == 101
    assert len(i_rk4) == 101
    assert np.all(w_rk4 >= 0.0) and np.all(w_rk4 <= 1.0)
    # State variable should decrease during positive pulse
    assert w_rk4[80] < w_rk4[10]

    # 2. Test Euler solver
    w_euler, i_euler = model.solve_sweep(
        voltage_sweep, time_points, w_init=1.0, solver_type="euler"
    )
    assert len(w_euler) == 101
    assert np.all(w_euler >= 0.0) and np.all(w_euler <= 1.0)
    assert w_euler[80] < w_euler[10]
