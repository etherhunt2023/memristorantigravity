"""Unit tests for synaptic plasticity models (STDP).

Verifies phenomenological analytical calculations, weight update type differences,
and pulse-based overlap-driven state updates.
"""

import numpy as np
import pytest

from compact_models.vteam import VTEAMModel
from device.memristor import MemristorDevice
from plasticity.stdp import PhenomenologicalSTDP, PulseBasedSTDP


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


def test_phenomenological_stdp_curves() -> None:
    """Tests analytical weight change curves for potentiation and depression."""
    stdp = PhenomenologicalSTDP(a_plus=0.1, a_minus=0.08, tau_plus=0.02, tau_minus=0.02)

    # Positive dt -> potentiation (dw > 0)
    dw_pos1 = stdp.calculate_delta_w(0.01)
    dw_pos2 = stdp.calculate_delta_w(0.03)
    assert dw_pos1 > 0
    assert dw_pos2 > 0
    assert dw_pos1 > dw_pos2  # decay over time

    # Negative dt -> depression (dw < 0)
    dw_neg1 = stdp.calculate_delta_w(-0.01)
    dw_neg2 = stdp.calculate_delta_w(-0.03)
    assert dw_neg1 < 0
    assert dw_neg2 < 0
    assert abs(dw_neg1) > abs(dw_neg2)  # decay over time

    # Zero dt -> no change
    assert np.allclose(stdp.calculate_delta_w(0.0), 0.0)

    # Vectorized check
    dt_arr = np.array([0.01, -0.01, 0.0])
    dw_arr = stdp.calculate_delta_w(dt_arr)
    assert dw_arr.shape == (3,)
    assert dw_arr[0] > 0
    assert dw_arr[1] < 0
    assert np.allclose(dw_arr[2], 0.0)


def test_phenomenological_stdp_updates(base_vteam_params: dict[str, float]) -> None:
    """Tests additive and multiplicative state updates on MemristorDevice."""
    # Additive update check
    stdp_add = PhenomenologicalSTDP(
        a_plus=0.05, a_minus=0.05, tau_plus=0.02, tau_minus=0.02, update_type="additive"
    )

    device_add = MemristorDevice(
        VTEAMModel,
        base_vteam_params,
        device_config={"d2d": {"enabled": False}, "c2c": {"enabled": False}},
        w_init=0.5,
    )

    # Potentiation update (should decrease w from 0.5)
    stdp_add.apply_stdp(device_add, 0.01)  # delta_t = +10ms
    assert device_add.w < 0.5

    # Reset and try depression (should increase w from 0.5)
    device_add.reset()
    stdp_add.apply_stdp(device_add, -0.01)  # delta_t = -10ms
    assert device_add.w > 0.5

    # Multiplicative update check (should scale update according to boundaries)
    stdp_mult = PhenomenologicalSTDP(
        a_plus=0.1, a_minus=0.1, tau_plus=0.02, tau_minus=0.02, update_type="multiplicative"
    )
    device_mult = MemristorDevice(
        VTEAMModel,
        base_vteam_params,
        device_config={"d2d": {"enabled": False}, "c2c": {"enabled": False}},
        w_init=0.9,  # close to w_max (1.0), so potentiation should be larger than depression
    )

    # Calculate update size at w=0.9
    w_before = device_mult.w
    stdp_mult.apply_stdp(device_mult, 0.01)  # Potentiation
    pot_size = w_before - device_mult.w

    device_mult.w = 0.9
    stdp_mult.apply_stdp(device_mult, -0.01)  # Depression
    dep_size = device_mult.w - w_before

    # At w=0.9, w is far from w_min=0, so (w - w_min) = 0.9 is large (high potentiation speed)
    # whereas (w_max - w) = 0.1 is small (low depression speed)
    assert pot_size > dep_size


def test_pulse_based_waveform() -> None:
    """Verifies that PulseBasedSTDP generates correct overlapping voltage profiles."""
    pb = PulseBasedSTDP(v_pre_amp=1.5, v_post_amp=1.5, tau_pulse=0.01, pulse_duration=0.04)

    time_points = np.linspace(0.0, 0.04, 100)

    # Test dt > 0 (pre-spike before post-spike)
    v_net_pos = pb.generate_waveform(0.005, time_points)

    # Pre-spike occurs at t=20ms, post-spike at t=25ms
    # Net voltage drop should be positive between 20ms and 25ms, then drop negative
    idx_22ms = int(22.0e-3 / (0.04 / 100.0))
    idx_27ms = int(27.0e-3 / (0.04 / 100.0))

    assert v_net_pos[idx_22ms] > 0
    assert v_net_pos[idx_27ms] < 0


def test_pulse_based_stdp_execution(base_vteam_params: dict[str, float]) -> None:
    """Tests physical pulse overlap-driven state updates in MemristorDevice."""
    # Use higher amplitudes to ensure threshold switching is active
    pb = PulseBasedSTDP(v_pre_amp=1.2, v_post_amp=1.2, tau_pulse=0.01, pulse_duration=0.04)

    device_pos = MemristorDevice(
        VTEAMModel,
        base_vteam_params,
        device_config={
            "d2d": {"enabled": False},
            "c2c": {"enabled": False},
            "drift": {"enabled": False},
        },
        w_init=0.5,
    )
    device_neg = MemristorDevice(
        VTEAMModel,
        base_vteam_params,
        device_config={
            "d2d": {"enabled": False},
            "c2c": {"enabled": False},
            "drift": {"enabled": False},
        },
        w_init=0.5,
    )

    # Positive delta_t (t_post > t_pre) -> Potentiation -> w should decrease from 0.5
    pb.apply_stdp(device_pos, 0.003)

    # Negative delta_t (t_post < t_pre) -> Depression -> w should increase from 0.5
    pb.apply_stdp(device_neg, -0.003)

    assert device_pos.w < 0.5
    assert device_neg.w > 0.5
