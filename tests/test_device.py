"""Unit tests for the hardware-aware MemristorDevice class.

Verifies device-to-device (D2D) parameter variation, cycle-to-cycle (C2C)
temporal noise, conductance/resistance drift, and current noise (thermal and shot).
"""

import numpy as np
import pytest

from compact_models.vteam import VTEAMModel
from device.memristor import MemristorDevice


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


def test_d2d_variation(base_vteam_params: dict[str, float]) -> None:
    """Tests that device-to-device variation samples parameters within expected statistics."""
    device_config = {
        "d2d": {
            "enabled": True,
            "parameters": {
                "r_on": {"dist": "lognormal", "std": 0.1},
                "r_off": {"dist": "lognormal", "std": 0.1},
                "v_on": {"dist": "gaussian", "std": 0.05},
                "v_off": {"dist": "gaussian", "std": 0.05},
            },
        },
        "c2c": {"enabled": False},
        "drift": {"enabled": False},
        "noise": {"thermal": False, "shot": False, "generic_std": 0.0},
    }

    # Instantiate multiple devices
    num_devices = 50
    devices = [
        MemristorDevice(VTEAMModel, base_vteam_params, device_config=device_config)
        for _ in range(num_devices)
    ]

    r_ons = [d.params["r_on"] for d in devices]
    v_ons = [d.params["v_on"] for d in devices]

    # Verify that values vary
    assert len(set(r_ons)) > 1
    assert len(set(v_ons)) > 1

    # Verify bounds
    assert all(r > 0 for r in r_ons)
    assert all(v > 0 for v in v_ons)

    # Verify that mean is close to nominal (with some statistical tolerance)
    assert 900.0 < np.mean(r_ons) < 1100.0
    assert 0.7 < np.mean(v_ons) < 0.9


def test_c2c_variation(base_vteam_params: dict[str, float]) -> None:
    """Tests that cycle-to-cycle variation adds stochastic state updates."""
    # Config with C2C enabled
    c2c_config = {
        "d2d": {"enabled": False},
        "c2c": {"enabled": True, "state_noise_std": 0.2},
        "drift": {"enabled": False},
        "noise": {"thermal": False, "shot": False, "generic_std": 0.0},
    }

    # Simulate a single step multiple times from the same state on different devices
    final_states = []
    for _ in range(30):
        d = MemristorDevice(VTEAMModel, base_vteam_params, device_config=c2c_config, w_init=0.5)
        d.step(1.5, 0.01)  # step forward
        final_states.append(d.w)

    # Verify that states are stochastic
    assert len(set(final_states)) > 1
    assert all(0.0 <= w <= 1.0 for w in final_states)


def test_resistance_drift(base_vteam_params: dict[str, float]) -> None:
    """Tests resistance power-law drift over time."""
    drift_config = {
        "d2d": {"enabled": False},
        "c2c": {"enabled": False},
        "drift": {
            "enabled": True,
            "coeff": 0.1,
            "t_zero": 1.0,
            "type": "resistance",
            "programming_threshold": 0.5,
        },
        "noise": {"thermal": False, "shot": False, "generic_std": 0.0},
    }

    device = MemristorDevice(VTEAMModel, base_vteam_params, device_config=drift_config, w_init=0.0)

    # Set state, initially at LRS (w = 0.0, resistance = 1000 ohms)
    i_initial = device.step(0.1, 0.0)  # t_programming = 0, no drift yet
    assert np.allclose(i_initial, 0.1 / 1000.0)

    # Step with 0.1V (below 0.5V programming threshold), so drift is active
    # Let 10 seconds elapse
    i_drifed = device.step(0.1, 10.0)

    # Expected drift multiplier: (1 + 10 / 1.0) ^ 0.1 = 11 ^ 0.1 approx 1.27
    expected_drift_mult = (1.0 + 10.0 / 1.0) ** 0.1
    expected_current = (0.1 / 1000.0) / expected_drift_mult

    assert np.allclose(i_drifed, expected_current)


def test_state_drift(base_vteam_params: dict[str, float]) -> None:
    """Tests state relaxation drift towards relaxed state."""
    drift_config = {
        "d2d": {"enabled": False},
        "c2c": {"enabled": False},
        "drift": {
            "enabled": True,
            "coeff": 0.2,
            "t_zero": 1.0,
            "type": "state",
            "w_relaxed": 1.0,  # relax towards w_off (HRS)
            "programming_threshold": 0.5,
        },
        "noise": {"thermal": False, "shot": False, "generic_std": 0.0},
    }

    # Initial state is at LRS (w = 0.0)
    device = MemristorDevice(VTEAMModel, base_vteam_params, device_config=drift_config, w_init=0.0)

    # Apply small voltage (no programming), letting 9.0s elapse
    device.step(0.1, 9.0)

    # Expected state: w_relaxed + (w_programmed - w_relaxed) * (1 + t/t0)^-coeff
    # = 1.0 + (0.0 - 1.0) * (1 + 9/1)^-0.2 = 1.0 - 10^-0.2 = 1.0 - 0.6309 = 0.3691
    expected_w = 1.0 + (0.0 - 1.0) * ((1.0 + 9.0 / 1.0) ** -0.2)
    assert np.allclose(device.w, expected_w, rtol=1e-4)


def test_electrical_noise(base_vteam_params: dict[str, float]) -> None:
    """Tests Johnson-Nyquist thermal noise and shot noise."""
    noise_config = {
        "d2d": {"enabled": False},
        "c2c": {"enabled": False},
        "drift": {"enabled": False},
        "noise": {
            "thermal": True,
            "shot": True,
            "bandwidth": 1.0e8,  # very high bandwidth to make noise visible
            "temperature": 300.0,
            "generic_std": 0.0,
        },
    }

    # Read voltage of 0.1V, w=0.0 (R=1000)
    device = MemristorDevice(VTEAMModel, base_vteam_params, device_config=noise_config, w_init=0.0)

    currents = [device.current(0.1) for _ in range(100)]

    # Standard deviation should be positive and close to theoretical noise
    std_sampled = np.std(currents)
    assert std_sampled > 0.0

    # Theoretical noise:
    # r_eff = 1000
    # i_active = 0.1 / 1000 = 1e-4
    # S_thermal = 4 * k_B * T / R = 4 * 1.38e-23 * 300 / 1000 = 1.656e-23
    # S_shot = 2 * q * |I| = 2 * 1.602e-19 * 1e-4 = 3.204e-23
    # total_variance = (S_thermal + S_shot) * B = (1.656e-23 + 3.204e-23) * 1e8 = 4.86e-15
    # std_theoretical = sqrt(4.86e-15) approx 6.97e-8
    assert std_sampled < 1.0e-5  # upper limit
