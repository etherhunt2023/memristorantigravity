"""Unit tests for parameter fitting engine."""

import numpy as np
import pytest

from compact_models.vteam import VTEAMModel
from comsol.dataset import COMSOLDataset
from fitting.fitter import ModelFitter
from fitting.loss import calculate_linear_nmse, calculate_log_nmse, hybrid_loss
from utils.config_loader import Config


def test_loss_functions() -> None:
    """Verifies loss computation values and regularization behavior."""
    i_target = np.array([1e-6, 1e-5, 1e-4])
    # Identical currents should yield 0 loss
    assert np.isclose(calculate_linear_nmse(i_target, i_target), 0.0)
    assert np.isclose(calculate_log_nmse(i_target, i_target), 0.0)
    assert np.isclose(hybrid_loss(i_target, i_target, gamma=0.5), 0.0)

    # Different currents should yield positive loss
    i_sim = np.array([1e-6, 1e-5, 2e-4])
    assert calculate_linear_nmse(i_sim, i_target) > 0.0
    assert calculate_log_nmse(i_sim, i_target) > 0.0
    assert hybrid_loss(i_sim, i_target, gamma=0.5) > 0.0


def test_fitter_execution() -> None:
    """Verifies that ModelFitter successfully runs and converges on synthetic data."""
    # Generate synthetic target sweep using VTEAM Model
    v_sweep = np.array([0.0, 0.5, 1.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -1.0, -0.5, 0.0])
    t_points = np.linspace(0, 1.2e-3, len(v_sweep))

    target_params = {
        "w_on": 0.0,
        "w_off": 1.0,
        "v_on": 0.8,
        "v_off": -0.8,
        "k_on": -10.0,
        "k_off": 10.0,
        "alpha_on": 3,
        "alpha_off": 3,
        "r_on": 1000.0,
        "r_off": 100000.0,
        "p": 4,
    }
    target_model = VTEAMModel(target_params)
    _, i_target = target_model.solve_sweep(v_sweep, t_points, w_init=1.0, solver_type="euler")

    # Wrap in dataset
    dataset = COMSOLDataset(
        data=pytest.importorskip("pandas").DataFrame(
            {"time": t_points, "voltage": v_sweep, "current": i_target}
        ),
        metadata={},
    )

    # Configuration with low iteration count to keep test fast
    config_dict = {
        "compact_model": {
            "fitting": {"max_iter": 3, "pop_size": 4, "tolerance": 1e-2, "loss_gamma": 0.3}
        }
    }
    config = Config(config_dict)
    fitter = ModelFitter(config)

    # We will fit k_on and k_off, keeping others fixed
    fit_keys = ["k_on", "k_off"]
    bounds = {"k_on": (-100.0, -0.1), "k_off": (0.1, 100.0)}
    fixed = {
        "w_on": 0.0,
        "w_off": 1.0,
        "v_on": 0.8,
        "v_off": -0.8,
        "alpha_on": 3,
        "alpha_off": 3,
        "r_on": 1000.0,
        "r_off": 100000.0,
        "p": 4,
    }

    fitted = fitter.fit(
        dataset,
        model_class=VTEAMModel,
        fit_param_keys=fit_keys,
        bounds_dict=bounds,
        fixed_params=fixed,
        w_init=1.0,
        solver_type="euler",
    )

    # Assert fitted dict contains all expected keys
    for key in fit_keys:
        assert key in fitted
        # Value should lie within specified bounds
        val = fitted[key]
        assert bounds[key][0] <= val <= bounds[key][1]

    # Assert fixed parameters remain intact
    for key, val in fixed.items():
        assert fitted[key] == val
