"""Unit tests for the PyTorch integration module.

Verifies custom autograd functions, forward/backward passes (both MNA and ideal modes),
weight-to-state bidirectional synchronization, and training convergence.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from compact_models.vteam import VTEAMModel
from pytorch.layer import CrossbarFunction, MemristorLinear


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


def test_crossbar_function_ideal(base_vteam_params: dict[str, float]) -> None:
    """Tests CrossbarFunction in ideal mode (matrix VMM)."""
    in_features, out_features = 2, 2
    batch_size = 3

    # Instantiate layer just to get its crossbar instance
    layer = MemristorLinear(
        in_features=in_features,
        out_features=out_features,
        model_class=VTEAMModel,
        base_params=base_vteam_params,
        use_mna=False,
        bias=False,
    )

    x = torch.randn(batch_size, in_features, requires_grad=True)
    weights = torch.randn(in_features, out_features, requires_grad=True)

    y = CrossbarFunction.apply(x, weights, layer.crossbar, False)

    assert y.shape == (batch_size, out_features)

    # Check forward output mathematically (ideal mode is just matrix multiplication)
    expected_y = x @ weights
    assert torch.allclose(y, expected_y, atol=1e-5)

    # Run backward pass
    loss = y.sum()
    loss.backward()

    # Check gradients
    assert x.grad is not None
    assert weights.grad is not None

    # For ideal mode:
    # y_ij = sum_k x_ik * w_kj -> d(sum y_ij)/dx_il = sum_j w_lj
    # hence grad_x = grad_output @ weights^T where grad_output is all ones.
    expected_grad_x = torch.ones(batch_size, out_features) @ weights.t()
    expected_grad_weights = x.t() @ torch.ones(batch_size, out_features)

    assert torch.allclose(x.grad, expected_grad_x, atol=1e-5)
    assert torch.allclose(weights.grad, expected_grad_weights, atol=1e-5)


def test_crossbar_function_mna(base_vteam_params: dict[str, float]) -> None:
    """Tests CrossbarFunction in MNA mode."""
    in_features, out_features = 2, 2
    batch_size = 2

    # Config to disable noise and variations for deterministic MNA
    device_config = {
        "d2d": {"enabled": False},
        "c2c": {"enabled": False},
        "drift": {"enabled": False},
        "noise": {"thermal": False, "shot": False, "generic_std": 0.0},
    }

    layer = MemristorLinear(
        in_features=in_features,
        out_features=out_features,
        model_class=VTEAMModel,
        base_params=base_vteam_params,
        device_config=device_config,
        use_mna=True,
        bias=False,
    )

    # Make inputs positive and within typical read voltage range to avoid programming
    x = torch.tensor([[0.1, 0.2], [0.05, 0.15]], requires_grad=True)

    # Set weight to deterministic values (conductances)
    g_init = torch.tensor([[1e-4, 5e-5], [2e-5, 8e-5]])
    with torch.no_grad():
        layer.weight.copy_(g_init)
    layer.sync_weights_to_devices()

    # Forward pass under MNA
    y = CrossbarFunction.apply(x, layer.weight, layer.crossbar, True)

    assert y.shape == (batch_size, out_features)

    # Output currents should be positive and non-zero
    assert torch.all(y > 0.0)

    # Backward pass using STE (ideal gradients)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert layer.weight.grad is not None

    # Check that gradients propagate and are finite
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.weight.grad).all()


def test_memristor_linear_sync(base_vteam_params: dict[str, float]) -> None:
    """Tests bidirectional weight-to-state synchronization in MemristorLinear."""
    in_features, out_features = 2, 2
    device_config = {
        "d2d": {"enabled": False},
        "c2c": {"enabled": False},
        "drift": {"enabled": False},
        "noise": {"thermal": False, "shot": False, "generic_std": 0.0},
    }

    layer = MemristorLinear(
        in_features=in_features,
        out_features=out_features,
        model_class=VTEAMModel,
        base_params=base_vteam_params,
        device_config=device_config,
        use_mna=False,
        bias=True,
    )

    # Check boundaries
    assert layer.g_min == 1.0 / base_vteam_params["r_off"]
    assert layer.g_max == 1.0 / base_vteam_params["r_on"]

    # 1. Test weight to device sync
    # Set weights to intermediate conductances
    g_target = torch.tensor([[5e-4, 1e-4], [8e-5, 1e-5]])
    with torch.no_grad():
        layer.weight.copy_(g_target)

    # Sync
    layer.sync_weights_to_devices()

    # Verify state variables w in crossbar devices
    for i in range(in_features):
        for j in range(out_features):
            g_val = g_target[i, j].item()
            r_eff = 1.0 / g_val
            # w = w_on + (w_off - w_on) * (R_eff - R_on) / (R_off - R_on)
            expected_w = layer.w_on + (layer.w_off - layer.w_on) * (r_eff - layer.r_on) / (
                layer.r_off - layer.r_on
            )
            expected_w = np.clip(expected_w, layer.w_on, layer.w_off)
            actual_w = layer.crossbar.devices[i, j].w
            assert np.allclose(actual_w, expected_w, rtol=1e-4)

    # 2. Test device to weight sync
    # Manually modify device state variables w
    layer.crossbar.devices[0, 0].w = 0.5
    layer.crossbar.devices[1, 1].w = 0.1

    # Sync back
    layer.sync_devices_to_weights()

    # Get programmatically sync'ed weight values
    weight_after_sync = layer.weight.detach().cpu().numpy()

    # Verify w=0.5 maps correctly
    # R_eff = R_on + (R_off - R_on) * w_norm
    r_on = base_vteam_params["r_on"]
    r_off = base_vteam_params["r_off"]
    r_eff_00 = r_on + (r_off - r_on) * 0.5
    expected_g_00 = 1.0 / r_eff_00
    assert np.allclose(weight_after_sync[0, 0], expected_g_00, rtol=1e-4)


def test_memristor_linear_training_loop(base_vteam_params: dict[str, float]) -> None:
    """Tests that MemristorLinear layer converges in a basic optimization loop."""
    torch.manual_seed(42)
    np.random.seed(42)

    in_features, out_features = 2, 2
    batch_size = 4

    device_config = {
        "d2d": {"enabled": False},
        "c2c": {"enabled": False},
        "drift": {"enabled": False},
        "noise": {"thermal": False, "shot": False, "generic_std": 0.0},
    }

    # Ideal mode first
    layer = MemristorLinear(
        in_features=in_features,
        out_features=out_features,
        model_class=VTEAMModel,
        base_params=base_vteam_params,
        device_config=device_config,
        use_mna=False,
        bias=False,
    )

    # Target parameters (ground truth weights)
    w_target = torch.tensor([[3.0e-4, 5.0e-4], [1.0e-4, 8.0e-4]])

    # Generate synthetic dataset
    x = torch.randn(batch_size, in_features)
    y_target = x @ w_target

    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)  # Reasonable LR for weight range
    criterion = nn.MSELoss()

    initial_loss = 0.0
    for epoch in range(10):
        optimizer.zero_grad()
        y_pred = layer(x)
        loss = criterion(y_pred, y_target)
        if epoch == 0:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    # Loss should decrease
    assert final_loss < initial_loss
