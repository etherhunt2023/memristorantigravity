"""PyTorch integration for memristor crossbars with hardware-aware autograd.

This module provides custom PyTorch autograd functions and neural network layers
that wrap the CrossbarArray simulator, allowing direct hardware-aware training (HAT)
of memristive synapses using backpropagation.
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from compact_models.base import MemristorCompactModel
from crossbar.array import CrossbarArray
from utils.config_loader import Config
from utils.logger import get_logger

logger = get_logger("pytorch")


class CrossbarFunction(torch.autograd.Function):
    """Custom PyTorch autograd function wrapping memristor crossbar solver.

    Forward pass: simulates currents using MNA or idealized VMM.
    Backward pass: backpropagates gradients using straight-through ideal VMM gradients.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weights: torch.Tensor,
        crossbar: CrossbarArray,
        use_mna: bool,
    ) -> torch.Tensor:
        """Forward pass executing the crossbar array solver.

        Args:
            ctx: Autograd context to save variables.
            x: Input voltage/activations tensor of shape (batch_size, in_features).
            weights: Synaptic weight (conductance) tensor of shape (in_features, out_features).
            crossbar: The physical CrossbarArray instance.
            use_mna: If True, solves Modified Nodal Analysis.

        Returns:
            torch.Tensor: Output current tensor of shape (batch_size, out_features).
        """
        # Save tensors for backward pass
        ctx.save_for_backward(x, weights)

        # Convert PyTorch tensor to NumPy array
        x_np = x.detach().cpu().numpy()
        batch_size = x_np.shape[0]
        out_features = weights.shape[1]

        y_np = np.zeros((batch_size, out_features))

        if use_mna:
            # Run MNA solver sample-by-sample for the batch
            for b in range(batch_size):
                v_row = x_np[b]
                # Solve crossbar voltages
                _, v_col = crossbar.solve_mna(v_row, use_nonlinear=True)
                # Output column currents: I = V_terminal / R_load
                y_np[b] = v_col[-1, :] / crossbar.r_load
        else:
            # Idealized mode: Matrix VMM
            weights_np = weights.detach().cpu().numpy()
            y_np = x_np @ weights_np

        # Return output PyTorch tensor
        return torch.tensor(y_np, dtype=x.dtype, device=x.device)

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        """Backward pass computing gradients.

        Uses approximate ideal VMM derivatives (Straight-Through Estimator).

        Args:
            ctx: Autograd context with saved tensors.
            grad_output: Gradients from the next layer (batch_size, out_features).

        Returns:
            Tuple: Gradients for each input to forward pass.
        """
        x, weights = ctx.saved_tensors

        # grad_x = grad_output @ weights^T
        grad_x = grad_output @ weights.t()
        # grad_weights = x^T @ grad_output
        grad_weights = x.t() @ grad_output

        return grad_x, grad_weights, None, None


class MemristorLinear(nn.Module):
    """Custom hardware-aware linear layer mapping PyTorch weights to memristors."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        model_class: type[MemristorCompactModel],
        base_params: dict[str, Any],
        device_config: dict[str, Any] | Config | None = None,
        crossbar_config: dict[str, Any] | Config | None = None,
        use_mna: bool = False,
        bias: bool = True,
    ) -> None:
        """Initializes the MemristorLinear layer.

        Args:
            in_features: Size of input sample (number of wordlines).
            out_features: Size of output sample (number of bitlines).
            model_class: Compact model class (e.g., VTEAMModel).
            base_params: Nominal parameter dictionary.
            device_config: Device non-idealities configuration.
            crossbar_config: Crossbar array configurations.
            use_mna: If True, uses MNA solver in forward pass.
            bias: If True, adds a learnable bias to output.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_mna = use_mna

        # Instantiate physical crossbar
        self.crossbar = CrossbarArray(
            rows=in_features,
            cols=out_features,
            model_class=model_class,
            base_params=base_params,
            device_config=device_config,
            crossbar_config=crossbar_config,
        )

        # Get device limits
        dev0 = self.crossbar.devices[0, 0]
        self.r_on = float(dev0.params.get("r_on", dev0.params.get("R_on", 1.0e3)))
        self.r_off = float(dev0.params.get("r_off", dev0.params.get("R_off", 1.0e5)))
        self.w_on = float(dev0.params.get("w_on", dev0.model.w_min))
        self.w_off = float(dev0.params.get("w_off", dev0.model.w_max))

        # Conductance boundaries
        self.g_min = 1.0 / self.r_off
        self.g_max = 1.0 / self.r_on

        # Define PyTorch weight parameter (conductance G)
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()
        self.sync_weights_to_devices()

        logger.info(
            f"Initialized MemristorLinear layer ({in_features} -> {out_features}). "
            f"use_mna={use_mna}, bias={bias}"
        )

    def reset_parameters(self) -> None:
        """Initializes weight conductances uniformly between g_min and g_max."""
        nn.init.uniform_(self.weight, a=self.g_min, b=self.g_max)
        if self.bias is not None:
            nn.init.uniform_(self.bias, a=-0.01, b=0.01)

    def sync_weights_to_devices(self) -> None:
        """Synchronizes PyTorch weight parameter (conductance) to memristor state variables w."""
        # Clip weight to physical conductance limits
        with torch.no_grad():
            self.weight.clamp_(min=self.g_min, max=self.g_max)

        g_arr = self.weight.detach().cpu().numpy()

        for i in range(self.in_features):
            for j in range(self.out_features):
                g_val = g_arr[i, j]
                r_eff = 1.0 / g_val

                # Map R_eff to state w: w = w_on + (w_off - w_on) * (R_eff - R_on) / (R_off - R_on)
                w_val = self.w_on + (self.w_off - self.w_on) * (r_eff - self.r_on) / (
                    self.r_off - self.r_on + 1.0e-20
                )

                dev = self.crossbar.devices[i, j]
                dev.w = np.clip(w_val, dev.model.w_min, dev.model.w_max)
                dev.w_programmed = dev.w
                dev.time_since_programming = 0.0

    def sync_devices_to_weights(self) -> None:
        """Synchronizes memristor device state variables w back to PyTorch weights (conductance)."""
        g_arr = np.zeros((self.in_features, self.out_features))

        for i in range(self.in_features):
            for j in range(self.out_features):
                dev = self.crossbar.devices[i, j]

                # Map state w to R_eff: R_eff = R_on + (R_off - R_on) * (w - w_on) / (w_off - w_on)
                w_norm = (dev.w - self.w_on) / (self.w_off - self.w_on + 1.0e-20)
                r_eff = self.r_on + (self.r_off - self.r_on) * w_norm
                r_eff = np.clip(r_eff, min(self.r_on, self.r_off), max(self.r_on, self.r_off))

                g_arr[i, j] = 1.0 / r_eff

        with torch.no_grad():
            self.weight.copy_(
                torch.tensor(g_arr, dtype=self.weight.dtype, device=self.weight.device)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass through memristor crossbar layer.

        Args:
            x: Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output current/activations tensor (batch_size, out_features).
        """
        # Synchronize weights from PyTorch parameter to physical devices
        # to ensure MNA solver runs on up-to-date states
        # Clamp weight to physical conductance limits
        with torch.no_grad():
            self.weight.clamp_(min=self.g_min, max=self.g_max)

        if self.use_mna:
            self.sync_weights_to_devices()

        y = CrossbarFunction.apply(x, self.weight, self.crossbar, self.use_mna)

        if self.bias is not None:
            return y + self.bias
        return y
