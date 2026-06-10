"""PyTorch-compatible spiking neuron models with surrogate gradient backpropagation.

This module provides the TorchLIFNeurons class and surrogate gradient functions
enabling end-to-end backpropagation through time (BPTT) for training memristive SNNs.
"""

from typing import Any

import torch
import torch.nn as nn


class SurrogateHeaviside(torch.autograd.Function):
    """Custom autograd function implementing surrogate gradients for spiking threshold.

    Forward: returns Heaviside step output (0 or 1).
    Backward: approximates the derivative using the Fast Sigmoid function.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, alpha: float = 10.0) -> torch.Tensor:
        """Forward pass for Heaviside step function.

        Args:
            ctx: Autograd context.
            x: Input tensor (typically v - v_thresh).
            alpha: Slope parameter for surrogate gradient.

        Returns:
            torch.Tensor: Spikes tensor (0.0 or 1.0).
        """
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0.0).float()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Backward pass approximating the gradient with Fast Sigmoid.

        Args:
            ctx: Autograd context with saved tensors.
            grad_output: incoming gradient.

        Returns:
            Tuple: gradient for input x, and None for alpha.
        """
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        # Fast Sigmoid derivative: f'(x) = alpha / (1 + alpha * |x|)^2
        # We scale the incoming gradient by this derivative
        grad_x = grad_output * (alpha / (1.0 + alpha * torch.abs(x)).pow(2))
        return grad_x, None


class TorchLIFNeurons(nn.Module):
    """Leaky Integrate-and-Fire (LIF) spiking neuron population in PyTorch.

    Supports backpropagation through time (BPTT) using surrogate gradients.
    """

    def __init__(
        self,
        num_neurons: int,
        v_thresh: float = 1.0,
        v_rest: float = 0.0,
        v_reset: float = 0.0,
        leak: float = 0.95,
        r_membrane: float = 1.0e4,
        t_refractory: float = 2.0e-3,
        alpha: float = 10.0,
    ) -> None:
        """Initializes the TorchLIFNeurons module.

        Args:
            num_neurons: Number of neurons in this population.
            v_thresh: Spiking threshold voltage (V).
            v_rest: Resting membrane potential (V).
            v_reset: Reset membrane potential after spike (V).
            leak: Decay factor of membrane potential per time step (0 to 1).
            r_membrane: Membrane resistance (Ohms).
            t_refractory: Refractory period duration (s).
            alpha: Surrogate gradient slope parameter.
        """
        super().__init__()
        self.num_neurons = num_neurons
        self.v_thresh = v_thresh
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.leak = leak
        self.r_membrane = r_membrane
        self.t_refractory = t_refractory
        self.alpha = alpha

        # State variables (initialized dynamically during forward pass)
        self.v = None
        self.refractory_timers = None

    def reset(self) -> None:
        """Resets state variables to force re-initialization on next forward pass."""
        self.v = None
        self.refractory_timers = None

    def forward(self, i_in: torch.Tensor, dt: float = 1.0e-3) -> torch.Tensor:
        """Steps membrane potential forward by dt under input current i_in.

        Args:
            i_in: Input current tensor of shape (batch_size, num_neurons).
            dt: Time step duration (s).

        Returns:
            torch.Tensor: Binary spikes tensor of shape (batch_size, num_neurons).
        """
        batch_size = i_in.shape[0]

        # Initialize state variables dynamically if not set or batch size changed
        if self.v is None or self.v.shape[0] != batch_size or self.v.device != i_in.device:
            self.v = torch.full(
                (batch_size, self.num_neurons), self.v_rest, dtype=i_in.dtype, device=i_in.device
            )
            self.refractory_timers = torch.zeros(
                (batch_size, self.num_neurons), dtype=i_in.dtype, device=i_in.device
            )

        # Update refractory timers
        self.refractory_timers = torch.clamp(self.refractory_timers - dt, min=0.0)

        # Integrate membrane potential for non-refractory neurons
        # V(t+dt) = V_rest + (V(t) - V_rest) * leak + I_in * R_mem * (1 - leak)
        v_decayed = (
            self.v_rest
            + (self.v - self.v_rest) * self.leak
            + i_in * self.r_membrane * (1.0 - self.leak)
        )

        # Apply integration only to non-refractory neurons
        self.v = torch.where(self.refractory_timers == 0.0, v_decayed, self.v)

        # Generate output spikes using surrogate gradient function
        spikes = SurrogateHeaviside.apply(self.v - self.v_thresh, self.alpha)

        # Apply hard reset to voltages and set refractory timers for firing neurons
        self.v = torch.where(spikes > 0.5, torch.full_like(self.v, self.v_reset), self.v)
        self.refractory_timers = torch.where(
            spikes > 0.5,
            torch.full_like(self.refractory_timers, self.t_refractory),
            self.refractory_timers,
        )

        return spikes
