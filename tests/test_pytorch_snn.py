"""Unit tests for the PyTorch-compatible SNN components.

Verifies surrogate gradient threshold crossing, LIF neuron dynamics (integration,
leakage, firing, reset, refractory period), and full BPTT SNN gradient flow.
"""

import pytest
import torch

from compact_models.vteam import VTEAMModel
from pytorch.layer import MemristorLinear
from pytorch.neuron import SurrogateHeaviside, TorchLIFNeurons


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


def test_surrogate_heaviside() -> None:
    """Tests SurrogateHeaviside forward step and surrogate backward gradient."""
    # Test forward pass
    x = torch.tensor([-0.5, 0.0, 0.5], requires_grad=True)
    spikes = SurrogateHeaviside.apply(x, 10.0)
    assert torch.allclose(spikes, torch.tensor([0.0, 1.0, 1.0]))

    # Test backward pass (gradients)
    loss = spikes.sum()
    loss.backward()

    # Fast Sigmoid derivative: alpha / (1 + alpha * |x|)^2
    # For x = -0.5: 10 / (1 + 10 * 0.5)^2 = 10 / 36 approx 0.2778
    # For x = 0.0: 10 / (1 + 0)^2 = 10.0
    # For x = 0.5: 10 / (1 + 10 * 0.5)^2 = 10 / 36 approx 0.2778
    expected_grad = torch.tensor([10.0 / 36.0, 10.0, 10.0 / 36.0])
    assert torch.allclose(x.grad, expected_grad, rtol=1e-4)


def test_torch_lif_neurons_dynamics() -> None:
    """Tests TorchLIFNeurons integration, leak, reset, and refractory periods."""
    num_neurons = 2
    batch_size = 3
    dt = 1e-3

    neurons = TorchLIFNeurons(
        num_neurons=num_neurons,
        v_thresh=1.0,
        v_rest=0.0,
        v_reset=0.1,
        leak=0.9,
        r_membrane=2.0e3,
        t_refractory=2.0e-3,  # 2 ms -> 2 time steps of dt=1ms
    )

    # 1. First step: check dynamic initialization and leaky integration
    # Apply input current of 0.2A to neuron 0, 0A to neuron 1
    i_in = torch.tensor([[0.2, 0.0], [0.2, 0.0], [0.2, 0.0]])
    spikes = neurons(i_in, dt=dt)

    assert spikes.shape == (batch_size, num_neurons)

    # Check membrane potential after integration:
    # V(1) = 0 + (0 - 0)*0.9 + I_in * R_mem * (1 - 0.9)
    # For neuron 0: 0.2 * 2000 * 0.1 = 40.0V (exceeds threshold)
    # Wait, because 40V > v_thresh (1.0), it should spike immediately in this step!
    # Yes! In the forward step, v becomes 40.0, spikes=1.0, and then v is reset to v_reset (0.1).
    assert torch.all(spikes[:, 0] == 1.0)
    assert torch.all(spikes[:, 1] == 0.0)

    # Output potentials should be reset to v_reset (0.1) for neuron 0, and remain 0.0 for neuron 1
    assert torch.allclose(neurons.v[:, 0], torch.tensor(0.1))
    assert torch.allclose(neurons.v[:, 1], torch.tensor(0.0))

    # Refractory timers should be set to t_refractory (2e-3) for neuron 0
    assert torch.allclose(neurons.refractory_timers[:, 0], torch.tensor(2.0e-3))
    assert torch.allclose(neurons.refractory_timers[:, 1], torch.tensor(0.0))

    # 2. Second step (dt = 1ms): refractory period check
    # Apply same input current. Neuron 0 is refractory (timer was 2ms,
    # decremented by 1ms to 1ms > 0).
    # So it should NOT integrate new current and its voltage should remain reset value.
    spikes2 = neurons(i_in, dt=dt)

    assert torch.all(spikes2[:, 0] == 0.0)
    assert torch.allclose(neurons.v[:, 0], torch.tensor(0.1))
    assert torch.allclose(neurons.refractory_timers[:, 0], torch.tensor(1.0e-3))


def test_torch_snn_gradient_flow(base_vteam_params: dict[str, float]) -> None:
    """Tests gradient flow through SNN model using BPTT."""
    torch.manual_seed(42)

    in_features, out_features = 3, 2
    batch_size = 2
    timesteps = 5

    device_config = {
        "d2d": {"enabled": False},
        "c2c": {"enabled": False},
        "drift": {"enabled": False},
        "noise": {"thermal": False, "shot": False, "generic_std": 0.0},
    }

    # SNN layers
    synapse = MemristorLinear(
        in_features=in_features,
        out_features=out_features,
        model_class=VTEAMModel,
        base_params=base_vteam_params,
        device_config=device_config,
        use_mna=False,
        bias=False,
    )

    # Initialize synapse conductances to intermediate values
    with torch.no_grad():
        synapse.weight.fill_(5e-4)

    neurons = TorchLIFNeurons(
        num_neurons=out_features,
        v_thresh=0.5,
        v_rest=0.0,
        v_reset=0.0,
        leak=0.9,
        r_membrane=2.0e3,
        t_refractory=2.0e-3,
    )

    # Run SNN transient simulation
    # Inputs: batch_size x timesteps x in_features (voltages)
    # Let's apply 0.2V input pulses
    x = torch.zeros(batch_size, timesteps, in_features)
    x[:, :, 0] = 0.2  # Input 0 is active

    neurons.reset()
    all_spikes = []

    for t in range(timesteps):
        # Forward pass through synapses (input voltage to current)
        i_syn = synapse(x[:, t, :])
        # Forward pass through output LIF neurons
        spikes = neurons(i_syn, dt=1.0e-3)
        all_spikes.append(spikes)

    # Stack spikes: batch_size x timesteps x out_features
    output_spikes = torch.stack(all_spikes, dim=1)

    # We want output neuron 0 to spike, and neuron 1 to stay quiet
    # Calculate a simple loss
    target = torch.zeros_like(output_spikes)
    target[:, :, 0] = 1.0  # target: spikes on channel 0

    loss = torch.nn.functional.mse_loss(output_spikes, target)
    loss.backward()

    # Gradients must flow back to synapse weight parameter
    assert synapse.weight.grad is not None
    assert torch.isfinite(synapse.weight.grad).all()
    # Check that gradient is non-zero
    assert torch.any(synapse.weight.grad != 0.0)
