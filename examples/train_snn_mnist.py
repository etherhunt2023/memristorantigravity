"""Example script demonstrating SNN training on MNIST.

Rate-encodes MNIST digits into Poisson spike trains, processes them through a
MemristorLinear synaptic layer and a population of LIF neurons, trains the network
using Backpropagation Through Time (BPTT) with surrogate gradients, and plots
the loss/accuracy convergence.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from compact_models.vteam import VTEAMModel
from pytorch.layer import MemristorLinear
from pytorch.neuron import TorchLIFNeurons
from utils.logger import get_logger

logger = get_logger("train_snn_mnist")


class SyntheticMNISTDataset(Dataset):
    """Fallback synthetic dataset mimicking MNIST-like digits for offline running."""

    def __init__(self, num_samples: int = 1000) -> None:
        """Initializes the synthetic dataset.

        Args:
            num_samples: Number of samples to generate.
        """
        # Create random patterns representing digits
        self.data = torch.rand(num_samples, 28, 28) * 0.1
        self.targets = torch.randint(0, 10, (num_samples,))

        # Inject some structured shapes (crosses, boxes) to make it learnable
        for i in range(num_samples):
            digit = self.targets[i].item()
            # Draw a horizontal line for even numbers
            if digit % 2 == 0:
                self.data[i, 14, 5:23] = 0.8
            # Draw a vertical line for odd numbers
            else:
                self.data[i, 5:23, 14] = 0.8
            # Draw extra details based on digit modulo
            if digit < 5:
                self.data[i, 5:10, 5:10] = 0.7
            else:
                self.data[i, 18:23, 18:23] = 0.7

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Returns the data and label at the specified index."""
        return self.data[idx], int(self.targets[idx])


def load_mnist_data(batch_size: int = 64) -> tuple[DataLoader, DataLoader, bool]:
    """Loads MNIST using torchvision, falling back to a synthetic dataset if needed.

    Args:
        batch_size: DataLoader batch size.

    Returns:
        Tuple: (train_loader, test_loader, is_synthetic_flag)
    """
    try:
        import torchvision
        import torchvision.transforms as transforms

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))]
        )
        # Try loading real MNIST with download=True
        train_set = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_set = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )

        # Limit dataset size for fast training demonstration (1200 train, 300 test)
        indices_train = list(range(1200))
        indices_test = list(range(300))
        train_set = torch.utils.data.Subset(train_set, indices_train)
        test_set = torch.utils.data.Subset(test_set, indices_test)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        print("Successfully loaded MNIST via torchvision.")
        return train_loader, test_loader, False

    except Exception as e:
        print(f"Could not load MNIST via torchvision ({e}). Falling back to synthetic dataset.")
        # Generate synthetic MNIST dataset
        train_set = SyntheticMNISTDataset(num_samples=1000)
        test_set = SyntheticMNISTDataset(num_samples=200)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, True


class MemristiveSNN(nn.Module):
    """Spiking Neural Network with a memristive synaptic layer and LIF neurons."""

    def __init__(self, in_features: int, out_features: int, base_params: dict[str, Any]) -> None:
        """Initializes SNN model.

        Args:
            in_features: Number of input channels (e.g. 784).
            out_features: Number of output classes (e.g. 10).
            base_params: Nominal VTEAM device parameters.
        """
        super().__init__()

        # Memristor configuration (no D2D/C2C during parameter updates for clean gradient updates)
        device_config = {
            "d2d": {"enabled": False},
            "c2c": {"enabled": False},
            "drift": {"enabled": False},
            "noise": {"thermal": False, "shot": False, "generic_std": 0.0},
        }

        # Memristor linear synaptic layer
        self.synapse = MemristorLinear(
            in_features=in_features,
            out_features=out_features,
            model_class=VTEAMModel,
            base_params=base_params,
            device_config=device_config,
            use_mna=False,  # Fast ideal matrix VMM for training
            bias=False,
        )

        # Output LIF neurons
        self.neurons = TorchLIFNeurons(
            num_neurons=out_features,
            v_thresh=0.8,
            v_rest=0.0,
            v_reset=0.0,
            leak=0.9,
            r_membrane=2.0e3,
            t_refractory=2.0e-3,
            alpha=10.0,
        )

    def forward(self, x: torch.Tensor, num_steps: int = 15, dt: float = 1.0e-3) -> torch.Tensor:
        """Performs forward pass over multiple timesteps (BPTT).

        Args:
            x: Input images of shape (batch_size, in_features).
            num_steps: Simulation time steps.
            dt: Simulation time step size (s).

        Returns:
            torch.Tensor: Summed output spikes over all timesteps (batch_size, out_features).
        """
        batch_size = x.shape[0]
        self.neurons.reset()

        # Poisson rate coding: convert pixel intensities to spike probabilities
        # Normal inputs are clamped between 0 and 0.2V (read voltage limit)
        max_voltage = 0.2
        input_voltages = x * max_voltage

        summed_spikes = torch.zeros(batch_size, self.neurons.num_neurons, device=x.device)

        for _ in range(num_steps):
            # Sample Poisson spikes: if random value < input probability, spike is active (1)
            spike_mask = (torch.rand_like(input_voltages) < input_voltages).float()

            # Synaptic current
            i_syn = self.synapse(spike_mask)

            # Post-synaptic LIF response
            spikes = self.neurons(i_syn, dt=dt)
            summed_spikes += spikes

        return summed_spikes


def train_snn() -> None:
    """Trains the MemristiveSNN on MNIST/Synthetic dataset."""
    print("==================================================")
    print("TRAINING MEMRISTIVE SNN ON IMAGE RECOGNITION TASK")
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
        "r_on": 1000.0,  # 1k Ohm LRS
        "r_off": 100000.0,  # 100k Ohm HRS
        "d": 3.0e-9,
        "p": 4.0,
    }

    # Data loaders
    batch_size = 64
    train_loader, test_loader, is_synthetic = load_mnist_data(batch_size=batch_size)

    # Dimensions
    in_features = 784  # 28x28 images
    out_features = 10  # 10 digits

    # Model, Optimizer, Loss
    model = MemristiveSNN(in_features, out_features, base_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    epochs = 15
    num_steps = 15  # timesteps for BPTT

    train_loss_history = []
    test_acc_history = []

    print("\nStarting training loop (BPTT)...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_batches = 0

        for data, targets in train_loader:
            optimizer.zero_grad()

            # Flatten image: (batch_size, 1, 28, 28) -> (batch_size, 784)
            data = data.view(data.size(0), -1)

            # Forward pass
            summed_spikes = model(data, num_steps=num_steps)

            # Compute classification loss
            loss = criterion(summed_spikes, targets)

            # Backward pass & update weights
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_batches += 1

        avg_loss = running_loss / total_batches
        train_loss_history.append(avg_loss)

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, targets in test_loader:
                data = data.view(data.size(0), -1)
                summed_spikes = model(data, num_steps=num_steps)
                predictions = summed_spikes.argmax(dim=1)
                correct += (predictions == targets).sum().item()
                total += targets.size(0)

        accuracy = 100.0 * correct / total
        test_acc_history.append(accuracy)

        print(
            f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.4f} - Test Accuracy: {accuracy:.2f}%"
        )

    print("\nTraining completed!")

    # 6. Save Plot
    temp_dir = Path(__file__).resolve().parent / "temp_data"
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_path = temp_dir / "mnist_snn_training.png"

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = "#d62728"
    ax1.set_xlabel("Epochs", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Loss", color=color, fontsize=12, fontweight="bold")
    line1 = ax1.plot(
        range(1, epochs + 1), train_loss_history, color=color, linewidth=2, label="Train Loss"
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color = "#1f77b4"
    ax2.set_ylabel("Accuracy (%)", color=color, fontsize=12, fontweight="bold")
    line2 = ax2.plot(
        range(1, epochs + 1),
        test_acc_history,
        color=color,
        linewidth=2,
        linestyle="--",
        label="Test Accuracy",
    )
    ax2.tick_params(axis="y", labelcolor=color)

    # Add legends
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="center right", fontsize=11)

    plt.title(
        f"Memristive SNN Training Convergence ({'Synthetic' if is_synthetic else 'MNIST'})",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(
        f"\nTraining plot saved successfully to:\n"
        f"![SNN MNIST Training Plot](file:///{output_path.as_posix()})"
    )
    print("==================================================\n")


if __name__ == "__main__":
    train_snn()
