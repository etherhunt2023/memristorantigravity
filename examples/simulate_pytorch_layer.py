"""Example script demonstrating PyTorch integration for Memristor Crossbars.

Performs hardware-aware training (HAT) on a MemristorLinear layer to learn a target
linear relationship. Then evaluates the trained layer under both ideal conditions
and physical MNA solver mode with wire parasitics, visualizing the convergence
and the effect of non-idealities.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from compact_models.vteam import VTEAMModel
from pytorch.layer import MemristorLinear
from utils.logger import get_logger

logger = get_logger("simulate_pytorch_layer")


def run_pytorch_simulation() -> None:
    """Runs the training loop and evaluations for MemristorLinear."""
    print("==================================================")
    print("HARDWARE-AWARE TRAINING WITH MEMRISTOR LINEAR LAYER")
    print("==================================================")

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Define device and crossbar configurations
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

    # Configuration for physical crossbar
    crossbar_config = {
        "line_resistance": 5.0,  # 5 Ohms wire resistance per segment
        "source_resistance": 50.0,  # 50 Ohms driver impedance
        "load_resistance": 100.0,  # 100 Ohms load/sense resistance
    }

    # Device configuration: disable noises during parameter learning for clean gradients
    device_config = {
        "d2d": {"enabled": False},
        "c2c": {"enabled": False},
        "drift": {"enabled": False},
        "noise": {"thermal": False, "shot": False, "generic_std": 0.0},
    }

    # Problem dimensions: 4 inputs -> 2 outputs
    in_features = 4
    out_features = 2
    batch_size = 16

    # 2. Define target weights (conductances in S)
    # Target conductances are chosen within physical bounds [1e-5, 1e-3]
    w_target = torch.tensor(
        [
            [2.0e-4, 8.0e-4],
            [5.0e-4, 1.0e-4],
            [1.5e-4, 6.0e-4],
            [7.0e-4, 3.0e-4],
        ],
        dtype=torch.float32,
    )

    print("Target Weights (Conductances, S):")
    print(w_target.numpy())
    print()

    # Generate synthetic training and testing datasets
    # Inputs representing voltages (0 to 0.2V) to prevent programming/state disturbance
    x_train = torch.rand(batch_size, in_features) * 0.2
    y_train = x_train @ w_target

    x_test = torch.rand(8, in_features) * 0.2
    y_test = x_test @ w_target

    # 3. Instantiate the MemristorLinear layer (training in ideal mode)
    layer = MemristorLinear(
        in_features=in_features,
        out_features=out_features,
        model_class=VTEAMModel,
        base_params=base_params,
        device_config=device_config,
        crossbar_config=crossbar_config,
        use_mna=False,  # Train using fast ideal gradients
        bias=False,
    )

    initial_weights = layer.weight.detach().clone().numpy()
    print("Initial Weights (Conductances, S):")
    print(initial_weights)
    print()

    # 4. Training Loop
    epochs = 100
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    loss_history = []

    print("Starting training...")
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass (ideal mode)
        y_pred = layer(x_train)
        loss = criterion(y_pred, y_train)

        # Backward pass & Optimize
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4e}")

    print("\nTraining completed!")
    final_weights = layer.weight.detach().numpy()
    print("Final Trained Weights (Conductances, S):")
    print(final_weights)
    print()

    # 5. Evaluate Trained Model under MNA (with parasitics) vs Ideal
    print("Evaluating trained weights...")

    # Fast evaluation in ideal mode
    layer.use_mna = False
    with torch.no_grad():
        y_test_pred_ideal = layer(x_test)
        loss_ideal = criterion(y_test_pred_ideal, y_test).item()

    # Evaluation in MNA mode (incorporates line parasitics & load resistance)
    layer.use_mna = True
    with torch.no_grad():
        y_test_pred_mna = layer(x_test)
        loss_mna = criterion(y_test_pred_mna, y_test).item()

    print(f"Ideal Inference Test Loss: {loss_ideal:.4e}")
    print(f"MNA Inference Test Loss: {loss_mna:.4e}")
    print(
        "Note: The MNA test loss is typically higher due to parasitic "
        "voltage drops (IR drop) along the array wires."
    )
    print()

    # 6. Plot results
    temp_dir = Path(__file__).resolve().parent / "temp_data"
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_path = temp_dir / "pytorch_training_convergence.png"

    # Set up matplotlib figure
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)

    # Subplot 1: Training Loss Convergence
    ax_loss = fig.add_subplot(gs[0, :])
    ax_loss.plot(
        range(1, epochs + 1),
        loss_history,
        color="#1f77b4",
        linewidth=2.5,
        label="Training Loss (Ideal)",
    )
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax_loss.set_ylabel("Mean Squared Error (Log Scale)", fontsize=12, fontweight="bold")
    ax_loss.set_title(
        "Training Loss Convergence over Epochs", fontsize=14, fontweight="bold", pad=10
    )
    ax_loss.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_loss.legend(loc="upper right", fontsize=11)

    # Subplot 2: Heatmap comparisons
    cmap = "viridis"
    vmin = min(w_target.min().item(), initial_weights.min(), final_weights.min())
    vmax = max(w_target.max().item(), initial_weights.max(), final_weights.max())

    ax_init = fig.add_subplot(gs[1, 0])
    im_init = ax_init.imshow(initial_weights, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax_init.set_title("Initial Weights", fontsize=12, fontweight="bold")
    ax_init.set_xlabel("Outputs")
    ax_init.set_ylabel("Inputs")
    fig.colorbar(im_init, ax=ax_init, shrink=0.8)

    ax_final = fig.add_subplot(gs[1, 1])
    im_final = ax_final.imshow(final_weights, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax_final.set_title("Final Learned Weights", fontsize=12, fontweight="bold")
    ax_final.set_xlabel("Outputs")
    fig.colorbar(im_final, ax=ax_final, shrink=0.8)

    ax_target = fig.add_subplot(gs[1, 2])
    im_target = ax_target.imshow(w_target.numpy(), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax_target.set_title("Target Weights", fontsize=12, fontweight="bold")
    ax_target.set_xlabel("Outputs")
    fig.colorbar(im_target, ax=ax_target, shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(
        f"Plots saved successfully!\n"
        f"View the convergence chart here:\n"
        f"![Training Convergence](file:///{output_path.as_posix()})"
    )
    print("==================================================\n")


if __name__ == "__main__":
    run_pytorch_simulation()
