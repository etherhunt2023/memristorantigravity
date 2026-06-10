"""PyTorch integration package for hardware-aware memristive layers.

This package provides custom PyTorch autograd functions and neural network layers
that wrap the CrossbarArray simulator, allowing direct hardware-aware training (HAT)
of memristive synapses using backpropagation.
"""

from pytorch.layer import CrossbarFunction, MemristorLinear
from pytorch.neuron import SurrogateHeaviside, TorchLIFNeurons

__all__ = ["CrossbarFunction", "MemristorLinear", "SurrogateHeaviside", "TorchLIFNeurons"]
