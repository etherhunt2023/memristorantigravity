"""Spiking Neural Network (SNN) simulation package.

This package provides Leaky Integrate-and-Fire neuron models and crossbar-based
synaptic layer connection interfaces.
"""

from snn.neuron import LIFNeurons
from snn.synapse import CrossbarSynapse

__all__ = ["LIFNeurons", "CrossbarSynapse"]
