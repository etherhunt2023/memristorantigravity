"""Synaptic plasticity module.

This package provides classes to simulate Spike-Timing-Dependent Plasticity (STDP)
mechanisms, including analytical models and pulse-based physical models.
"""

from plasticity.stdp import PhenomenologicalSTDP, PulseBasedSTDP

__all__ = ["PhenomenologicalSTDP", "PulseBasedSTDP"]
