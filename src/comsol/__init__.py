"""COMSOL parser and dataset management module.

This module exposes the main parser class, the dataset representation,
and exception classes for handling COMSOL exports.
"""

from comsol.dataset import COMSOLDataset
from comsol.parser import COMSOLParser, COMSOLParsingError

__all__ = [
    "COMSOLParser",
    "COMSOLParsingError",
    "COMSOLDataset",
]
