"""GUI tab widget subpackage.

Exports all tab widgets for the main window's tabbed interface.
"""

from gui.widgets.compact_model_tab import CompactModelTab
from gui.widgets.comsol_tab import COMSOLTab
from gui.widgets.crossbar_tab import CrossbarTab
from gui.widgets.device_tab import DeviceTab
from gui.widgets.extraction_tab import ExtractionTab
from gui.widgets.fitting_tab import FittingTab
from gui.widgets.snn_tab import SNNTab

__all__ = [
    "COMSOLTab",
    "ExtractionTab",
    "CompactModelTab",
    "FittingTab",
    "DeviceTab",
    "CrossbarTab",
    "SNNTab",
]
