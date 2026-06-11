"""Unit tests for the GUI module.

Tests widget instantiation, signal wiring, and basic interaction logic
without requiring a display server (uses QApplication in offscreen mode).
"""

import os

import numpy as np
import pytest

# Force offscreen rendering for CI/headless environments
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow
from gui.workers import (
    SimulationWorker,
    SNNWorker,
    WorkerSignals,
)


@pytest.fixture(scope="module")
def qapp():
    """Creates a QApplication instance for tests.

    Returns:
        QApplication: The application instance.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_main_window_creation(qapp) -> None:
    """Verifies that the main window can be instantiated with all tabs."""
    window = MainWindow()
    assert window is not None
    assert window.tabs.count() == 7

    # Check tab titles contain expected keywords
    tab_texts = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert any("COMSOL" in t for t in tab_texts)
    assert any("Extraction" in t for t in tab_texts)
    assert any("Compact" in t for t in tab_texts)
    assert any("Fitting" in t for t in tab_texts)
    assert any("Device" in t for t in tab_texts)
    assert any("Crossbar" in t for t in tab_texts)
    assert any("SNN" in t for t in tab_texts)


def test_comsol_tab_initial_state(qapp) -> None:
    """Verifies the COMSOL tab initializes with empty state."""
    from gui.widgets.comsol_tab import COMSOLTab

    tab = COMSOLTab()
    assert tab.dataset is None
    assert tab.file_path_edit.text() == ""
    assert tab.data_table.rowCount() == 0


def test_extraction_tab_initial_state(qapp) -> None:
    """Verifies the Extraction tab initializes with default configuration."""
    from gui.widgets.extraction_tab import ExtractionTab

    tab = ExtractionTab()
    assert tab.extracted_params is None
    assert tab.lrs_voltage_spin.value() == pytest.approx(0.1)
    assert tab.hrs_voltage_spin.value() == pytest.approx(0.1)


def test_compact_model_tab_model_switching(qapp) -> None:
    """Verifies that switching models updates the parameter form."""
    from gui.widgets.compact_model_tab import CompactModelTab

    tab = CompactModelTab()

    # Default model should have parameters populated
    assert len(tab.param_spins) > 0

    # Switch to Yakopcic
    tab.model_combo.setCurrentText("Yakopcic")
    assert "Ap" in tab.param_spins
    assert "An" in tab.param_spins

    # Switch to VTEAM
    tab.model_combo.setCurrentText("VTEAM")
    assert "r_on" in tab.param_spins
    assert "k_on" in tab.param_spins


def test_fitting_tab_initial_state(qapp) -> None:
    """Verifies the Fitting tab initializes correctly."""
    from gui.widgets.fitting_tab import FittingTab

    tab = FittingTab()
    assert tab.gamma_spin.value() == pytest.approx(0.3)
    assert tab.results_table.rowCount() == 0


def test_device_tab_initial_state(qapp) -> None:
    """Verifies the Device tab initializes with defaults."""
    from gui.widgets.device_tab import DeviceTab

    tab = DeviceTab()
    assert tab.d2d_check.isChecked()
    assert tab.c2c_check.isChecked()
    assert tab.drift_check.isChecked()
    assert tab.thermal_check.isChecked()
    assert tab.shot_check.isChecked()
    assert tab.n_trials_spin.value() == 10


def test_crossbar_tab_initial_state(qapp) -> None:
    """Verifies the Crossbar tab initializes with defaults."""
    from gui.widgets.crossbar_tab import CrossbarTab

    tab = CrossbarTab()
    assert tab.rows_spin.value() == 8
    assert tab.cols_spin.value() == 8
    assert tab.r_line_spin.value() == pytest.approx(1.5)


def test_snn_tab_initial_state(qapp) -> None:
    """Verifies the SNN tab initializes with defaults."""
    from gui.widgets.snn_tab import SNNTab

    tab = SNNTab()
    assert tab.n_inputs_spin.value() == 8
    assert tab.n_outputs_spin.value() == 4
    assert tab.v_thresh_spin.value() == pytest.approx(0.8)


def test_worker_signals_creation() -> None:
    """Verifies that WorkerSignals can be instantiated."""
    signals = WorkerSignals()
    assert hasattr(signals, "started")
    assert hasattr(signals, "progress")
    assert hasattr(signals, "result")
    assert hasattr(signals, "error")
    assert hasattr(signals, "finished")


def test_simulation_worker_creation() -> None:
    """Verifies SimulationWorker can be created with valid inputs."""
    from compact_models.vteam import VTEAMModel

    worker = SimulationWorker(
        model_class=VTEAMModel,
        params={
            "r_on": 1000,
            "r_off": 100000,
            "v_on": 0.8,
            "v_off": -0.8,
            "k_on": -10,
            "k_off": 10,
            "w_on": 0,
            "w_off": 1e-9,
            "a_on": 1e-9,
            "a_off": 1e-9,
        },
        voltage_sweep=np.linspace(-1, 1, 100),
        time_points=np.linspace(0, 1, 100),
        w_init=0.0,
    )
    assert worker is not None


def test_snn_worker_creation() -> None:
    """Verifies SNNWorker can be created with valid inputs."""
    worker = SNNWorker(
        n_inputs=4,
        n_outputs=2,
        n_timesteps=50,
        dt_ms=0.5,
        input_rate=0.1,
    )
    assert worker is not None
