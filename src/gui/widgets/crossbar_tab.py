"""Crossbar array simulation tab widget.

Provides controls to configure and run MNA simulations on memristive crossbar
arrays, visualizing junction voltage drop heatmaps and conductance maps.
"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.workers import CrossbarWorker


class CrossbarTab(QWidget):
    """Tab widget for crossbar array MNA simulation and visualization."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initializes the CrossbarTab.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self.worker = None
        self.thread = None
        self._init_ui()

    def _init_ui(self) -> None:
        """Builds the crossbar tab UI."""
        layout = QVBoxLayout(self)

        top_layout = QHBoxLayout()

        # Array configuration
        array_group = QGroupBox("Array Configuration")
        array_layout = QFormLayout(array_group)

        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(2, 64)
        self.rows_spin.setValue(8)
        array_layout.addRow("Rows (Wordlines):", self.rows_spin)

        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(2, 64)
        self.cols_spin.setValue(8)
        array_layout.addRow("Columns (Bitlines):", self.cols_spin)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["VTEAM", "Yakopcic"])
        array_layout.addRow("Device Model:", self.model_combo)

        top_layout.addWidget(array_group)

        # Parasitic configuration
        parasitic_group = QGroupBox("Parasitic Parameters")
        parasitic_layout = QFormLayout(parasitic_group)

        self.r_line_spin = QDoubleSpinBox()
        self.r_line_spin.setRange(0.01, 100.0)
        self.r_line_spin.setValue(1.5)
        self.r_line_spin.setSuffix(" Ω")
        parasitic_layout.addRow("Line Resistance:", self.r_line_spin)

        self.r_source_spin = QDoubleSpinBox()
        self.r_source_spin.setRange(1.0, 10000.0)
        self.r_source_spin.setValue(100.0)
        self.r_source_spin.setSuffix(" Ω")
        parasitic_layout.addRow("Source Resistance:", self.r_source_spin)

        self.r_load_spin = QDoubleSpinBox()
        self.r_load_spin.setRange(1.0, 10000.0)
        self.r_load_spin.setValue(100.0)
        self.r_load_spin.setSuffix(" Ω")
        parasitic_layout.addRow("Load Resistance:", self.r_load_spin)

        self.v_read_spin = QDoubleSpinBox()
        self.v_read_spin.setRange(0.01, 5.0)
        self.v_read_spin.setValue(0.5)
        self.v_read_spin.setSuffix(" V")
        parasitic_layout.addRow("Read Voltage:", self.v_read_spin)

        top_layout.addWidget(parasitic_group)

        layout.addLayout(top_layout)

        # Run button
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("▶  Run Crossbar Analysis")
        self.run_btn.setObjectName("runCrossbarButton")
        self.run_btn.setMinimumHeight(36)
        self.run_btn.clicked.connect(self._run_analysis)
        btn_layout.addWidget(self.run_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        btn_layout.addWidget(self.progress_bar)
        layout.addLayout(btn_layout)

        # Dual heatmap plot
        plot_group = QGroupBox("Crossbar Spatial Analysis")
        plot_layout = QVBoxLayout(plot_group)
        self.figure = Figure(figsize=(10, 4.5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        layout.addWidget(plot_group, 1)

        # Status
        self.status_label = QLabel("Ready. Configure the crossbar array and run analysis.")
        layout.addWidget(self.status_label)

    def _get_model_class(self) -> type:
        """Resolves the compact model class.

        Returns:
            type: Model class.
        """
        import importlib

        mapping = {
            "VTEAM": "compact_models.vteam.VTEAMModel",
            "Yakopcic": "compact_models.yakopcic.YakopcicModel",
        }
        class_path = mapping[self.model_combo.currentText()]
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def _get_default_params(self) -> dict:
        """Returns default parameters.

        Returns:
            dict: Default params.
        """
        name = self.model_combo.currentText()
        if name == "VTEAM":
            return {
                "r_on": 1000.0,
                "r_off": 100000.0,
                "v_on": 0.8,
                "v_off": -0.8,
                "k_on": -10.0,
                "k_off": 10.0,
                "w_on": 0.0,
                "w_off": 1e-9,
                "a_on": 1e-9,
                "a_off": 1e-9,
            }
        else:
            return {
                "a1": 0.17,
                "a2": 0.17,
                "b": 0.05,
                "Ap": 4000.0,
                "An": 4000.0,
                "Vp": 0.16,
                "Vn": 0.15,
                "alphap": 1.0,
                "alphan": 5.0,
                "xp": 0.3,
                "xn": 0.5,
                "eta": 1.0,
                "w_min": 0.0,
                "w_max": 1.0,
            }

    def _run_analysis(self) -> None:
        """Starts the crossbar MNA analysis."""
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.information(self, "Busy", "Analysis is already running.")
            return

        rows = self.rows_spin.value()
        cols = self.cols_spin.value()
        model_class = self._get_model_class()
        base_params = self._get_default_params()

        # Apply read voltage to all rows
        row_voltages = np.full(rows, self.v_read_spin.value())

        crossbar_config = {
            "line_resistance": self.r_line_spin.value(),
            "source_resistance": self.r_source_spin.value(),
            "load_resistance": self.r_load_spin.value(),
        }

        device_config = {
            "d2d": {
                "enabled": True,
                "parameters": {
                    "r_on": {"dist": "lognormal", "std": 0.05},
                    "r_off": {"dist": "lognormal", "std": 0.05},
                },
            },
            "c2c": {"enabled": False},
            "drift": {"enabled": False},
            "noise": {"thermal": False, "shot": False},
        }

        self.worker = CrossbarWorker(
            rows=rows,
            cols=cols,
            model_class=model_class,
            base_params=base_params,
            row_voltages=row_voltages,
            device_config=device_config,
            crossbar_config=crossbar_config,
        )

        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.signals.progress.connect(self._on_progress)
        self.worker.signals.result.connect(self._on_result)
        self.worker.signals.error.connect(self._on_error)
        self.worker.signals.finished.connect(self.thread.quit)

        self.run_btn.setEnabled(False)
        self.status_label.setText("Running crossbar MNA analysis...")
        self.thread.start()

    def _on_progress(self, pct: int, msg: str) -> None:
        """Handles progress.

        Args:
            pct: Percentage.
            msg: Status message.
        """
        self.progress_bar.setValue(pct)
        self.status_label.setText(msg)

    def _on_result(self, result: dict) -> None:
        """Handles crossbar analysis results.

        Args:
            result: Result dictionary with voltage and conductance maps.
        """
        self.run_btn.setEnabled(True)
        self.progress_bar.setValue(100)

        v_drop = result["v_drop"]
        g_map = result["g_map"]

        self.figure.clear()

        # Voltage drop heatmap
        ax1 = self.figure.add_subplot(121)
        im1 = ax1.imshow(v_drop, cmap="plasma", aspect="auto")
        ax1.set_title("Junction Voltage Drop (V)", fontweight="bold")
        ax1.set_xlabel("Bitline Index")
        ax1.set_ylabel("Wordline Index")
        self.figure.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # Conductance map
        ax2 = self.figure.add_subplot(122)
        im2 = ax2.imshow(g_map * 1e3, cmap="viridis", aspect="auto")
        ax2.set_title("Conductance Map (mS)", fontweight="bold")
        ax2.set_xlabel("Bitline Index")
        ax2.set_ylabel("Wordline Index")
        self.figure.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        self.figure.tight_layout()
        self.canvas.draw()

        rows = v_drop.shape[0]
        cols = v_drop.shape[1]
        self.status_label.setText(f"✓ Crossbar analysis complete — {rows}×{cols} array solved.")

    def _on_error(self, msg: str) -> None:
        """Handles errors.

        Args:
            msg: Error message.
        """
        self.run_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        QMessageBox.critical(self, "Crossbar Error", f"Analysis failed:\n{msg}")
        self.status_label.setText(f"✗ Error: {msg}")
