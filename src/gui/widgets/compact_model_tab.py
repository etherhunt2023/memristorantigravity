"""Compact model simulation tab widget.

Allows the user to select a physics-based memristor model (VTEAM, Yakopcic,
Simmons), configure parameters, run a voltage sweep simulation, and view
the resulting I-V curves with embedded matplotlib plots.
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
    QVBoxLayout,
    QWidget,
)

from gui.workers import SimulationWorker


class CompactModelTab(QWidget):
    """Tab widget for compact model simulation and I-V curve visualization."""

    # Model registry
    MODEL_REGISTRY = {
        "VTEAM": {
            "class_path": "compact_models.vteam.VTEAMModel",
            "defaults": {
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
            },
        },
        "Yakopcic": {
            "class_path": "compact_models.yakopcic.YakopcicModel",
            "defaults": {
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
            },
        },
        "Simmons Tunneling": {
            "class_path": "compact_models.simmons.SimmonsModel",
            "defaults": {
                "phi_0": 0.6,
                "d_min": 1.0e-9,
                "d_max": 3.0e-9,
                "area": 1.0e-14,
                "v_on": 0.8,
                "v_off": -0.8,
                "k_on": -5.0,
                "k_off": 5.0,
                "w_min": 0.0,
                "w_max": 1.0,
            },
        },
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initializes the CompactModelTab.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self.worker = None
        self.thread = None
        self.param_spins: dict[str, QDoubleSpinBox] = {}
        self._init_ui()
        self._on_model_changed()

    def _init_ui(self) -> None:
        """Builds the compact model tab UI."""
        layout = QVBoxLayout(self)

        # Top: controls
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)

        # Left: model selection and parameters
        left_group = QGroupBox("Model Configuration")
        left_layout = QVBoxLayout(left_group)

        # Model selector
        model_form = QFormLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(self.MODEL_REGISTRY.keys()))
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        model_form.addRow("Compact Model:", self.model_combo)

        self.solver_combo = QComboBox()
        self.solver_combo.addItems(["rk4", "euler"])
        model_form.addRow("ODE Solver:", self.solver_combo)

        self.w_init_spin = QDoubleSpinBox()
        self.w_init_spin.setRange(0.0, 1.0)
        self.w_init_spin.setValue(0.0)
        self.w_init_spin.setDecimals(4)
        model_form.addRow("Initial State (w₀):", self.w_init_spin)

        left_layout.addLayout(model_form)

        # Dynamic parameter form
        self.param_form_layout = QFormLayout()
        left_layout.addLayout(self.param_form_layout)

        top_layout.addWidget(left_group)

        # Right: sweep configuration
        right_group = QGroupBox("Voltage Sweep")
        right_layout = QFormLayout(right_group)

        self.v_min_spin = QDoubleSpinBox()
        self.v_min_spin.setRange(-10.0, 0.0)
        self.v_min_spin.setValue(-1.5)
        self.v_min_spin.setDecimals(2)
        self.v_min_spin.setSuffix(" V")
        right_layout.addRow("V_min:", self.v_min_spin)

        self.v_max_spin = QDoubleSpinBox()
        self.v_max_spin.setRange(0.0, 10.0)
        self.v_max_spin.setValue(1.5)
        self.v_max_spin.setDecimals(2)
        self.v_max_spin.setSuffix(" V")
        right_layout.addRow("V_max:", self.v_max_spin)

        self.n_points_spin = QDoubleSpinBox()
        self.n_points_spin.setRange(50, 5000)
        self.n_points_spin.setValue(400)
        self.n_points_spin.setDecimals(0)
        right_layout.addRow("Points per segment:", self.n_points_spin)

        top_layout.addWidget(right_group)

        layout.addWidget(top_widget)

        # Run button and progress bar
        run_layout = QHBoxLayout()
        self.run_btn = QPushButton("▶  Run Simulation")
        self.run_btn.setObjectName("runSimButton")
        self.run_btn.setMinimumHeight(36)
        self.run_btn.clicked.connect(self._run_simulation)
        run_layout.addWidget(self.run_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        run_layout.addWidget(self.progress_bar)
        layout.addLayout(run_layout)

        # Matplotlib canvas for I-V plot
        plot_group = QGroupBox("I-V Characteristic")
        plot_layout = QVBoxLayout(plot_group)

        self.figure = Figure(figsize=(7, 4.5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)

        layout.addWidget(plot_group, 1)

        # Status
        self.status_label = QLabel("Ready. Configure a model and run the simulation.")
        layout.addWidget(self.status_label)

    def _on_model_changed(self) -> None:
        """Updates parameter spinboxes when the selected model changes."""
        # Clear existing parameter widgets
        while self.param_form_layout.rowCount() > 0:
            self.param_form_layout.removeRow(0)
        self.param_spins.clear()

        model_name = self.model_combo.currentText()
        if model_name not in self.MODEL_REGISTRY:
            return

        defaults = self.MODEL_REGISTRY[model_name]["defaults"]
        for param_name, default_val in defaults.items():
            spin = QDoubleSpinBox()
            spin.setDecimals(6)
            # Set reasonable ranges
            if abs(default_val) < 1e-6:
                spin.setRange(-1e-3, 1e-3)
            elif abs(default_val) < 1.0:
                spin.setRange(-100.0, 100.0)
            elif abs(default_val) < 1e3:
                spin.setRange(-1e6, 1e6)
            else:
                spin.setRange(-1e9, 1e9)
            spin.setValue(default_val)
            self.param_spins[param_name] = spin
            self.param_form_layout.addRow(f"{param_name}:", spin)

    def _get_model_class(self) -> type:
        """Resolves the model class from the registry.

        Returns:
            type: The compact model class.
        """
        model_name = self.model_combo.currentText()
        class_path = self.MODEL_REGISTRY[model_name]["class_path"]
        module_path, class_name = class_path.rsplit(".", 1)
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def _run_simulation(self) -> None:
        """Starts the compact model simulation in a background thread."""
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.information(self, "Busy", "A simulation is already running.")
            return

        # Collect parameters
        params = {k: spin.value() for k, spin in self.param_spins.items()}

        # Build voltage sweep (triangular: 0 → V_max → V_min → 0)
        n_pts = int(self.n_points_spin.value())
        v_max = self.v_max_spin.value()
        v_min = self.v_min_spin.value()

        voltage_sweep = np.concatenate(
            [
                np.linspace(0, v_max, n_pts),
                np.linspace(v_max, v_min, 2 * n_pts),
                np.linspace(v_min, 0, n_pts),
            ]
        )
        time_points = np.linspace(0, 1.0, len(voltage_sweep))

        model_class = self._get_model_class()

        self.worker = SimulationWorker(
            model_class=model_class,
            params=params,
            voltage_sweep=voltage_sweep,
            time_points=time_points,
            w_init=self.w_init_spin.value(),
            solver_type=self.solver_combo.currentText(),
        )

        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.signals.progress.connect(self._on_progress)
        self.worker.signals.result.connect(self._on_result)
        self.worker.signals.error.connect(self._on_error)
        self.worker.signals.finished.connect(self.thread.quit)

        self.run_btn.setEnabled(False)
        self.status_label.setText("Running simulation...")
        self.thread.start()

    def _on_progress(self, pct: int, msg: str) -> None:
        """Handles progress updates from the worker.

        Args:
            pct: Progress percentage (0-100).
            msg: Status message.
        """
        self.progress_bar.setValue(pct)
        self.status_label.setText(msg)

    def _on_result(self, result: dict) -> None:
        """Handles simulation results.

        Args:
            result: Dictionary with voltages, currents, states, time arrays.
        """
        self.run_btn.setEnabled(True)
        self.progress_bar.setValue(100)

        voltages = result["voltages"]
        currents = result["currents"]

        # Plot I-V curve
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        t = np.linspace(0, 1, len(voltages))
        scatter = ax.scatter(voltages, currents, c=t, cmap="viridis", s=3, zorder=3)
        ax.plot(voltages, currents, alpha=0.3, color="gray", linewidth=0.8)
        ax.set_xlabel("Applied Voltage (V)", fontweight="bold")
        ax.set_ylabel("Current (A)", fontweight="bold")
        ax.set_title(
            f"{self.model_combo.currentText()} — I-V Hysteresis Loop",
            fontweight="bold",
        )
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
        ax.axvline(0, color="black", linewidth=0.5, alpha=0.5)
        ax.grid(True, linestyle="--", alpha=0.4)
        self.figure.colorbar(scatter, ax=ax, label="Normalized Sweep Time")
        self.figure.tight_layout()
        self.canvas.draw()

        self.status_label.setText(f"✓ Simulation complete — {len(voltages)} points plotted.")

    def _on_error(self, msg: str) -> None:
        """Handles worker errors.

        Args:
            msg: Error message string.
        """
        self.run_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        QMessageBox.critical(self, "Simulation Error", f"Simulation failed:\n{msg}")
        self.status_label.setText(f"✗ Error: {msg}")
