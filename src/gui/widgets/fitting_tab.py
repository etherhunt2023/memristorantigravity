"""Model fitting tab widget.

Runs the two-stage optimization pipeline (Differential Evolution + Nelder-Mead)
to fit compact model parameters to target COMSOL data, with real-time progress
and comparison plots.
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
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.workers import FittingWorker


class FittingTab(QWidget):
    """Tab widget for parameter fitting with progress tracking."""

    MODEL_DEFAULTS = {
        "VTEAM": {
            "class_path": "compact_models.vteam.VTEAMModel",
            "fit_keys": ["k_on", "k_off", "v_on", "v_off"],
            "bounds": {
                "k_on": (-50.0, -0.1),
                "k_off": (0.1, 50.0),
                "v_on": (0.1, 2.0),
                "v_off": (-2.0, -0.1),
            },
            "fixed": {
                "r_on": 1000.0,
                "r_off": 100000.0,
                "w_on": 0.0,
                "w_off": 1e-9,
                "a_on": 1e-9,
                "a_off": 1e-9,
            },
        },
        "Yakopcic": {
            "class_path": "compact_models.yakopcic.YakopcicModel",
            "fit_keys": ["Ap", "An", "Vp", "Vn"],
            "bounds": {
                "Ap": (100.0, 50000.0),
                "An": (100.0, 50000.0),
                "Vp": (0.01, 1.0),
                "Vn": (0.01, 1.0),
            },
            "fixed": {
                "a1": 0.17,
                "a2": 0.17,
                "b": 0.05,
                "alphap": 1.0,
                "alphan": 5.0,
                "xp": 0.3,
                "xn": 0.5,
                "eta": 1.0,
                "w_min": 0.0,
                "w_max": 1.0,
            },
        },
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initializes the FittingTab.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self.worker = None
        self.thread = None
        self._init_ui()

    def _init_ui(self) -> None:
        """Builds the fitting tab UI."""
        layout = QVBoxLayout(self)

        # Config
        config_group = QGroupBox("Fitting Configuration")
        config_layout = QFormLayout(config_group)

        self.model_combo = QComboBox()
        self.model_combo.addItems(list(self.MODEL_DEFAULTS.keys()))
        config_layout.addRow("Target Model:", self.model_combo)

        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.0, 1.0)
        self.gamma_spin.setValue(0.3)
        self.gamma_spin.setDecimals(2)
        config_layout.addRow("Loss Weight (γ):", self.gamma_spin)

        self.w_init_spin = QDoubleSpinBox()
        self.w_init_spin.setRange(0.0, 1.0)
        self.w_init_spin.setValue(0.0)
        self.w_init_spin.setDecimals(4)
        config_layout.addRow("Initial State (w₀):", self.w_init_spin)

        layout.addWidget(config_group)

        # Action
        btn_layout = QHBoxLayout()
        self.fit_btn = QPushButton("▶  Start Fitting")
        self.fit_btn.setObjectName("fitButton")
        self.fit_btn.setMinimumHeight(36)
        self.fit_btn.clicked.connect(self._run_fitting)
        btn_layout.addWidget(self.fit_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        btn_layout.addWidget(self.progress_bar)
        layout.addLayout(btn_layout)

        # Results: fitted params table + comparison plot
        results_layout = QHBoxLayout()

        # Fitted params table
        table_group = QGroupBox("Fitted Parameters")
        table_layout = QVBoxLayout(table_group)
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Parameter", "Fitted Value"])
        self.results_table.setAlternatingRowColors(True)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        table_layout.addWidget(self.results_table)

        self.loss_label = QLabel("Loss: —")
        table_layout.addWidget(self.loss_label)
        results_layout.addWidget(table_group)

        # Comparison plot
        plot_group = QGroupBox("Fitting Comparison")
        plot_layout = QVBoxLayout(plot_group)
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        results_layout.addWidget(plot_group, 1)

        layout.addLayout(results_layout, 1)

        # Status
        self.status_label = QLabel(
            "Ready. Parse COMSOL data and run fitting against a compact model."
        )
        layout.addWidget(self.status_label)

    def _get_model_class(self, model_name: str) -> type:
        """Resolves the model class.

        Args:
            model_name: Model registry key.

        Returns:
            type: Compact model class.
        """
        class_path = self.MODEL_DEFAULTS[model_name]["class_path"]
        module_path, class_name = class_path.rsplit(".", 1)
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def _run_fitting(self) -> None:
        """Starts the fitting pipeline in a background thread."""
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.information(self, "Busy", "Fitting is already running.")
            return

        # Try to get COMSOL dataset
        main_window = self.window()
        dataset = None
        if hasattr(main_window, "tabs"):
            for i in range(main_window.tabs.count()):
                widget = main_window.tabs.widget(i)
                if hasattr(widget, "get_dataset") and widget.get_dataset() is not None:
                    dataset = widget.get_dataset()
                    break

        model_name = self.model_combo.currentText()
        model_info = self.MODEL_DEFAULTS[model_name]

        # Generate synthetic target data if no COMSOL dataset is loaded
        if dataset is not None:
            df = dataset.data
            voltages = df["voltage"].to_numpy()
            currents = df["current"].to_numpy()
            time_points = (
                df["time"].to_numpy() if "time" in df.columns else np.linspace(0, 1, len(voltages))
            )
        else:
            # Generate synthetic data for demonstration
            n_pts = 200
            voltages = np.concatenate(
                [
                    np.linspace(0, 1.2, n_pts),
                    np.linspace(1.2, -1.2, 2 * n_pts),
                    np.linspace(-1.2, 0, n_pts),
                ]
            )
            time_points = np.linspace(0, 1, len(voltages))
            model_class = self._get_model_class(model_name)
            true_params = {**model_info["fixed"]}
            for k in model_info["fit_keys"]:
                b = model_info["bounds"][k]
                true_params[k] = (b[0] + b[1]) / 2.0
            model = model_class(true_params)
            _, currents = model.solve_sweep(voltages, time_points, 0.0)
            currents += np.random.normal(0, np.max(np.abs(currents)) * 0.02, len(currents))

        model_class = self._get_model_class(model_name)

        self.worker = FittingWorker(
            model_class=model_class,
            voltages=voltages,
            currents=currents,
            time_points=time_points,
            fit_param_keys=model_info["fit_keys"],
            bounds_dict=model_info["bounds"],
            fixed_params=model_info["fixed"],
            w_init=self.w_init_spin.value(),
            loss_gamma=self.gamma_spin.value(),
        )

        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.signals.progress.connect(self._on_progress)
        self.worker.signals.result.connect(lambda r: self._on_result(r, voltages, currents))
        self.worker.signals.error.connect(self._on_error)
        self.worker.signals.finished.connect(self.thread.quit)

        self.fit_btn.setEnabled(False)
        self.status_label.setText("Fitting in progress... This may take a few minutes.")
        self.thread.start()

    def _on_progress(self, pct: int, msg: str) -> None:
        """Handles progress updates.

        Args:
            pct: Percentage.
            msg: Message.
        """
        self.progress_bar.setValue(pct)
        self.status_label.setText(msg)

    def _on_result(self, result: dict, voltages: np.ndarray, currents: np.ndarray) -> None:
        """Handles fitting results.

        Args:
            result: Fitting output dictionary.
            voltages: Target voltage data.
            currents: Target current data.
        """
        self.fit_btn.setEnabled(True)
        self.progress_bar.setValue(100)

        fitted = result["fitted_params"]
        fit_currents = result["fit_currents"]

        # Populate table
        self.results_table.setRowCount(len(fitted))
        for row_idx, (key, val) in enumerate(fitted.items()):
            self.results_table.setItem(row_idx, 0, QTableWidgetItem(key))
            self.results_table.setItem(row_idx, 1, QTableWidgetItem(f"{val:.6g}"))

        self.loss_label.setText(
            f"Loss — DE: {result['loss_de']:.6f} | Refined: {result['loss_refined']:.6f}"
        )

        # Plot comparison
        self.figure.clear()
        ax_main = self.figure.add_subplot(211)
        ax_main.plot(voltages, currents, "o", label="Target", markersize=2, alpha=0.6)
        ax_main.plot(voltages, fit_currents, "-", label="Fitted Model", linewidth=1.5, color="red")
        ax_main.set_ylabel("Current (A)")
        ax_main.set_title("Experimental vs. Fitted Model", fontweight="bold")
        ax_main.legend()
        ax_main.grid(True, linestyle="--", alpha=0.4)

        ax_res = self.figure.add_subplot(212, sharex=ax_main)
        residuals = currents - fit_currents
        ax_res.plot(voltages, residuals, "x", color="gray", markersize=2, alpha=0.6)
        ax_res.axhline(0, color="red", linestyle="--", linewidth=0.8)
        ax_res.set_xlabel("Voltage (V)")
        ax_res.set_ylabel("Residual (A)")
        ax_res.grid(True, linestyle="--", alpha=0.4)

        self.figure.tight_layout()
        self.canvas.draw()

        self.status_label.setText("✓ Fitting complete.")

    def _on_error(self, msg: str) -> None:
        """Handles errors.

        Args:
            msg: Error message.
        """
        self.fit_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        QMessageBox.critical(self, "Fitting Error", f"Fitting failed:\n{msg}")
        self.status_label.setText(f"✗ Error: {msg}")
