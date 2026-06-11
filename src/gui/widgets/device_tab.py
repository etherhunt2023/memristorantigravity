"""Hardware-aware device simulation tab widget.

Enables simulation of memristor devices with non-idealities (D2D variation,
C2C noise, drift, thermal/shot noise) and visualizes the resulting I-V curves
with multiple Monte Carlo trials overlaid.
"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QCheckBox,
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


class DeviceTab(QWidget):
    """Tab widget for hardware-aware device simulation with non-idealities."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initializes the DeviceTab.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self.thread = None
        self._init_ui()

    def _init_ui(self) -> None:
        """Builds the device tab UI."""
        layout = QVBoxLayout(self)

        top_layout = QHBoxLayout()

        # Left: model + sweep config
        model_group = QGroupBox("Device Model")
        model_layout = QFormLayout(model_group)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["VTEAM", "Yakopcic", "Simmons Tunneling"])
        model_layout.addRow("Compact Model:", self.model_combo)

        self.n_trials_spin = QSpinBox()
        self.n_trials_spin.setRange(1, 50)
        self.n_trials_spin.setValue(10)
        model_layout.addRow("Monte Carlo Trials:", self.n_trials_spin)

        self.v_max_spin = QDoubleSpinBox()
        self.v_max_spin.setRange(0.1, 10.0)
        self.v_max_spin.setValue(1.5)
        self.v_max_spin.setSuffix(" V")
        model_layout.addRow("V_max:", self.v_max_spin)

        top_layout.addWidget(model_group)

        # Right: non-ideality toggles
        noise_group = QGroupBox("Non-Idealities")
        noise_layout = QFormLayout(noise_group)

        self.d2d_check = QCheckBox("Enable D2D Variation")
        self.d2d_check.setChecked(True)
        noise_layout.addRow(self.d2d_check)

        self.c2c_check = QCheckBox("Enable C2C Noise")
        self.c2c_check.setChecked(True)
        noise_layout.addRow(self.c2c_check)

        self.drift_check = QCheckBox("Enable Resistance Drift")
        self.drift_check.setChecked(True)
        noise_layout.addRow(self.drift_check)

        self.thermal_check = QCheckBox("Enable Thermal Noise")
        self.thermal_check.setChecked(True)
        noise_layout.addRow(self.thermal_check)

        self.shot_check = QCheckBox("Enable Shot Noise")
        self.shot_check.setChecked(True)
        noise_layout.addRow(self.shot_check)

        top_layout.addWidget(noise_group)

        layout.addLayout(top_layout)

        # Run button
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("▶  Run Device Simulation")
        self.run_btn.setObjectName("runDeviceButton")
        self.run_btn.setMinimumHeight(36)
        self.run_btn.clicked.connect(self._run_simulation)
        btn_layout.addWidget(self.run_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        btn_layout.addWidget(self.progress_bar)
        layout.addLayout(btn_layout)

        # Plot area
        plot_group = QGroupBox("Device I-V Response (Monte Carlo Overlay)")
        plot_layout = QVBoxLayout(plot_group)
        self.figure = Figure(figsize=(7, 4.5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        layout.addWidget(plot_group, 1)

        # Status
        self.status_label = QLabel("Ready. Configure non-idealities and run.")
        layout.addWidget(self.status_label)

    def _get_model_class(self) -> type:
        """Resolves the compact model class.

        Returns:
            type: The model class.
        """
        import importlib

        mapping = {
            "VTEAM": "compact_models.vteam.VTEAMModel",
            "Yakopcic": "compact_models.yakopcic.YakopcicModel",
            "Simmons Tunneling": "compact_models.simmons.SimmonsModel",
        }
        class_path = mapping[self.model_combo.currentText()]
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def _get_default_params(self) -> dict:
        """Returns default parameters for the selected model.

        Returns:
            dict: Default parameter dictionary.
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
        elif name == "Yakopcic":
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
        else:
            return {
                "phi_0": 0.6,
                "d_min": 1e-9,
                "d_max": 3e-9,
                "area": 1e-14,
                "v_on": 0.8,
                "v_off": -0.8,
                "k_on": -5.0,
                "k_off": 5.0,
                "w_min": 0.0,
                "w_max": 1.0,
            }

    def _run_simulation(self) -> None:
        """Runs Monte Carlo device simulation."""
        from device.memristor import MemristorDevice

        self.run_btn.setEnabled(False)
        self.status_label.setText("Running Monte Carlo device simulation...")
        self.progress_bar.setValue(10)

        model_class = self._get_model_class()
        base_params = self._get_default_params()
        v_max = self.v_max_spin.value()
        n_pts = 200

        voltage_sweep = np.concatenate(
            [
                np.linspace(0, v_max, n_pts),
                np.linspace(v_max, -v_max, 2 * n_pts),
                np.linspace(-v_max, 0, n_pts),
            ]
        )
        time_points = np.linspace(0, 1.0, len(voltage_sweep))

        device_config = {
            "d2d": {
                "enabled": self.d2d_check.isChecked(),
                "parameters": {
                    "r_on": {"dist": "lognormal", "std": 0.1},
                    "r_off": {"dist": "lognormal", "std": 0.1},
                },
            },
            "c2c": {
                "enabled": self.c2c_check.isChecked(),
                "state_noise_std": 0.02,
            },
            "drift": {
                "enabled": self.drift_check.isChecked(),
                "coeff": 0.05,
                "type": "resistance",
            },
            "noise": {
                "thermal": self.thermal_check.isChecked(),
                "shot": self.shot_check.isChecked(),
                "temperature": 300.0,
                "bandwidth": 1e6,
            },
        }

        n_trials = self.n_trials_spin.value()

        try:
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            for trial in range(n_trials):
                dev = MemristorDevice(model_class, base_params, device_config=device_config)
                _, i_hist = dev.solve_sweep(voltage_sweep, time_points)
                alpha = max(0.15, 0.8 / n_trials)
                ax.plot(voltage_sweep, i_hist, alpha=alpha, linewidth=0.8)
                self.progress_bar.setValue(10 + int(85 * (trial + 1) / n_trials))

            ax.set_xlabel("Applied Voltage (V)", fontweight="bold")
            ax.set_ylabel("Current (A)", fontweight="bold")
            ax.set_title(
                f"Hardware-Aware Device — {n_trials} Monte Carlo Trials",
                fontweight="bold",
            )
            ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
            ax.axvline(0, color="black", linewidth=0.5, alpha=0.5)
            ax.grid(True, linestyle="--", alpha=0.4)
            self.figure.tight_layout()
            self.canvas.draw()

            self.progress_bar.setValue(100)
            self.status_label.setText(f"✓ Completed {n_trials} trials with non-idealities.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Simulation failed:\n{e}")
            self.status_label.setText(f"✗ Error: {e}")

        finally:
            self.run_btn.setEnabled(True)
