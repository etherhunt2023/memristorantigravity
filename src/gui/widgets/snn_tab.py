"""Spiking Neural Network simulation tab widget.

Allows the user to configure LIF neuron populations, generate Poisson spike
trains, run the SNN simulation, and visualize input spikes, membrane potentials,
and output spike rasters in a three-panel publication-quality layout.
"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
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

from gui.workers import SNNWorker


class SNNTab(QWidget):
    """Tab widget for SNN simulation and spike visualization."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initializes the SNNTab.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self.worker = None
        self.thread = None
        self._init_ui()

    def _init_ui(self) -> None:
        """Builds the SNN tab UI."""
        layout = QVBoxLayout(self)

        top_layout = QHBoxLayout()

        # Network configuration
        net_group = QGroupBox("Network Configuration")
        net_layout = QFormLayout(net_group)

        self.n_inputs_spin = QSpinBox()
        self.n_inputs_spin.setRange(2, 128)
        self.n_inputs_spin.setValue(8)
        net_layout.addRow("Input Neurons:", self.n_inputs_spin)

        self.n_outputs_spin = QSpinBox()
        self.n_outputs_spin.setRange(1, 64)
        self.n_outputs_spin.setValue(4)
        net_layout.addRow("Output LIF Neurons:", self.n_outputs_spin)

        self.n_timesteps_spin = QSpinBox()
        self.n_timesteps_spin.setRange(50, 5000)
        self.n_timesteps_spin.setValue(200)
        net_layout.addRow("Time Steps:", self.n_timesteps_spin)

        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.01, 10.0)
        self.dt_spin.setValue(0.5)
        self.dt_spin.setSuffix(" ms")
        net_layout.addRow("Δt:", self.dt_spin)

        top_layout.addWidget(net_group)

        # Neuron parameters
        neuron_group = QGroupBox("LIF Neuron Parameters")
        neuron_layout = QFormLayout(neuron_group)

        self.v_thresh_spin = QDoubleSpinBox()
        self.v_thresh_spin.setRange(0.1, 5.0)
        self.v_thresh_spin.setValue(0.8)
        self.v_thresh_spin.setDecimals(2)
        neuron_layout.addRow("V_threshold:", self.v_thresh_spin)

        self.tau_spin = QDoubleSpinBox()
        self.tau_spin.setRange(0.5, 0.999)
        self.tau_spin.setValue(0.95)
        self.tau_spin.setDecimals(3)
        neuron_layout.addRow("τ_decay:", self.tau_spin)

        self.rate_spin = QDoubleSpinBox()
        self.rate_spin.setRange(0.01, 0.9)
        self.rate_spin.setValue(0.15)
        self.rate_spin.setDecimals(2)
        neuron_layout.addRow("Input Spike Rate:", self.rate_spin)

        self.amplitude_spin = QDoubleSpinBox()
        self.amplitude_spin.setRange(0.001, 1.0)
        self.amplitude_spin.setValue(0.08)
        self.amplitude_spin.setDecimals(3)
        neuron_layout.addRow("Spike Amplitude:", self.amplitude_spin)

        top_layout.addWidget(neuron_group)

        layout.addLayout(top_layout)

        # Run button
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("▶  Run SNN Simulation")
        self.run_btn.setObjectName("runSNNButton")
        self.run_btn.setMinimumHeight(36)
        self.run_btn.clicked.connect(self._run_simulation)
        btn_layout.addWidget(self.run_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        btn_layout.addWidget(self.progress_bar)
        layout.addLayout(btn_layout)

        # Three-panel plot
        plot_group = QGroupBox("SNN Activity Visualization")
        plot_layout = QVBoxLayout(plot_group)
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        layout.addWidget(plot_group, 1)

        # Status
        self.status_label = QLabel("Ready. Configure network and run SNN simulation.")
        layout.addWidget(self.status_label)

    def _run_simulation(self) -> None:
        """Starts the SNN simulation in a background thread."""
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.information(self, "Busy", "Simulation is already running.")
            return

        self.worker = SNNWorker(
            n_inputs=self.n_inputs_spin.value(),
            n_outputs=self.n_outputs_spin.value(),
            n_timesteps=self.n_timesteps_spin.value(),
            dt_ms=self.dt_spin.value(),
            input_rate=self.rate_spin.value(),
            v_thresh=self.v_thresh_spin.value(),
            v_rest=0.0,
            tau_decay=self.tau_spin.value(),
            spike_amplitude=self.amplitude_spin.value(),
        )

        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.signals.progress.connect(self._on_progress)
        self.worker.signals.result.connect(self._on_result)
        self.worker.signals.error.connect(self._on_error)
        self.worker.signals.finished.connect(self.thread.quit)

        self.run_btn.setEnabled(False)
        self.status_label.setText("Running SNN simulation...")
        self.thread.start()

    def _on_progress(self, pct: int, msg: str) -> None:
        """Handles progress.

        Args:
            pct: Percentage.
            msg: Message.
        """
        self.progress_bar.setValue(pct)
        self.status_label.setText(msg)

    def _on_result(self, result: dict) -> None:
        """Handles SNN simulation results and plots the three-panel visualization.

        Args:
            result: Dictionary with time, spikes, membrane voltages.
        """
        self.run_btn.setEnabled(True)
        self.progress_bar.setValue(100)

        time_ms = result["time_ms"]
        input_spikes = result["input_spikes"]
        mem_voltages = result["mem_voltages"]
        output_spikes = result["output_spikes"]
        v_thresh = result["v_thresh"]

        self.figure.clear()

        # Panel 1: Input spike raster
        ax1 = self.figure.add_subplot(311)
        in_features = input_spikes.shape[1]
        for idx in range(in_features):
            spike_times = time_ms[input_spikes[:, idx]]
            ax1.scatter(
                spike_times,
                np.full_like(spike_times, idx),
                marker="|",
                color="black",
                s=20,
                linewidths=0.7,
            )
        ax1.set_ylabel("Input Index", fontweight="bold")
        ax1.set_title("Input Spike Raster", fontweight="bold")
        ax1.set_ylim(-0.5, in_features - 0.5)
        ax1.grid(True, linestyle="--", alpha=0.3)

        # Panel 2: Membrane potentials
        ax2 = self.figure.add_subplot(312, sharex=ax1)
        out_features = mem_voltages.shape[1]
        for idx in range(out_features):
            ax2.plot(time_ms, mem_voltages[:, idx], label=f"N{idx}", linewidth=1.0)
        ax2.axhline(v_thresh, color="red", linestyle=":", linewidth=0.8, label="Thresh")
        ax2.set_ylabel("Membrane V", fontweight="bold")
        ax2.set_title("Output Neuron Membrane Potentials", fontweight="bold")
        ax2.legend(loc="upper right", fontsize=7, framealpha=0.8)
        ax2.grid(True, linestyle="--", alpha=0.3)

        # Panel 3: Output spike raster
        ax3 = self.figure.add_subplot(313, sharex=ax1)
        for idx in range(out_features):
            spike_times = time_ms[output_spikes[:, idx]]
            ax3.scatter(
                spike_times,
                np.full_like(spike_times, idx),
                marker="|",
                color="blue",
                s=30,
                linewidths=1.0,
            )
        ax3.set_xlabel("Time (ms)", fontweight="bold")
        ax3.set_ylabel("Neuron Index", fontweight="bold")
        ax3.set_title("Output Spikes", fontweight="bold")
        ax3.set_ylim(-0.5, out_features - 0.5)
        ax3.grid(True, linestyle="--", alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()

        total_out = int(output_spikes.sum())
        self.status_label.setText(
            f"✓ SNN simulation complete — {total_out} output spikes generated."
        )

    def _on_error(self, msg: str) -> None:
        """Handles errors.

        Args:
            msg: Error message.
        """
        self.run_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        QMessageBox.critical(self, "SNN Error", f"Simulation failed:\n{msg}")
        self.status_label.setText(f"✗ Error: {msg}")
