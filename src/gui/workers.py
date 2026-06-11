"""QThread-based workers for long-running simulation and fitting tasks.

Each worker emits progress and result signals so that the GUI remains responsive
while computationally intensive operations run in background threads.
"""

from typing import Any

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot


class WorkerSignals(QObject):
    """Defines signals available from a running worker thread."""

    started = Signal()
    progress = Signal(int, str)  # (percentage 0-100, description)
    result = Signal(object)  # Emitted with the final result payload
    error = Signal(str)  # Emitted with error message string
    finished = Signal()


class SimulationWorker(QObject):
    """Worker that runs a compact model sweep simulation in a background thread."""

    signals = WorkerSignals()

    def __init__(
        self,
        model_class: type,
        params: dict[str, Any],
        voltage_sweep: np.ndarray,
        time_points: np.ndarray,
        w_init: float,
        solver_type: str = "rk4",
    ) -> None:
        """Initializes the SimulationWorker.

        Args:
            model_class: Compact model class (e.g. VTEAMModel).
            params: Model parameter dictionary.
            voltage_sweep: Array of applied voltages.
            time_points: Array of time points.
            w_init: Initial state variable value.
            solver_type: ODE solver type.
        """
        super().__init__()
        self.model_class = model_class
        self.params = params
        self.voltage_sweep = voltage_sweep
        self.time_points = time_points
        self.w_init = w_init
        self.solver_type = solver_type

    @Slot()
    def run(self) -> None:
        """Executes the compact model simulation."""
        try:
            self.signals.started.emit()
            self.signals.progress.emit(10, "Instantiating compact model...")

            model = self.model_class(self.params)
            self.signals.progress.emit(30, "Running voltage sweep simulation...")

            w_hist, i_hist = model.solve_sweep(
                self.voltage_sweep, self.time_points, self.w_init, self.solver_type
            )

            self.signals.progress.emit(100, "Simulation complete.")
            self.signals.result.emit(
                {
                    "voltages": self.voltage_sweep,
                    "currents": i_hist,
                    "states": w_hist,
                    "time": self.time_points,
                }
            )
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


class DeviceSimulationWorker(QObject):
    """Worker that runs a hardware-aware device sweep in a background thread."""

    signals = WorkerSignals()

    def __init__(
        self,
        model_class: type,
        base_params: dict[str, Any],
        device_config: dict[str, Any],
        voltage_sweep: np.ndarray,
        time_points: np.ndarray,
    ) -> None:
        """Initializes the DeviceSimulationWorker.

        Args:
            model_class: Compact model class.
            base_params: Nominal model parameters.
            device_config: Device non-ideality configuration.
            voltage_sweep: Array of applied voltages.
            time_points: Array of time points.
        """
        super().__init__()
        self.model_class = model_class
        self.base_params = base_params
        self.device_config = device_config
        self.voltage_sweep = voltage_sweep
        self.time_points = time_points

    @Slot()
    def run(self) -> None:
        """Executes the hardware-aware device simulation."""
        from device.memristor import MemristorDevice

        try:
            self.signals.started.emit()
            self.signals.progress.emit(10, "Creating MemristorDevice...")

            dev = MemristorDevice(
                self.model_class,
                self.base_params,
                device_config=self.device_config,
            )

            self.signals.progress.emit(30, "Running transient sweep...")
            w_hist, i_hist = dev.solve_sweep(self.voltage_sweep, self.time_points)

            self.signals.progress.emit(100, "Device simulation complete.")
            self.signals.result.emit(
                {
                    "voltages": self.voltage_sweep,
                    "currents": i_hist,
                    "states": w_hist,
                    "time": self.time_points,
                }
            )
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


class CrossbarWorker(QObject):
    """Worker that runs a crossbar MNA simulation in a background thread."""

    signals = WorkerSignals()

    def __init__(
        self,
        rows: int,
        cols: int,
        model_class: type,
        base_params: dict[str, Any],
        row_voltages: np.ndarray,
        device_config: dict[str, Any] | None = None,
        crossbar_config: dict[str, Any] | None = None,
    ) -> None:
        """Initializes the CrossbarWorker.

        Args:
            rows: Number of row lines.
            cols: Number of column lines.
            model_class: Compact model class.
            base_params: Model parameters.
            row_voltages: Row input voltages.
            device_config: Device configuration.
            crossbar_config: Crossbar configuration.
        """
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.model_class = model_class
        self.base_params = base_params
        self.row_voltages = row_voltages
        self.device_config = device_config
        self.crossbar_config = crossbar_config

    @Slot()
    def run(self) -> None:
        """Executes the crossbar MNA analysis."""
        from crossbar.array import CrossbarArray

        try:
            self.signals.started.emit()
            self.signals.progress.emit(10, "Initializing crossbar array...")

            xbar = CrossbarArray(
                self.rows,
                self.cols,
                self.model_class,
                self.base_params,
                device_config=self.device_config,
                crossbar_config=self.crossbar_config,
            )

            self.signals.progress.emit(40, "Solving MNA equations...")
            v_row, v_col = xbar.solve_mna(self.row_voltages)
            v_drop = v_row - v_col

            self.signals.progress.emit(80, "Computing conductance map...")
            g_map = np.zeros((self.rows, self.cols))
            for i in range(self.rows):
                for j in range(self.cols):
                    dev = xbar.devices[i, j]
                    r_on = dev.params.get("r_on", dev.params.get("R_on", 1e3))
                    r_off = dev.params.get("r_off", dev.params.get("R_off", 1e5))
                    w_norm = (dev.w - dev.model.w_min) / (dev.model.w_max - dev.model.w_min + 1e-20)
                    r_eff = r_on + (r_off - r_on) * w_norm
                    g_map[i, j] = 1.0 / max(r_eff, 1.0)

            self.signals.progress.emit(100, "Crossbar analysis complete.")
            self.signals.result.emit(
                {
                    "v_row": v_row,
                    "v_col": v_col,
                    "v_drop": v_drop,
                    "g_map": g_map,
                    "crossbar": xbar,
                }
            )
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


class FittingWorker(QObject):
    """Worker that runs model parameter fitting in a background thread."""

    signals = WorkerSignals()

    def __init__(
        self,
        model_class: type,
        voltages: np.ndarray,
        currents: np.ndarray,
        time_points: np.ndarray,
        fit_param_keys: list[str],
        bounds_dict: dict[str, tuple[float, float]],
        fixed_params: dict[str, Any],
        w_init: float,
        loss_gamma: float = 0.3,
    ) -> None:
        """Initializes the FittingWorker.

        Args:
            model_class: Compact model class to fit.
            voltages: Target voltage array.
            currents: Target current array.
            time_points: Time points array.
            fit_param_keys: Parameter names to optimize.
            bounds_dict: Parameter bounds.
            fixed_params: Fixed parameters.
            w_init: Initial state variable.
            loss_gamma: Hybrid loss weight.
        """
        super().__init__()
        self.model_class = model_class
        self.voltages = voltages
        self.currents = currents
        self.time_points = time_points
        self.fit_param_keys = fit_param_keys
        self.bounds_dict = bounds_dict
        self.fixed_params = fixed_params
        self.w_init = w_init
        self.loss_gamma = loss_gamma

    @Slot()
    def run(self) -> None:
        """Executes the parameter fitting pipeline."""
        from scipy.optimize import differential_evolution, minimize

        from fitting.loss import hybrid_loss

        try:
            self.signals.started.emit()
            self.signals.progress.emit(5, "Setting up optimization problem...")

            bounds = [self.bounds_dict[k] for k in self.fit_param_keys]

            def objective(x: np.ndarray) -> float:
                """Objective function for optimization."""
                params = self.fixed_params.copy()
                for key, val in zip(self.fit_param_keys, x, strict=False):
                    params[key] = float(val)
                try:
                    model = self.model_class(params)
                    _, i_sim = model.solve_sweep(self.voltages, self.time_points, self.w_init)
                    loss = hybrid_loss(i_sim, self.currents, gamma=self.loss_gamma)
                    if np.isnan(loss) or np.isinf(loss):
                        return 1e10
                    return loss
                except Exception:
                    return 1e10

            self.signals.progress.emit(10, "Running Differential Evolution...")
            res_de = differential_evolution(
                objective, bounds=bounds, maxiter=300, popsize=20, tol=1e-4, disp=False
            )

            self.signals.progress.emit(70, "Refining with Nelder-Mead...")
            res_nm = minimize(
                objective,
                x0=res_de.x,
                method="Nelder-Mead",
                bounds=bounds,
                options={"maxiter": 200, "xatol": 1e-4},
            )

            # Build fitted params
            fitted = self.fixed_params.copy()
            for key, val in zip(self.fit_param_keys, res_nm.x, strict=False):
                fitted[key] = float(val)

            # Simulate fitted model
            model = self.model_class(fitted)
            _, i_fit = model.solve_sweep(self.voltages, self.time_points, self.w_init)

            self.signals.progress.emit(100, "Fitting complete.")
            self.signals.result.emit(
                {
                    "fitted_params": fitted,
                    "fit_currents": i_fit,
                    "loss_de": float(res_de.fun),
                    "loss_refined": float(res_nm.fun),
                }
            )
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


class SNNWorker(QObject):
    """Worker that runs a spiking neural network simulation in a background thread."""

    signals = WorkerSignals()

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        n_timesteps: int,
        dt_ms: float,
        input_rate: float,
        v_thresh: float = 0.8,
        v_rest: float = 0.0,
        tau_decay: float = 0.95,
        spike_amplitude: float = 0.08,
    ) -> None:
        """Initializes the SNNWorker.

        Args:
            n_inputs: Number of input neurons.
            n_outputs: Number of output LIF neurons.
            n_timesteps: Number of simulation time steps.
            dt_ms: Time step in milliseconds.
            input_rate: Poisson firing rate for input spikes.
            v_thresh: LIF threshold voltage.
            v_rest: LIF resting potential.
            tau_decay: Membrane decay factor per step.
            spike_amplitude: Input current amplitude per spike.
        """
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_timesteps = n_timesteps
        self.dt_ms = dt_ms
        self.input_rate = input_rate
        self.v_thresh = v_thresh
        self.v_rest = v_rest
        self.tau_decay = tau_decay
        self.spike_amplitude = spike_amplitude

    @Slot()
    def run(self) -> None:
        """Executes the SNN simulation."""
        try:
            self.signals.started.emit()
            self.signals.progress.emit(10, "Generating input spike trains...")

            time_ms = np.arange(self.n_timesteps) * self.dt_ms
            input_spikes = np.random.rand(self.n_timesteps, self.n_inputs) < self.input_rate

            # Random synaptic weight matrix
            weights = np.random.rand(self.n_inputs, self.n_outputs) * 0.15

            self.signals.progress.emit(30, "Running LIF neuron simulation...")
            mem = np.zeros(self.n_outputs)
            mem_history = np.zeros((self.n_timesteps, self.n_outputs))
            out_spikes = np.zeros((self.n_timesteps, self.n_outputs), dtype=bool)

            for t in range(self.n_timesteps):
                # Compute input current
                i_syn = input_spikes[t].astype(float) @ weights * self.spike_amplitude
                mem = mem * self.tau_decay + i_syn
                spikes = mem >= self.v_thresh
                out_spikes[t] = spikes
                mem_history[t] = mem
                mem[spikes] = self.v_rest

            self.signals.progress.emit(100, "SNN simulation complete.")
            self.signals.result.emit(
                {
                    "time_ms": time_ms,
                    "input_spikes": input_spikes,
                    "mem_voltages": mem_history,
                    "output_spikes": out_spikes,
                    "weights": weights,
                    "v_thresh": self.v_thresh,
                }
            )
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()
