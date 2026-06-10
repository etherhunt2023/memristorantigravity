"""Crossbar array simulator with Modified Nodal Analysis (MNA).

This module simulates a 2D memristor crossbar array, accounting for wire
parasitic resistances (IR drop), driver source resistance, load resistance,
and sneak paths.
"""

from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from compact_models.base import MemristorCompactModel
from device.memristor import MemristorDevice
from utils.config_loader import Config, load_config
from utils.logger import get_logger

logger = get_logger("crossbar")


class CrossbarArray:
    """Simulates a 2D crossbar array of memristor devices with parasitics."""

    def __init__(
        self,
        rows: int,
        cols: int,
        model_class: type[MemristorCompactModel],
        base_params: dict[str, Any],
        device_config: dict[str, Any] | Config | None = None,
        crossbar_config: dict[str, Any] | Config | None = None,
    ) -> None:
        """Initializes the CrossbarArray.

        Args:
            rows: Number of row lines (wordlines).
            cols: Number of column lines (bitlines).
            model_class: The compact model class (e.g., VTEAMModel).
            base_params: Nominal parameters for the compact model.
            device_config: Configurations for memristor devices.
            crossbar_config: Configurations for crossbar parameters.
        """
        self.rows = rows
        self.cols = cols

        # Load crossbar configs
        if crossbar_config is None:
            try:
                full_config = load_config()
                self.config = Config(full_config.raw.get("crossbar", {}))
            except Exception:
                self.config = Config({})
        elif isinstance(crossbar_config, Config):
            self.config = crossbar_config
        else:
            self.config = Config(crossbar_config)

        # Parse crossbar parasities
        # Row segment resistance
        self.r_line_r = max(float(self.config.get("line_resistance", 1.5)), 1.0e-5)
        # Column segment resistance
        self.r_line_c = max(float(self.config.get("line_resistance", 1.5)), 1.0e-5)

        self.r_source = float(self.config.get("source_resistance", 100.0))
        self.r_load = float(self.config.get("load_resistance", 100.0))

        self.g_line_r = 1.0 / self.r_line_r
        self.g_line_c = 1.0 / self.r_line_c

        # Instantiate devices array
        # Note: D2D is sampled individually for each cell
        self.devices = np.empty((rows, cols), dtype=object)
        for i in range(rows):
            for j in range(cols):
                self.devices[i, j] = MemristorDevice(
                    model_class,
                    base_params,
                    device_config=device_config,
                    w_init=base_params.get("w_off", 1.0),
                )

        logger.info(f"Initialized {rows}x{cols} CrossbarArray with parasitics.")

    def reset(self) -> None:
        """Resets all devices in the crossbar array."""
        for i in range(self.rows):
            for j in range(self.cols):
                self.devices[i, j].reset()

    def _construct_conductance_matrix(
        self, g_cell: np.ndarray, g_s: np.ndarray, g_l: np.ndarray
    ) -> sp.csc_matrix:
        """Constructs the sparse MNA conductance matrix.

        Args:
            g_cell: 2D array of memristor conductances (shape: rows x cols).
            g_s: 1D array of row source conductances (shape: rows).
            g_l: 1D array of column load conductances (shape: cols).

        Returns:
            sp.csc_matrix: Sparse conductance matrix of size 2MN x 2MN.
        """
        m = self.rows
        n = self.cols

        row_list = []
        col_list = []
        val_list = []

        # 1. Row line wire segments (horizontal resistors)
        # Resistors connect node (i, j) and (i, j+1)
        row_idx = np.arange(m)[:, None]
        col_idx = np.arange(n - 1)[None, :]
        idx1 = (row_idx * n + col_idx).flatten()
        idx2 = (row_idx * n + col_idx + 1).flatten()

        g_r = self.g_line_r
        row_list.extend([idx1, idx2, idx1, idx2])
        col_list.extend([idx1, idx2, idx2, idx1])
        val_list.extend(
            [
                np.full(len(idx1), g_r),
                np.full(len(idx1), g_r),
                np.full(len(idx1), -g_r),
                np.full(len(idx1), -g_r),
            ]
        )

        # 2. Column line wire segments (vertical resistors)
        # Resistors connect node (i, j) and (i+1, j) on the columns
        row_idx_c = np.arange(m - 1)[:, None]
        col_idx_c = np.arange(n)[None, :]
        idx1_c = (m * n + row_idx_c * n + col_idx_c).flatten()
        idx2_c = (m * n + (row_idx_c + 1) * n + col_idx_c).flatten()

        g_c = self.g_line_c
        row_list.extend([idx1_c, idx2_c, idx1_c, idx2_c])
        col_list.extend([idx1_c, idx2_c, idx2_c, idx1_c])
        val_list.extend(
            [
                np.full(len(idx1_c), g_c),
                np.full(len(idx1_c), g_c),
                np.full(len(idx1_c), -g_c),
                np.full(len(idx1_c), -g_c),
            ]
        )

        # 3. Source conductances on row nodes (j = 0)
        idx_s = np.arange(m) * n
        row_list.append(idx_s)
        col_list.append(idx_s)
        val_list.append(g_s)

        # 4. Load conductances on column nodes (i = M-1)
        idx_l = m * n + (m - 1) * n + np.arange(n)
        row_list.append(idx_l)
        col_list.append(idx_l)
        val_list.append(g_l)

        # 5. Memristors (connect row node (i, j) to column node (i, j))
        idx_r = np.arange(m * n)
        idx_c = m * n + idx_r
        g_m = g_cell.flatten()

        row_list.extend([idx_r, idx_c, idx_r, idx_c])
        col_list.extend([idx_r, idx_c, idx_c, idx_r])
        val_list.extend([g_m, g_m, -g_m, -g_m])

        # Concatenate coordinate formats
        rows_concat = np.concatenate(row_list)
        cols_concat = np.concatenate(col_list)
        vals_concat = np.concatenate(val_list)

        # Create sparse COO matrix and convert to CSC
        a = sp.coo_matrix((vals_concat, (rows_concat, cols_concat)), shape=(2 * m * n, 2 * m * n))
        return a.tocsc()

    def solve_mna(
        self,
        row_voltages: np.ndarray,
        col_voltages: np.ndarray | None = None,
        row_rs: np.ndarray | None = None,
        col_rl: np.ndarray | None = None,
        use_nonlinear: bool = True,
        max_iter: int = 50,
        tol: float = 1.0e-5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solves the crossbar node voltages using Modified Nodal Analysis (MNA).

        Args:
            row_voltages: Voltages applied to row inputs (shape: M).
            col_voltages: Voltages applied to column outputs (shape: N). Defaults to 0V.
            row_rs: Source resistances for each row terminal (shape: M).
            col_rl: Load resistances for each column terminal (shape: N).
            use_nonlinear: If True, uses Newton-Raphson iteration.
            max_iter: Maximum Newton-Raphson iterations.
            tol: Tolerance for convergence.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (v_row, v_col) arrays of node voltages
                                           of shape M x N.
        """
        m = self.rows
        n = self.cols

        # Default values
        col_voltages = col_voltages if col_voltages is not None else np.zeros(n)
        row_rs_arr = row_rs if row_rs is not None else np.full(m, self.r_source)
        col_rl_arr = col_rl if col_rl is not None else np.full(n, self.r_load)

        # Conductances
        g_s = 1.0 / np.clip(row_rs_arr, 1.0e-20, None)
        g_l = 1.0 / np.clip(col_rl_arr, 1.0e-20, None)

        # Compute initial guess using linear chord conductances at current state
        # We estimate linear conductances G = 1/R(w)
        g_cell_lin = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                dev = self.devices[i, j]
                r_on = dev.params.get("r_on", dev.params.get("R_on", 1.0e3))
                r_off = dev.params.get("r_off", dev.params.get("R_off", 1.0e5))
                w_norm = (dev.w - dev.model.w_min) / (dev.model.w_max - dev.model.w_min + 1e-20)
                r_eff = r_on + (r_off - r_on) * w_norm
                g_cell_lin[i, j] = 1.0 / max(r_eff, 1.0)

        # Solve linear system
        a_lin = self._construct_conductance_matrix(g_cell_lin, g_s, g_l)

        # Build RHS vector b
        b = np.zeros(2 * m * n)
        idx_s = np.arange(m) * n
        b[idx_s] = g_s * row_voltages
        idx_l = m * n + (m - 1) * n + np.arange(n)
        b[idx_l] = g_l * col_voltages

        # Linear solve
        v_node = spsolve(a_lin, b)

        if not use_nonlinear:
            # Unpack linear solution
            v_row = v_node[: m * n].reshape((m, n))
            v_col = v_node[m * n :].reshape((m, n))
            return v_row, v_col

        # Newton-Raphson iteration loop
        delta = 1.0e-5
        for _it in range(max_iter):
            v_row = v_node[: m * n].reshape((m, n))
            v_col = v_node[m * n :].reshape((m, n))
            v_drop = v_row - v_col

            # Compute current and differential conductance for each cell
            i_cell = np.zeros((m, n))
            g_diff = np.zeros((m, n))
            for i in range(m):
                for j in range(n):
                    dev = self.devices[i, j]
                    v_d = v_drop[i, j]
                    i_cell[i, j] = dev.current_noise_free(v_d)

                    # Numerical derivative for Jacobian g_diff
                    i_plus = dev.current_noise_free(v_d + delta)
                    i_minus = dev.current_noise_free(v_d - delta)
                    g_diff[i, j] = max((i_plus - i_minus) / (2.0 * delta), 1.0e-12)

            # Compute residual vector f
            f_row = np.zeros((m, n))
            f_col = np.zeros((m, n))

            # Row nodes residuals
            f_row[:, 0] += (v_row[:, 0] - row_voltages) * g_s
            f_row[:, 1:] += (v_row[:, 1:] - v_row[:, :-1]) * self.g_line_r
            f_row[:, :-1] += (v_row[:, :-1] - v_row[:, 1:]) * self.g_line_r
            f_row += i_cell

            # Column nodes residuals
            f_col[1:, :] += (v_col[1:, :] - v_col[:-1, :]) * self.g_line_c
            f_col[:-1, :] += (v_col[:-1, :] - v_col[1:, :]) * self.g_line_c
            f_col[-1, :] += (v_col[-1, :] - col_voltages) * g_l
            f_col -= i_cell

            f = np.concatenate([f_row.flatten(), f_col.flatten()])

            # Convergence check
            f_norm = np.linalg.norm(f)
            if f_norm < tol:
                break

            # Construct Jacobian matrix j_mat
            j_mat = self._construct_conductance_matrix(g_diff, g_s, g_l)

            # Solve update step: j_mat * dv = -f
            try:
                dv = spsolve(j_mat, -f)
            except Exception as e:
                logger.warning(f"Jacobian matrix solve failed: {e}. Aborting Newton-Raphson.")
                break

            v_node += dv

            # Second convergence check on update step size
            if np.linalg.norm(dv) < tol:
                break
        else:
            logger.warning("Newton-Raphson solver did not converge within maximum iterations.")

        v_row = v_node[: m * n].reshape((m, n))
        v_col = v_node[m * n :].reshape((m, n))
        return v_row, v_col

    def step(
        self, row_voltages: np.ndarray, col_voltages: np.ndarray | None = None, dt: float = 1.0e-3
    ) -> np.ndarray:
        """Steps all device states forward by dt based on solved MNA node voltages.

        Args:
            row_voltages: Voltages applied to row terminals (shape: M).
            col_voltages: Voltages applied to column terminals (shape: N). Defaults to 0V.
            dt: Time step (s).

        Returns:
            np.ndarray: Vector of output column terminal currents (shape: N).
        """
        # Solve voltages
        v_row, v_col = self.solve_mna(row_voltages, col_voltages)
        v_drop = v_row - v_col

        # Step each device
        for i in range(self.rows):
            for j in range(self.cols):
                self.devices[i, j].step(v_drop[i, j], dt)

        # Compute output column currents
        col_v = col_voltages if col_voltages is not None else np.zeros(self.cols)
        # Output current leaving bitlines to columns loads
        i_out = (v_col[-1, :] - col_v) / self.r_load
        return i_out

    def solve_sweep(
        self,
        row_voltages_sweep: np.ndarray,
        col_voltages_sweep: np.ndarray | None = None,
        time_points: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solves the transient crossbar response for a voltage sweep over time.

        Args:
            row_voltages_sweep: 2D array of applied row voltages (shape: timesteps x M).
            col_voltages_sweep: 2D array of applied col voltages (shape: timesteps x N).
            time_points: Array of time points corresponding to the sweep.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (w_history, col_currents_history)
                                           w_history shape: timesteps x M x N.
                                           col_currents_history shape: timesteps x N.
        """
        num_steps = len(row_voltages_sweep)
        m, n = self.rows, self.cols

        if time_points is None:
            time_points = np.linspace(0.0, float(num_steps) * 1.0e-3, num_steps)

        col_voltages_sweep = (
            col_voltages_sweep if col_voltages_sweep is not None else np.zeros((num_steps, n))
        )

        w_hist = np.zeros((num_steps, m, n))
        i_hist = np.zeros((num_steps, n))

        # Record initial states
        w_hist[0] = np.array([[dev.w for dev in row] for row in self.devices])
        # Solve initial currents
        v_row, v_col = self.solve_mna(row_voltages_sweep[0], col_voltages_sweep[0])
        i_hist[0] = (v_col[-1, :] - col_voltages_sweep[0]) / self.r_load

        # Step through time
        for k in range(num_steps - 1):
            dt = time_points[k + 1] - time_points[k]
            v_row_next = row_voltages_sweep[k + 1]
            v_col_next = col_voltages_sweep[k + 1]

            # Step array
            i_hist[k + 1] = self.step(v_row_next, v_col_next, dt)
            w_hist[k + 1] = np.array([[dev.w for dev in row] for row in self.devices])

        return w_hist, i_hist

    def analyze_sneak_paths(
        self, target_row: int, target_col: int, read_voltage: float, scheme: str = "floating"
    ) -> dict[str, float]:
        """Performs a sneak path analysis for reading a specific target cell.

        Args:
            target_row: Target cell row index.
            target_col: Target cell column index.
            read_voltage: Voltage applied to read the target cell.
            scheme: Sneak-path biasing scheme: "floating", "grounded", or "half_bias".

        Returns:
            Dict[str, float]: Dictionary containing:
                              - "target_current": current through target memristor (A)
                              - "total_column_current": current leaving the target column load (A)
                              - "sneak_current": difference between total and target current (A)
                              - "sneak_efficiency": ratio of target current to total column current
        """
        m = self.rows
        n = self.cols

        row_v = np.zeros(m)
        col_v = np.zeros(n)
        row_rs = np.full(m, self.r_source)
        col_rl = np.full(n, self.r_load)

        # Apply voltages according to scheme
        if scheme.lower() == "floating":
            row_v[target_row] = read_voltage
            # Set all other terminals to extremely high resistance (floating)
            row_rs[np.arange(m) != target_row] = 1.0e12
            col_rl[np.arange(n) != target_col] = 1.0e12

        elif scheme.lower() == "grounded":
            row_v[target_row] = read_voltage
            # All other terminals remain grounded (0V) with nominal resistances

        elif scheme.lower() == "half_bias":
            # 1/2 Bias scheme: target row = V, target col = 0, other rows/cols = V/2
            row_v[target_row] = read_voltage
            row_v[np.arange(m) != target_row] = read_voltage / 2.0
            col_v[np.arange(n) != target_col] = read_voltage / 2.0

        else:
            raise ValueError(f"Unknown sneak path bias scheme: {scheme}")

        # Solve network
        v_row, v_col = self.solve_mna(row_v, col_v, row_rs, col_rl)
        v_drop = v_row - v_col

        # Target cell current
        i_target = self.devices[target_row, target_col].current_noise_free(
            v_drop[target_row, target_col]
        )

        # Total current leaving target column terminal load
        v_terminal = v_col[-1, target_col]
        i_total = (v_terminal - col_v[target_col]) / col_rl[target_col]

        sneak_current = i_total - i_target
        efficiency = i_target / max(abs(i_total), 1.0e-20)

        return {
            "target_current": float(i_target),
            "total_column_current": float(i_total),
            "sneak_current": float(sneak_current),
            "sneak_efficiency": float(efficiency),
        }
