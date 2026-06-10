"""Parameter fitting engine for memristor compact models.

This module provides the ModelFitter class to optimize compact model parameters
against COMSOL-exported simulation datasets.
"""

from typing import Any

import numpy as np
from scipy.optimize import differential_evolution, minimize

from compact_models.base import MemristorCompactModel
from comsol.dataset import COMSOLDataset
from fitting.loss import hybrid_loss
from utils.config_loader import Config
from utils.logger import get_logger

logger = get_logger()


class ModelFitter:
    """Fitter class to optimize memristor model parameters."""

    def __init__(self, config: Config | None = None) -> None:
        """Initializes the ModelFitter.

        Args:
            config: Config object representing parameters. If None,
                    loads the default configuration.
        """
        self.config = config or Config({})
        self.loss_gamma = self.config.get("compact_model.fitting.loss_gamma", 0.3)
        self.max_iter = self.config.get("compact_model.fitting.max_iter", 500)
        self.pop_size = self.config.get("compact_model.fitting.pop_size", 30)
        self.tolerance = self.config.get("compact_model.fitting.tolerance", 1.0e-4)

    def fit(
        self,
        dataset: COMSOLDataset,
        model_class: type[MemristorCompactModel],
        fit_param_keys: list[str],
        bounds_dict: dict[str, tuple[float, float]],
        fixed_params: dict[str, Any],
        w_init: float,
        solver_type: str = "rk4",
    ) -> dict[str, float]:
        """Fits compact model parameters to the target dataset current.

        Args:
            dataset: COMSOLDataset containing target 'voltage', 'current', and 'time'.
            model_class: Compact model class to fit (e.g., VTEAMModel).
            fit_param_keys: List of parameter names to optimize.
            bounds_dict: Dictionary mapping parameter names to (min, max) bounds.
            fixed_params: Dictionary of parameters held constant during optimization.
            w_init: Initial state variable value.
            solver_type: ODE integration solver type ('euler' or 'rk4').

        Returns:
            Dict[str, float]: Optimized parameters dictionary.
        """
        df = dataset.data
        v_target = df["voltage"].to_numpy()
        i_target = df["current"].to_numpy()

        # Check if time column exists; fallback to uniform time step
        if "time" in df.columns:
            t_target = df["time"].to_numpy()
        else:
            t_target = np.linspace(0.0, 1.0, len(v_target))

        # Setup optimization bounds array
        bounds = [bounds_dict[key] for key in fit_param_keys]

        logger.info(
            f"Starting parameter optimization for {model_class.__name__}. "
            f"Fitting {len(fit_param_keys)} parameters."
        )

        def objective_function(x: np.ndarray) -> float:
            """Objective function evaluating hybrid loss between simulation and target.

            Args:
                x: Array of parameter values being optimized.

            Returns:
                float: Objective loss value.
            """
            # Create parameter mapping
            model_params = fixed_params.copy()
            for key, val in zip(fit_param_keys, x, strict=False):
                model_params[key] = float(val)

            # Instantiate model and solve
            try:
                model = model_class(model_params)
                _, i_sim = model.solve_sweep(
                    v_target, t_target, w_init=w_init, solver_type=solver_type
                )
                # Compute loss
                loss = hybrid_loss(i_sim, i_target, gamma=self.loss_gamma)
                if np.isnan(loss) or np.isinf(loss):
                    return 1.0e10
                return loss
            except Exception:
                return 1.0e10

        # Stage 1: Global optimization using Differential Evolution
        logger.info("Stage 1: Running Differential Evolution global search...")
        res_de = differential_evolution(
            objective_function,
            bounds=bounds,
            maxiter=self.max_iter,
            popsize=self.pop_size,
            tol=self.tolerance,
            disp=False,
        )

        best_x = res_de.x
        best_loss = res_de.fun
        logger.info(f"Differential Evolution finished. Best Loss: {best_loss:.6f}")

        # Stage 2: Local refinement using Nelder-Mead minimize
        logger.info("Stage 2: Refining parameters using Nelder-Mead local search...")
        res_nm = minimize(
            objective_function,
            x0=best_x,
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxiter": 200, "xatol": self.tolerance, "disp": False},
        )

        refined_x = res_nm.x
        refined_loss = res_nm.fun
        logger.info(f"Nelder-Mead refinement finished. Refined Loss: {refined_loss:.6f}")

        # Construct final fitted parameters dict
        fitted_params = fixed_params.copy()
        for key, val in zip(fit_param_keys, refined_x, strict=False):
            fitted_params[key] = float(val)

        return fitted_params
