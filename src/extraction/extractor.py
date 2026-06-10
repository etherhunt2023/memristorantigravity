"""High-level interface for parameter extraction from COMSOL datasets."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from comsol.dataset import COMSOLDataset
from extraction import metrics
from extraction.exceptions import ExtractionError
from utils.config_loader import Config
from utils.logger import get_logger

logger = get_logger()


@dataclass
class ExtractedParameters:
    """Dataclass holding extracted memristive device parameters."""

    r_lrs: float
    r_hrs: float
    v_set: float
    v_reset: float
    non_linearity: float
    switching_time: float | None = None
    r_ratio: float = field(init=False)
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Calculates secondary parameters like R_ratio."""
        if self.r_lrs > 0:
            self.r_ratio = self.r_hrs / self.r_lrs
        else:
            self.r_ratio = float("inf")

    def to_dict(self) -> dict[str, Any]:
        """Converts the extracted parameters into a dictionary.

        Returns:
            Dict[str, Any]: Extracted parameters mapping.
        """
        return {
            "r_lrs": self.r_lrs,
            "r_hrs": self.r_hrs,
            "r_ratio": self.r_ratio,
            "v_set": self.v_set,
            "v_reset": self.v_reset,
            "non_linearity": self.non_linearity,
            "switching_time": self.switching_time,
        }

    def __repr__(self) -> str:
        """Pretty print of extracted parameters.

        Returns:
            str: Description of extracted parameters.
        """
        return (
            f"ExtractedParameters(\n"
            f"  R_LRS:         {self.r_lrs:.2f} Ohms\n"
            f"  R_HRS:         {self.r_hrs:.2f} Ohms\n"
            f"  R_ratio (H/L): {self.r_ratio:.2f}\n"
            f"  V_set:         {self.v_set:.2f} V\n"
            f"  V_reset:       {self.v_reset:.2f} V\n"
            f"  Non-linearity: {self.non_linearity:.2f}\n"
            f"  Switching Time: "
            f"{f'{self.switching_time:.2e} s' if self.switching_time is not None else 'N/A'}\n"
            f")"
        )


class ElectricalParameterExtractor:
    """High-level memristor electrical parameter extractor."""

    def __init__(self, config: Config | None = None) -> None:
        """Initializes the ElectricalParameterExtractor.

        Args:
            config: Config object representing parameters. If None,
                    loads the default configuration.
        """
        self.config = config or Config({})

        # Extract config values with safe fallbacks
        self.lrs_read_voltage = self.config.get("extraction.lrs_read_voltage", 0.1)
        self.hrs_read_voltage = self.config.get("extraction.hrs_read_voltage", 0.1)
        self.non_linearity_voltage = self.config.get("extraction.non_linearity_voltage", 1.0)
        self.set_threshold_curr = self.config.get("extraction.set_threshold_current", 1e-5)
        self.reset_threshold_curr = self.config.get("extraction.reset_threshold_current", 1e-5)
        self.method = self.config.get("extraction.method", "derivative")

    def extract(self, dataset: COMSOLDataset) -> ExtractedParameters:
        """Extracts parameters from the given COMSOLDataset.

        Args:
            dataset: The parsed COMSOLDataset.

        Returns:
            ExtractedParameters: Struct containing extracted physical quantities.

        Raises:
            ExtractionError: If a critical parameter cannot be resolved.
        """
        logger.info("Starting parameter extraction pipeline...")
        df = dataset.data

        # Validate that required columns exist
        if "voltage" not in df.columns or "current" not in df.columns:
            msg = "Dataset must contain standardized 'voltage' and 'current' columns."
            raise ExtractionError(msg)

        # 1. Extract resistances
        # LRS and HRS are read at their respective voltages (often the same)
        try:
            r_lrs, _ = metrics.extract_lrs_hrs(df, self.lrs_read_voltage)
            _, r_hrs = metrics.extract_lrs_hrs(df, self.hrs_read_voltage)
        except Exception as e:
            msg = f"Failed to extract LRS/HRS resistances: {e}"
            logger.error(msg)
            raise ExtractionError(msg) from e

        # 2. Extract switching voltages
        try:
            v_set, v_reset = metrics.extract_switching_voltages(
                df,
                method=self.method,
                set_threshold_curr=self.set_threshold_curr,
                reset_threshold_curr=self.reset_threshold_curr,
            )
        except Exception as e:
            msg = f"Failed to extract switching voltages: {e}"
            logger.error(msg)
            raise ExtractionError(msg) from e

        # 3. Extract non-linearity
        try:
            non_linearity = metrics.extract_non_linearity(df, self.non_linearity_voltage)
        except Exception as e:
            logger.warning(f"Could not calculate non-linearity: {e}. Defaulting to 1.0.")
            non_linearity = 1.0

        # 4. Extract switching time (optional, check if 'time' and transient details exist)
        switching_time = None
        if "time" in df.columns and ("state_var" in df.columns or "current" in df.columns):
            try:
                # Only attempt if there's a significant change in voltage indicating a step/pulse
                v = df["voltage"].to_numpy()
                v_range = np.max(v) - np.min(v)
                if v_range > 1e-3:
                    switching_time = metrics.extract_switching_time(df)
                    logger.info(f"Extracted transient switching time: {switching_time:.2e} s")
            except Exception as e:
                logger.warning(f"Could not extract switching time from transient data: {e}")

        logger.info("Parameter extraction pipeline completed successfully.")
        return ExtractedParameters(
            r_lrs=r_lrs,
            r_hrs=r_hrs,
            v_set=v_set,
            v_reset=v_reset,
            non_linearity=non_linearity,
            switching_time=switching_time,
            raw_metadata=dataset.metadata,
        )
