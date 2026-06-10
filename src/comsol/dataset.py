"""Dataset representation for COMSOL export data.

This module defines the COMSOLDataset class, which encapsulates the tabular data
and experimental/simulation metadata exported from COMSOL Multiphysics.
"""

from typing import Any

import numpy as np
import pandas as pd


class COMSOLDataset:
    """Encapsulates COMSOL simulation data and associated metadata."""

    def __init__(self, data: pd.DataFrame, metadata: dict[str, Any]) -> None:
        """Initializes the COMSOLDataset.

        Args:
            data: Tabular simulation data as a Pandas DataFrame.
            metadata: Extracted comments, parameter values, and sweep variables.
        """
        self._data = data
        self._metadata = metadata

    @property
    def data(self) -> pd.DataFrame:
        """Returns the full Pandas DataFrame.

        Returns:
            pd.DataFrame: The numerical data.
        """
        return self._data

    @property
    def metadata(self) -> dict[str, Any]:
        """Returns the metadata dictionary containing parsed COMSOL comments.

        Returns:
            Dict[str, Any]: The metadata.
        """
        return self._metadata

    @property
    def columns(self) -> list[str]:
        """Returns the column names of the dataset.

        Returns:
            List[str]: Column names list.
        """
        return list(self._data.columns)

    @property
    def shape(self) -> tuple[int, int]:
        """Returns the dimensions of the tabular data.

        Returns:
            tuple[int, int]: (number of rows, number of columns).
        """
        return self._data.shape

    def get_sweep_parameters(self) -> list[str]:
        """Identifies column names or metadata properties that represent swept parameters.

        Typically, COMSOL parameter sweeps create columns with parameter values in the header.

        Returns:
            List[str]: List of parameter names.
        """
        return self._metadata.get("sweep_parameters", [])

    def filter_by_sweep(self, sweep_values: dict[str, Any]) -> pd.DataFrame:
        """Filters the dataset to retrieve a specific sweep run.

        Args:
            sweep_values: A dictionary mapping parameter names to their specific values.
                          For example: {'width': 1e-6, 'thickness': 10e-9}

        Returns:
            pd.DataFrame: A filtered subset of the data.

        Raises:
            ValueError: If a parameter is not found or no rows match the criteria.
        """
        filtered_df = self._data.copy()
        for param, value in sweep_values.items():
            if param in filtered_df.columns:
                # Handle potential floating point comparison issues
                filtered_df = filtered_df[np.isclose(filtered_df[param], float(value))]
            else:
                msg = f"Sweep parameter '{param}' not found in dataset columns."
                raise ValueError(msg)

        if filtered_df.empty:
            msg = f"No data matches the sweep parameters: {sweep_values}"
            raise ValueError(msg)

        return filtered_df

    def to_numpy(self, columns: list[str] | None = None) -> np.ndarray:
        """Converts the specified columns of the dataset into a NumPy array.

        Args:
            columns: List of columns to extract. If None, extracts all columns.

        Returns:
            np.ndarray: The selected data as a 2D numpy array.
        """
        if columns is None:
            return self._data.to_numpy()
        return self._data[columns].to_numpy()

    def __repr__(self) -> str:
        """Returns a string representation of the COMSOLDataset.

        Returns:
            str: Description of the dataset.
        """
        items = (f"{k}={v}" for k, v in self._metadata.items() if k != "comments")
        metadata_summary = ", ".join(items)
        return (
            f"COMSOLDataset(shape={self.shape}, "
            f"columns={self.columns}, "
            f"metadata={{{metadata_summary}}})"
        )
