"""COMSOL text file parser.

This module implements the COMSOLParser class, which parses tabular numerical data
and metadata comments exported from COMSOL Multiphysics simulations.
"""

import re
from pathlib import Path
from typing import Any

import pandas as pd

from comsol.dataset import COMSOLDataset
from utils.logger import get_logger

logger = get_logger()


class COMSOLParsingError(Exception):
    """Custom exception raised when parsing a COMSOL file fails."""

    pass


class COMSOLParser:
    """Parser for COMSOL tabular exports and parameter sweeps."""

    def __init__(
        self,
        column_mapping: dict[str, list[str]] | None = None,
        comment_char: str = "%",
        delimiter: str | None = None,
    ) -> None:
        r"""Initializes the COMSOLParser.

        Args:
            column_mapping: A dictionary mapping target standard names
                            (e.g., 'time', 'voltage', 'current') to lists of possible
                            COMSOL header labels.
            comment_char: Character designating comment lines in the export file.
            delimiter: Data delimiter (e.g. ',', '\t', or whitespace).
                       If None, the parser will attempt auto-detection.
        """
        self.column_mapping = column_mapping or {}
        self.comment_char = comment_char
        self.delimiter = delimiter

    def parse(self, file_path: Path) -> COMSOLDataset:
        """Parses a COMSOL text export file.

        Args:
            file_path: Absolute or relative Path to the COMSOL file.

        Returns:
            COMSOLDataset: The parsed dataset containing metadata and DataFrame.

        Raises:
            FileNotFoundError: If the file does not exist.
            COMSOLParsingError: If the file is empty or formatted incorrectly.
        """
        path = Path(file_path)
        if not path.exists():
            msg = f"COMSOL export file not found: {path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        logger.info(f"Parsing COMSOL file: {path}")

        comments: list[str] = []
        header_line: str | None = None
        data_start_line = 0

        # Pass 1: Parse metadata comments and find the header line
        with path.open(encoding="utf-8") as f:
            for idx, line in enumerate(f):
                stripped = line.strip()
                if not stripped:
                    continue

                if stripped.startswith(self.comment_char):
                    comments.append(stripped)
                    # Often the last comment line contains column names.
                    # We check if this line is likely the column header.
                    # COMSOL headers typically contain names separated by tabs/spaces.
                    # We'll save the latest comment line as a candidate header line.
                    header_line = stripped
                    data_start_line = idx + 1
                else:
                    # Non-comment line: data starts here
                    data_start_line = idx
                    break

        if not header_line and not comments:
            msg = "File does not contain COMSOL comments or headers."
            logger.error(msg)
            raise COMSOLParsingError(msg)

        # Parse metadata from comments
        metadata = self._parse_comments(comments)

        # Detect headers from the last comment line or construct them
        headers = self._clean_header(header_line) if header_line else None

        # Pass 2: Load tabular numerical data using Pandas
        try:
            # If delimiter is not specified, auto-detect it
            sep = self.delimiter
            if sep is None:
                sep = r"\s+"  # default to whitespace

            # Read the CSV starting from the data line
            df = pd.read_csv(
                path,
                sep=sep,
                comment=self.comment_char,
                header=None,
                skiprows=data_start_line,
                engine="python",
            )
        except Exception as e:
            msg = f"Failed to load numerical data from {path}: {e}"
            logger.error(msg)
            raise COMSOLParsingError(msg) from e

        if df.empty:
            msg = f"Numerical data section is empty in COMSOL file: {path}"
            logger.error(msg)
            raise COMSOLParsingError(msg)

        # Assign headers or generic columns if not detected
        if headers and len(headers) == len(df.columns):
            df.columns = headers
        else:
            if headers:
                logger.warning(
                    f"Header length mismatch. Detected {len(headers)} columns from comments, "
                    f"but read {len(df.columns)} data columns. Using default numbering."
                )
            df.columns = [f"Column_{i}" for i in range(len(df.columns))]

        # Standardize mapped columns
        standardized_df = self._standardize_columns(df)

        return COMSOLDataset(standardized_df, metadata)

    def _parse_comments(self, comments: list[str]) -> dict[str, Any]:
        """Extracts metadata dictionary from COMSOL comments.

        Matches key-value structures like "Model: memristor.mph" or "Parameters: L=1e-9".

        Args:
            comments: List of comment strings.

        Returns:
            Dict[str, Any]: Parsed metadata dictionary.
        """
        metadata: dict[str, Any] = {"comments": comments}
        sweep_parameters: list[str] = []

        # Regexes for common COMSOL metadata patterns
        kv_pattern = re.compile(r"^%\s*([\w\s\-\(\)]+?)\s*:\s*(.+)$")
        param_pattern = re.compile(r"([\w_]+)\s*=\s*([0-9\.\-eE+]+)(?:\s*\[[\w]+\])?")

        for line in comments:
            match = kv_pattern.match(line)
            if match:
                key = match.group(1).strip().lower().replace(" ", "_")
                value_str = match.group(2).strip()

                # Check if value matches multiple parameter assignments (e.g. L=1e-9, W=2e-8)
                params = param_pattern.findall(value_str)
                if params:
                    param_dict = {}
                    for p_name, p_val in params:
                        try:
                            param_dict[p_name] = float(p_val)
                            sweep_parameters.append(p_name)
                        except ValueError:
                            param_dict[p_name] = p_val
                    metadata[key] = param_dict
                else:
                    # Attempt basic numeric conversion
                    try:
                        if "." in value_str or "e" in value_str.lower():
                            metadata[key] = float(value_str)
                        else:
                            metadata[key] = int(value_str)
                    except ValueError:
                        metadata[key] = value_str

        metadata["sweep_parameters"] = list(set(sweep_parameters))
        return metadata

    def _clean_header(self, header_line: str) -> list[str]:
        """Cleans and extracts column names from the raw comment header line.

        Args:
            header_line: The comment line containing header names (e.g. "% t  V  I").

        Returns:
            List[str]: Cleaned column names.
        """
        cleaned = header_line.lstrip(self.comment_char).strip()

        # Try tab first
        if "\t" in cleaned:
            return [c.strip() for c in re.split(r"\t+", cleaned) if c.strip()]

        # Try splitting by 2 or more spaces
        cols = [c.strip() for c in re.split(r"\s{2,}", cleaned) if c.strip()]
        if len(cols) > 1:
            return cols

        # Fallback to single spaces
        return [c.strip() for c in re.split(r"\s+", cleaned) if c.strip()]

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies configured column mappings to standardize DataFrame headers.

        Maps arbitrary headers (like 'I (A)' or 'V_source') to standard keys
        such as 'voltage', 'current', etc.

        Args:
            df: Raw parsed DataFrame.

        Returns:
            pd.DataFrame: DataFrame with standardized column names.
        """
        renamed_cols: dict[str, str] = {}
        for std_name, aliases in self.column_mapping.items():
            for alias in aliases:
                # Case-insensitive, strip whitespace match
                matched_col = None
                for col in df.columns:
                    if str(col).strip().lower() == alias.strip().lower():
                        matched_col = col
                        break

                if matched_col is not None:
                    renamed_cols[matched_col] = std_name
                    logger.debug(f"Mapped column '{matched_col}' to standard '{std_name}'")
                    break

        if renamed_cols:
            return df.rename(columns=renamed_cols)
        return df
