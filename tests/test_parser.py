"""Unit tests for the COMSOL Parser and Dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from comsol.dataset import COMSOLDataset
from comsol.parser import COMSOLParser, COMSOLParsingError


@pytest.fixture
def mock_comsol_file(tmp_path: Path) -> Path:
    """Creates a temporary mock COMSOL export file for testing."""
    file_content = """% Model:         mock_memristor.mph
% Version:       COMSOL 6.1
% Parameters:    L=10e-9, W=50e-9, thickness=5
% Columns:       t  V_source  I_device  w
% Time (s)       Voltage (V)  Current (A)  FilamentWidth (nm)
0.0              0.0          0.0          1.0
0.1              0.5          1e-6         1.2
0.2              1.0          1e-5         2.0
0.3              -0.5         -1e-6        1.8
0.4              0.0          1e-8         1.5
"""
    file_path = tmp_path / "mock_export.txt"
    file_path.write_text(file_content, encoding="utf-8")
    return file_path


@pytest.fixture
def parser_instance() -> COMSOLParser:
    """Returns a COMSOLParser configured with column mappings."""
    mapping = {
        "time": ["t", "time", "time (s)"],
        "voltage": ["v_source", "voltage (v)"],
        "current": ["i_device", "current (a)"],
        "state_var": ["w", "filamentwidth (nm)"],
    }
    return COMSOLParser(column_mapping=mapping)


def test_parser_normal_file(mock_comsol_file: Path, parser_instance: COMSOLParser) -> None:
    """Verifies parsing of a normal formatted COMSOL export file."""
    dataset = parser_instance.parse(mock_comsol_file)

    assert isinstance(dataset, COMSOLDataset)
    assert dataset.shape == (5, 4)

    # Verify column renaming
    assert "time" in dataset.columns
    assert "voltage" in dataset.columns
    assert "current" in dataset.columns
    assert "state_var" in dataset.columns

    # Verify metadata extraction
    metadata = dataset.metadata
    assert metadata["model"] == "mock_memristor.mph"
    assert metadata["version"] == "COMSOL 6.1"

    # Verify parameters extraction
    params = metadata["parameters"]
    assert params["L"] == 10e-9
    assert params["W"] == 50e-9
    assert params["thickness"] == 5.0
    assert "L" in dataset.get_sweep_parameters()

    # Verify numpy conversion
    arr = dataset.to_numpy(["voltage", "current"])
    assert arr.shape == (5, 2)
    assert np.allclose(arr[2], [1.0, 1e-5])


def test_parser_missing_file(parser_instance: COMSOLParser) -> None:
    """Verifies FileNotFoundError is raised when file does not exist."""
    with pytest.raises(FileNotFoundError):
        parser_instance.parse(Path("non_existent_file.txt"))


def test_parser_empty_file(tmp_path: Path, parser_instance: COMSOLParser) -> None:
    """Verifies COMSOLParsingError is raised for an empty file."""
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")

    with pytest.raises(COMSOLParsingError):
        parser_instance.parse(empty_file)


def test_dataset_filtering() -> None:
    """Tests sweep parameter filtering in the COMSOLDataset class."""
    # Create a dataframe with synthetic sweeps
    data = pd.DataFrame(
        {
            "time": [0.1, 0.2, 0.1, 0.2],
            "voltage": [1.0, 2.0, 1.0, 2.0],
            "width": [10.0, 10.0, 20.0, 20.0],
            "current": [1e-3, 2e-3, 1.5e-3, 3e-3],
        }
    )
    metadata = {"sweep_parameters": ["width"]}
    dataset = COMSOLDataset(data, metadata)

    # Filter for width = 10.0
    filtered = dataset.filter_by_sweep({"width": 10.0})
    assert len(filtered) == 2
    assert all(filtered["width"] == 10.0)

    # Filter for width = 20.0
    filtered_20 = dataset.filter_by_sweep({"width": 20.0})
    assert len(filtered_20) == 2
    assert all(filtered_20["width"] == 20.0)

    # Non-existent sweep value
    with pytest.raises(ValueError):
        dataset.filter_by_sweep({"width": 30.0})

    # Non-existent parameter name
    with pytest.raises(ValueError):
        dataset.filter_by_sweep({"length": 10.0})
