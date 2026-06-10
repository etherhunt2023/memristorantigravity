"""Unit tests for the parameter extraction module."""

import numpy as np
import pandas as pd
import pytest

from comsol.dataset import COMSOLDataset
from extraction import metrics
from extraction.exceptions import ExtractionError
from extraction.extractor import ElectricalParameterExtractor, ExtractedParameters
from utils.config_loader import Config


@pytest.fixture
def synthetic_iv_data() -> pd.DataFrame:
    """Generates synthetic bipolar memristor hysteretic I-V loop.

    Sweep sequence:
    0V -> 1.5V (Forward positive, Set occurs at 1.0V)
    1.5V -> 0V (Backward positive, stays in LRS)
    0V -> -1.5V (Forward negative, Reset occurs at -1.0V)
    -1.5V -> 0V (Backward negative, stays in HRS)
    """
    # Create monotonic voltage branches
    v_fwd_pos = np.linspace(0, 1.5, 31)
    v_bwd_pos = np.linspace(1.5, 0, 31)
    v_fwd_neg = np.linspace(0, -1.5, 31)
    v_bwd_neg = np.linspace(-1.5, 0, 31)

    # Resistances
    r_hrs = 1e6
    r_lrs = 1e3

    # Generate current for forward positive (Set at 1.0V)
    i_fwd_pos = []
    for v in v_fwd_pos:
        r = r_hrs if v < 1.0 else r_lrs
        i_fwd_pos.append(v / r if r > 0 else 0.0)

    # Generate current for backward positive (stays in LRS)
    i_bwd_pos = v_bwd_pos / r_lrs

    # Generate current for forward negative (Reset at -1.0V)
    i_fwd_neg = []
    for v in v_fwd_neg:
        r = r_lrs if abs(v) < 1.0 else r_hrs
        i_fwd_neg.append(v / r)

    # Generate current for backward negative (stays in HRS)
    i_bwd_neg = v_bwd_neg / r_hrs

    # Concatenate all segments
    voltages = np.concatenate([v_fwd_pos, v_bwd_pos[1:], v_fwd_neg[1:], v_bwd_neg[1:]])
    currents = np.concatenate([i_fwd_pos, i_bwd_pos[1:], i_fwd_neg[1:], i_bwd_neg[1:]])

    # Add a mock time column
    times = np.linspace(0, 1.0, len(voltages))

    return pd.DataFrame({"time": times, "voltage": voltages, "current": currents})


@pytest.fixture
def synthetic_transient_data() -> pd.DataFrame:
    """Generates synthetic transient state variable switching transition."""
    times = np.linspace(0, 10e-9, 101)  # 0 to 10 ns
    # Sigmoidal switching transition centered at 5 ns with width parameter 0.5 ns
    state_var = 1.0 / (1.0 + np.exp(-(times - 5e-9) / 0.5e-9))
    voltage = np.ones_like(times) * 2.0  # Apply constant read pulse
    voltage[times < 1e-9] = 0.0
    current = state_var * 1e-5

    return pd.DataFrame(
        {"time": times, "voltage": voltage, "state_var": state_var, "current": current}
    )


def test_segment_iv_loop(synthetic_iv_data: pd.DataFrame) -> None:
    """Tests that I-V curve segmentation yields correct monotonic branches."""
    branches = metrics.segment_iv_loop(synthetic_iv_data)
    # Extrema are: 0V -> 1.5V -> -1.5V -> 0V.
    # This leads to 3 monotonic branches:
    # 1. 0 to 1.5 (increasing)
    # 2. 1.5 to -1.5 (decreasing)
    # 3. -1.5 to 0 (increasing)
    assert len(branches) == 3
    assert np.allclose(branches[0]["voltage"].iloc[0], 0.0)
    assert np.allclose(branches[0]["voltage"].iloc[-1], 1.5)
    assert np.allclose(branches[1]["voltage"].iloc[0], 1.5)
    assert np.allclose(branches[1]["voltage"].iloc[-1], -1.5)
    assert np.allclose(branches[2]["voltage"].iloc[0], -1.5)
    assert np.allclose(branches[2]["voltage"].iloc[-1], 0.0)


def test_extract_lrs_hrs(synthetic_iv_data: pd.DataFrame) -> None:
    """Tests LRS and HRS extraction from synthetic loop data."""
    # Measure at V_read = 0.5V
    r_lrs, r_hrs = metrics.extract_lrs_hrs(synthetic_iv_data, read_voltage=0.5)
    assert np.isclose(r_lrs, 1e3, rtol=1e-2)
    assert np.isclose(r_hrs, 1e6, rtol=1e-2)


def test_extract_switching_voltages(synthetic_iv_data: pd.DataFrame) -> None:
    """Tests Set and Reset voltage extraction using derivative and threshold methods."""
    # 1. Derivative method (default)
    v_set, v_reset = metrics.extract_switching_voltages(synthetic_iv_data, method="derivative")
    assert np.isclose(v_set, 1.0, atol=0.1)
    assert np.isclose(v_reset, -1.0, atol=0.1)

    # 2. Threshold method
    # Set threshold current = 2e-4 (corresponds to LRS transition at 1V)
    # Reset threshold current = 2e-4
    v_set_t, v_reset_t = metrics.extract_switching_voltages(
        synthetic_iv_data,
        method="threshold",
        set_threshold_curr=2e-4,
        reset_threshold_curr=2e-4,
    )
    assert np.isclose(v_set_t, 1.0, atol=0.1)
    assert np.isclose(v_reset_t, -1.0, atol=0.1)


def test_extract_non_linearity(synthetic_iv_data: pd.DataFrame) -> None:
    """Tests non-linearity ratio extraction."""
    # V_read = 1.0V. At 1.0V current in LRS is 1.0 / 1e3 = 1e-3.
    # At 0.5V current in LRS is 0.5 / 1e3 = 0.5e-3.
    # Non-linearity ratio = I(1.0) / I(0.5) = 2.0 (since LRS branch is linear in our mock)
    nl_ratio = metrics.extract_non_linearity(synthetic_iv_data, read_voltage=1.0)
    assert np.isclose(nl_ratio, 2.0, rtol=1e-2)


def test_extract_switching_time(synthetic_transient_data: pd.DataFrame) -> None:
    """Tests rise/fall response time calculation from transient data."""
    # Sigmoidal width parameter is 0.5 ns.
    # Analytically, sigmoid(x) goes from 10% to 90% in log(9) * width_param
    # log(9) * 0.5 ns = 2.197 * 0.5 ns = 1.098 ns
    sw_time = metrics.extract_switching_time(synthetic_transient_data)
    assert np.isclose(sw_time, 1.098e-9, rtol=1e-2)


def test_extractor_pipeline(synthetic_iv_data: pd.DataFrame) -> None:
    """Tests high-level parameter extractor pipeline class."""
    config_dict = {
        "extraction": {
            "lrs_read_voltage": 0.5,
            "hrs_read_voltage": 0.5,
            "non_linearity_voltage": 1.0,
            "method": "derivative",
        }
    }
    config = Config(config_dict)
    extractor = ElectricalParameterExtractor(config)

    dataset = COMSOLDataset(synthetic_iv_data, {"model": "test.mph"})
    params = extractor.extract(dataset)

    assert isinstance(params, ExtractedParameters)
    assert np.isclose(params.r_lrs, 1e3, rtol=1e-2)
    assert np.isclose(params.r_hrs, 1e6, rtol=1e-2)
    assert np.isclose(params.r_ratio, 1e3, rtol=1e-2)
    assert np.isclose(params.v_set, 1.0, atol=0.1)
    assert np.isclose(params.v_reset, -1.0, atol=0.1)
    assert np.isclose(params.non_linearity, 2.0, rtol=1e-2)


def test_extractor_invalid_data() -> None:
    """Verifies pipeline raises ExtractionError for invalid files."""
    extractor = ElectricalParameterExtractor()
    bad_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    dataset = COMSOLDataset(bad_df, {})

    with pytest.raises(ExtractionError):
        extractor.extract(dataset)
