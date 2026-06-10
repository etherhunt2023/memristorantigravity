"""Core metrics calculation for memristor parameter extraction.

This module provides utility functions to calculate resistances, threshold
switching voltages, non-linearity, and transient response times from raw data.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from extraction.exceptions import ExtractionError
from utils.logger import get_logger

logger = get_logger()


def segment_iv_loop(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Segments an I-V curve into monotonic voltage branches.

    Splits the dataframe based on local extrema of the voltage sweep to separate
    the forward and reverse sweep branches.

    Args:
        df: DataFrame containing 'voltage' and 'current' columns.

    Returns:
        list[pd.DataFrame]: Monotonic voltage branches.
    """
    v = df["voltage"].to_numpy()
    if len(v) < 3:
        return [df]

    # Find local extrema (peaks and valleys) of the voltage sweep
    extrema_indices = [0]
    for i in range(1, len(v) - 1):
        is_extremum = (v[i] > v[i - 1] and v[i] >= v[i + 1]) or (
            v[i] < v[i - 1] and v[i] <= v[i + 1]
        )
        if is_extremum and i - extrema_indices[-1] > 1:
            extrema_indices.append(i)
    extrema_indices.append(len(v) - 1)

    branches = []
    for start, end in zip(extrema_indices[:-1], extrema_indices[1:], strict=False):
        if end - start > 1:
            branches.append(df.iloc[start : end + 1])
    return branches


def extract_lrs_hrs(df: pd.DataFrame, read_voltage: float) -> tuple[float, float]:
    """Extracts the Low-Resistance State (LRS) and High-Resistance State (HRS) resistances.

    Interpolates the currents at the read voltage across the different sweep
    branches and maps them to LRS and HRS based on their conductance levels.

    Args:
        df: DataFrame containing 'voltage' and 'current' columns.
        read_voltage: The voltage at which to read the resistance.

    Returns:
        Tuple[float, float]: (R_lrs, R_hrs) in Ohms.

    Raises:
        ExtractionError: If the read voltage is outside the sweep range.
    """
    if "voltage" not in df.columns or "current" not in df.columns:
        raise ExtractionError("DataFrame must contain 'voltage' and 'current' columns.")

    branches = segment_iv_loop(df)
    currents_at_vread = []

    for branch in branches:
        v = branch["voltage"].to_numpy()
        i = branch["current"].to_numpy()

        # Check if read_voltage is within the range of this branch
        if min(v) <= read_voltage <= max(v):
            # Sort for 1d interpolation
            sort_idx = np.argsort(v)
            v_sorted = v[sort_idx]
            i_sorted = i[sort_idx]

            # Remove duplicates to avoid interpolation errors
            v_uniq, uniq_idx = np.unique(v_sorted, return_index=True)
            i_uniq = i_sorted[uniq_idx]

            if len(v_uniq) >= 2:
                f_interp = interp1d(v_uniq, i_uniq, kind="linear")
                currents_at_vread.append(float(f_interp(read_voltage)))

    if not currents_at_vread:
        msg = f"Read voltage {read_voltage} V is outside the simulated voltage range."
        raise ExtractionError(msg)

    # Convert currents to resistances (R = V / I)
    resistances = []
    for i_val in currents_at_vread:
        if abs(i_val) > 1e-20:
            resistances.append(abs(read_voltage / i_val))
        else:
            resistances.append(float("inf"))

    if len(resistances) < 2:
        logger.warning(
            f"Only one branch covered the read voltage {read_voltage} V. "
            "Using it as both LRS and HRS."
        )
        r_val = resistances[0] if resistances else float("inf")
        return r_val, r_val

    # Sort resistances: smallest is LRS, largest is HRS
    resistances.sort()
    return resistances[0], resistances[-1]


def extract_switching_voltages(
    df: pd.DataFrame,
    method: str = "derivative",
    set_threshold_curr: float = 1e-5,
    reset_threshold_curr: float = 1e-5,
) -> tuple[float, float]:
    """Extracts the set voltage (V_set) and reset voltage (V_reset).

    Args:
        df: DataFrame containing 'voltage' and 'current' columns.
        method: Extraction method ('derivative' or 'threshold').
        set_threshold_curr: Current threshold for Set voltage.
        reset_threshold_curr: Current threshold for Reset voltage.

    Returns:
        Tuple[float, float]: (V_set, V_reset) in Volts.

    Raises:
        ExtractionError: If set or reset transitions cannot be resolved.
    """
    if "voltage" not in df.columns or "current" not in df.columns:
        raise ExtractionError("DataFrame must contain 'voltage' and 'current' columns.")

    branches = segment_iv_loop(df)

    v_set = float("nan")
    v_reset = float("nan")

    if method == "threshold":
        # Find V_set (typically positive voltage, switching from low to high current)
        for branch in branches:
            # Look at positive sweep branch (increasing V)
            v = branch["voltage"].to_numpy()
            i = branch["current"].to_numpy()

            if v[-1] > v[0]:  # Forward sweep
                # Find where current crosses threshold in positive region
                pos_indices = np.where((v > 0) & (np.abs(i) >= set_threshold_curr))[0]
                if len(pos_indices) > 0:
                    v_set = float(v[pos_indices[0]])
                    break

        # Find V_reset (typically negative voltage, switching from high to low current magnitude)
        for branch in branches:
            v = branch["voltage"].to_numpy()
            i = branch["current"].to_numpy()

            if v[-1] < v[0]:  # Reverse sweep
                # Find where current magnitude drops below threshold in negative region
                neg_indices = np.where((v < 0) & (np.abs(i) >= reset_threshold_curr))[0]
                if len(neg_indices) > 0:
                    # Let's take the voltage where it drops below threshold
                    # or where the transition starts.
                    v_reset = float(v[neg_indices[-1]])
                    break

    elif method == "derivative":
        # Derivative method: Find peak of d(log(|I|))/dV or dI/dV
        for branch in branches:
            v = branch["voltage"].to_numpy()
            i = branch["current"].to_numpy()

            if len(v) < 3:
                continue

            # Remove duplicate adjacent voltage values to avoid divide-by-zero in gradient
            diffs = np.diff(v)
            unique_mask = np.concatenate(([True], diffs != 0.0))
            v = v[unique_mask]
            i = i[unique_mask]

            if len(v) < 3:
                continue

            # Compute derivatives
            # Use log-current to capture exponential switching transitions
            log_abs_i = np.log10(np.abs(i) + 1e-20)
            didv = np.gradient(log_abs_i, v)

            # Set transition: increasing positive voltage, positive gradient peak
            if v[-1] > v[0]:
                v_max = np.max(v)
                pos_mask = (v > 0.1 * v_max) & (didv > 0)
                if np.any(pos_mask):
                    peak_idx = np.where(pos_mask, didv, -np.inf).argmax()
                    v_set = float(v[peak_idx])

            # Reset transition: decreasing negative voltage, positive gradient peak
            if v[-1] < v[0]:
                v_min = np.min(v)
                neg_mask = (v < 0.1 * v_min) & (didv > 0)
                if np.any(neg_mask):
                    peak_idx = np.where(neg_mask, didv, -np.inf).argmax()
                    v_reset = float(v[peak_idx])

    else:
        raise ValueError(f"Unknown switching voltage extraction method: {method}")

    if np.isnan(v_set):
        logger.warning("V_set could not be extracted from the dataset.")
    if np.isnan(v_reset):
        logger.warning("V_reset could not be extracted from the dataset.")

    return v_set, v_reset


def extract_non_linearity(df: pd.DataFrame, read_voltage: float) -> float:
    """Calculates the non-linearity ratio of the memristor in LRS.

    Ratio is defined as: I(V_read) / I(V_read / 2)

    Args:
        df: DataFrame containing 'voltage' and 'current' columns.
        read_voltage: Read voltage (V_read).

    Returns:
        float: Non-linearity ratio.

    Raises:
        ExtractionError: If the LRS branch cannot be identified or ratio is invalid.
    """
    branches = segment_iv_loop(df)
    lrs_branch = None
    max_conductance = -1.0

    half_voltage = read_voltage / 2.0
    # Find the branch with the highest conductance at half read voltage
    for branch in branches:
        v = branch["voltage"].to_numpy()
        i = branch["current"].to_numpy()

        low_v = min(read_voltage, half_voltage)
        high_v = max(read_voltage, half_voltage)

        if min(v) <= low_v and high_v <= max(v):
            sort_idx = np.argsort(v)
            v_uniq, uniq_idx = np.unique(v[sort_idx], return_index=True)
            i_uniq = i[sort_idx][uniq_idx]

            if len(v_uniq) >= 2:
                f_interp = interp1d(v_uniq, i_uniq, kind="linear")
                i_half_val = abs(float(f_interp(half_voltage)))
                cond = i_half_val / abs(half_voltage)
                if cond > max_conductance:
                    max_conductance = cond
                    lrs_branch = (v_uniq, i_uniq)

    if lrs_branch is None:
        raise ExtractionError("Could not locate LRS branch covering the read voltage.")

    v_uniq, i_uniq = lrs_branch
    f_interp = interp1d(v_uniq, i_uniq, kind="linear")

    half_voltage = read_voltage / 2.0
    if not (min(v_uniq) <= half_voltage <= max(v_uniq)):
        msg = f"Half read voltage {half_voltage} V is outside the branch range."
        raise ExtractionError(msg)

    i_full = abs(float(f_interp(read_voltage)))
    i_half = abs(float(f_interp(half_voltage)))

    if i_half < 1e-20:
        return float("inf")

    return i_full / i_half


def extract_switching_time(df: pd.DataFrame) -> float:
    """Extracts switching response time from transient simulation datasets.

    Calculates the 10% to 90% rise/fall time of the state variable or current.

    Args:
        df: DataFrame containing 'time' and either 'state_var' or 'current'.

    Returns:
        float: Switching time in seconds.

    Raises:
        ExtractionError: If the dataset does not contain transient sweeps.
    """
    if "time" not in df.columns:
        raise ExtractionError("Transient dataset must contain a 'time' column.")

    # Determine which variable to track for switching (state_var takes priority, then current)
    target_col = None
    for col in ["state_var", "current"]:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        raise ExtractionError("Transient dataset must contain 'state_var' or 'current'.")

    t = df["time"].to_numpy()
    y = df[target_col].to_numpy()

    if len(t) < 3:
        raise ExtractionError("Insufficient data points in transient dataset.")

    y_min, y_max = np.min(y), np.max(y)
    y_range = y_max - y_min

    if y_range < 1e-15:
        return 0.0  # No transition occurred

    # Calculate 10% and 90% levels
    level_10 = y_min + 0.1 * y_range
    level_90 = y_min + 0.9 * y_range

    # Find when the signal crosses 10% and 90% thresholds
    # We support both rising and falling transitions
    is_rising = y[-1] > y[0]

    if is_rising:
        idx_10 = np.where(y >= level_10)[0][0]
        idx_90 = np.where(y >= level_90)[0][0]
    else:
        idx_10 = np.where(y <= level_90)[0][0]  # For falling, 90% level is crossed first
        idx_90 = np.where(y <= level_10)[0][0]

    t_10 = t[idx_10]
    t_90 = t[idx_90]

    return abs(t_90 - t_10)
