"""Example script illustrating parameter extraction.

This script writes a mock memristor I-V sweep, parses it with the COMSOLParser,
runs the ElectricalParameterExtractor, and prints the extracted physical results.
"""

from pathlib import Path

import numpy as np

from comsol.parser import COMSOLParser
from extraction.extractor import ElectricalParameterExtractor
from utils.config_loader import load_config
from utils.logger import setup_logger

# Setup logging
logger = setup_logger(log_level="INFO")


def create_mock_iv_file(file_path: Path) -> None:
    """Generates a mock hysteretic memristor dataset and writes it to a file.

    Args:
        file_path: Target path for the file.
    """
    # Create standard triangular voltage sweep
    v_forward = np.linspace(0.0, 2.0, 51)
    v_backward = np.linspace(2.0, -2.0, 101)
    v_return = np.linspace(-2.0, 0.0, 51)

    r_hrs = 1e5
    r_lrs = 5e2

    # Assemble currents with hysteretic transitions
    # Set at +1.2V, Reset at -1.2V
    currents = []
    voltages = np.concatenate([v_forward, v_backward, v_return])

    # State: starts in HRS (0), switches to LRS (1)
    state = 0.0

    for v in voltages:
        # Check Set switch
        if v > 1.2 and state == 0.0:
            state = 1.0
        # Check Reset switch
        elif v < -1.2 and state == 1.0:
            state = 0.0

        r = r_lrs if state == 1.0 else r_hrs
        # Add slight exponential nonlinearity to current
        nonlin_factor = 1.0 + 0.1 * (v**2)
        currents.append((v / r) * nonlin_factor)

    # Write formatted COMSOL export comments
    with file_path.open("w", encoding="utf-8") as f:
        f.write("% Model:         TiO2_Memristor_Sweep.mph\n")
        f.write("% Version:       COMSOL 6.2\n")
        f.write("% Parameters:    width=50e-9, height=50e-9, contact_metal=Pt\n")
        f.write("% Columns:       V_sweep  I_sweep\n")
        f.write("% Voltage (V)    Current (A)\n")

        for v, i in zip(voltages, currents, strict=False):
            f.write(f"{v:12.6f}    {i:14.6e}\n")


def run_example() -> None:
    """Runs the parameter extraction example."""
    logger.info("Starting ElectricalParameterExtractor example...")

    # Load configuration
    config = load_config()

    # Define temporary file path
    temp_dir = Path(__file__).resolve().parent / "temp_data"
    temp_dir.mkdir(exist_ok=True)
    mock_file = temp_dir / "memristor_iv_sweep.txt"

    try:
        # Create and write mock data
        create_mock_iv_file(mock_file)

        # Parse the dataset
        parser = COMSOLParser(column_mapping=config.get("comsol_parser.column_mapping"))
        dataset = parser.parse(mock_file)

        # Initialize extractor with loaded configs
        extractor = ElectricalParameterExtractor(config)

        # Perform extraction
        parameters = extractor.extract(dataset)

        logger.info("Successfully extracted memristor parameters:")
        print("\n================ Extracted Device Metrics ================")
        print(parameters)
        print("==========================================================")

        # Clean up files
        mock_file.unlink()
        temp_dir.rmdir()
        logger.info("Cleaned up temp data files.")

    except Exception as e:
        logger.exception(f"Error during parameter extraction execution: {e}")


if __name__ == "__main__":
    run_example()
