"""Example script illustrating how to use COMSOLParser.

This script loads the default configuration, writes a mock COMSOL output file,
parses it, maps the columns, and shows the dataset metadata and values.
"""

from pathlib import Path

from comsol.parser import COMSOLParser
from utils.config_loader import load_config
from utils.logger import setup_logger

# Setup basic logging
logger = setup_logger(log_level="INFO")


def run_example() -> None:
    """Runs the parser example."""
    logger.info("Starting COMSOLParser example...")

    # Load configuration
    config = load_config()
    column_mapping = config.get("comsol_parser.column_mapping")

    # Define temporary file path
    temp_dir = Path(__file__).resolve().parent / "temp_data"
    temp_dir.mkdir(exist_ok=True)
    mock_file = temp_dir / "memristor_hysteresis.txt"

    # Write a mock COMSOL file
    logger.info(f"Writing mock COMSOL file to {mock_file}...")
    mock_content = """% Model:         Al2O3_Memristor_1D.mph
% Version:       COMSOL 6.2
% Parameters:    thickness=8e-9, area=1e-12, temperature_initial=300
% Columns:       t  V_source  I_device  w
% Time (s)       Voltage (V)  Current (A)  FilamentWidth (nm)
0.0              0.0          0.0          1.0
0.1              0.2          1.2e-7       1.05
0.2              0.4          3.5e-7       1.12
0.3              0.6          1.8e-6       1.35
0.4              0.8          2.4e-5       2.50
0.5              1.0          1.1e-4       4.00
0.6              0.5          8.2e-5       3.95
0.7              0.0          5e-6         3.70
0.8              -0.5         -1e-6        3.10
0.9              -1.0         -8.5e-5      1.20
1.0              0.0          1e-9         1.00
"""
    mock_file.write_text(mock_content, encoding="utf-8")

    # Initialize parser
    parser = COMSOLParser(column_mapping=column_mapping)

    try:
        # Parse the dataset
        dataset = parser.parse(mock_file)

        # Output parsed details
        logger.info("Successfully parsed COMSOL file!")
        logger.info(f"Dataset Shape: {dataset.shape}")
        logger.info(f"Standardized Columns: {dataset.columns}")
        logger.info(f"Extracted Metadata: {dataset.metadata}")

        print("\n=== Standardized Tabular Data ===")
        print(dataset.data.head(10))
        print("=================================\n")

        # Clean up mock file
        mock_file.unlink()
        temp_dir.rmdir()
        logger.info("Cleaned up mock files.")

    except Exception as e:
        logger.exception(f"Error during parser execution: {e}")


if __name__ == "__main__":
    run_example()
