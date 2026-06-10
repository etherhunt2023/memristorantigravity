"""Logging utility for the COMSOL2Neuromorphic framework.

This module provides setup functions to initialize and retrieve unified loggers
configured to log to both stdout and a rolling log file.
"""

import logging
from pathlib import Path


def setup_logger(
    name: str = "comsol2neuromorphic",
    log_level: str = "INFO",
    log_file: Path | None = None,
    console_output: bool = True,
) -> logging.Logger:
    """Configures and retrieves a logger instance.

    Args:
        name: Name of the logger.
        log_level: Level of logging (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file: Optional file path to write log outputs.
        console_output: If True, outputs logs to stdout.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)

    # If the logger has handlers, it is already configured
    if logger.handlers:
        return logger

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        # Ensure parent directories exist
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "comsol2neuromorphic") -> logging.Logger:
    """Gets the existing logger instance by name.

    Args:
        name: Name of the logger to retrieve.

    Returns:
        logging.Logger: The logger instance.
    """
    return logging.getLogger(name)
