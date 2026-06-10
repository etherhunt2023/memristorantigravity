"""Configuration loader utility for COMSOL2Neuromorphic.

This module parses YAML files and maintains a configuration structure,
allowing unified parameter access across the framework.
"""

from pathlib import Path
from typing import Any

import yaml

from utils.logger import get_logger

logger = get_logger()


class Config:
    """A dictionary-like wrapper representing the application configuration."""

    def __init__(self, raw_config: dict[str, Any]) -> None:
        """Initializes the Config object.

        Args:
            raw_config: Dictionaried configuration parameters.
        """
        self._config = raw_config

    def get(self, key_path: str, default: Any = None) -> Any:
        """Retrieves a configuration value using a dot-separated path.

        Example:
            config.get("logging.level", "INFO")

        Args:
            key_path: Dot-separated string representing the path (e.g., 'logging.level').
            default: The fallback value if key does not exist.

        Returns:
            Any: The configuration value or default.
        """
        keys = key_path.split(".")
        current: Any = self._config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    @property
    def raw(self) -> dict[str, Any]:
        """Returns the underlying raw dictionary.

        Returns:
            Dict[str, Any]: The configuration dictionary.
        """
        return self._config


def load_config(config_path: Path | None = None) -> Config:
    """Loads configuration from default path, optionally merged with a user path.

    Args:
        config_path: Path to custom YAML configuration file.

    Returns:
        Config: Config object containing the configuration dictionary.

    Raises:
        FileNotFoundError: If the default configuration file is missing.
    """
    # Define default path relative to this file
    current_dir = Path(__file__).resolve().parent
    default_path = current_dir.parents[1] / "configs" / "default_config.yaml"

    if not default_path.exists():
        msg = f"Default configuration file not found at {default_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    with default_path.open(encoding="utf-8") as f:
        config_dict = yaml.safe_load(f) or {}

    if config_path:
        custom_path = Path(config_path)
        if custom_path.exists():
            logger.info(f"Merging user configuration from {custom_path}")
            with custom_path.open(encoding="utf-8") as f:
                custom_dict = yaml.safe_load(f) or {}
                # Simple recursive merge helper
                _merge_dicts(config_dict, custom_dict)
        else:
            logger.warning(f"Custom configuration path {custom_path} does not exist.")

    return Config(config_dict)


def _merge_dicts(base: dict[str, Any], update: dict[str, Any]) -> None:
    """Recursively merges dictionary updates into base dictionary.

    Args:
        base: The dictionary to update in-place.
        update: The dictionary containing the updates.
    """
    for key, val in update.items():
        if isinstance(val, dict) and key in base and isinstance(base[key], dict):
            _merge_dicts(base[key], val)
        else:
            base[key] = val
