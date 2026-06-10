"""Unit tests for utility modules (logger, config_loader)."""

from pathlib import Path

import yaml

from utils.config_loader import Config, load_config
from utils.logger import get_logger, setup_logger


def test_logger_creation(tmp_path: Path) -> None:
    """Verifies that the logger can be set up and writes to a file."""
    log_file = tmp_path / "test.log"
    logger = setup_logger(name="test_logger", log_level="DEBUG", log_file=log_file)
    assert logger.name == "test_logger"

    logger.debug("Test debug message")
    logger.info("Test info message")

    assert log_file.exists()
    log_content = log_file.read_text(encoding="utf-8")
    assert "DEBUG" in log_content
    assert "Test debug message" in log_content
    assert "INFO" in log_content
    assert "Test info message" in log_content

    # Get existing logger
    logger_retrieved = get_logger("test_logger")
    assert logger_retrieved == logger


def test_config_loader(tmp_path: Path) -> None:
    """Tests loading and merging configurations."""
    # Test dot-notation wrapper Config
    raw = {"a": {"b": 1, "c": "test"}, "d": 4}
    cfg = Config(raw)
    assert cfg.get("a.b") == 1
    assert cfg.get("a.c") == "test"
    assert cfg.get("d") == 4
    assert cfg.get("a.nonexistent", "default") == "default"
    assert cfg.get("nonexistent") is None
    assert cfg.raw == raw

    # Test loading default configuration
    default_cfg = load_config()
    assert default_cfg.get("logging.level") in ["INFO", "DEBUG", "WARNING", "ERROR"]

    # Test merging with custom configuration
    custom_yaml = tmp_path / "custom_config.yaml"
    custom_data = {
        "logging": {"level": "DEBUG"},
        "new_section": {"param": 42},
    }
    with custom_yaml.open("w", encoding="utf-8") as f:
        yaml.dump(custom_data, f)

    merged_cfg = load_config(custom_yaml)
    assert merged_cfg.get("logging.level") == "DEBUG"
    assert merged_cfg.get("new_section.param") == 42
    # Ensure default fields are preserved
    assert merged_cfg.get("crossbar.line_resistance") == 1.5
