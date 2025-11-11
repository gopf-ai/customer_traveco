"""Configuration management for Traveco forecasting system"""

import yaml
from pathlib import Path
from typing import Any, Optional


class ConfigLoader:
    """
    Load and manage project configuration from YAML files

    Supports nested configuration access using dot notation.
    Example: config.get('data.raw_path', default='/default/path')
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize configuration loader

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            # Try relative to project root
            project_root = Path(__file__).parent.parent.parent
            self.config_path = project_root / config_path

        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                f"Please create config/config.yaml"
            )

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports dot notation)

        Args:
            key: Configuration key (e.g., 'data.raw_path')
            default: Default value if key not found

        Returns:
            Configuration value or default

        Examples:
            >>> config = ConfigLoader()
            >>> config.get('data.raw_path')
            '../data/raw/'
            >>> config.get('data.invalid_key', default='fallback')
            'fallback'
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_path(self, key: str, default: Optional[Path] = None) -> Path:
        """
        Get configuration value as Path object

        Args:
            key: Configuration key
            default: Default Path if key not found

        Returns:
            Path object
        """
        value = self.get(key, default)

        if value is None:
            raise ValueError(f"Configuration key '{key}' not found and no default provided")

        path = Path(value)

        # If relative path, make it relative to project root
        if not path.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            path = project_root / path

        return path

    def reload(self):
        """Reload configuration from file"""
        self.config = self._load_config()

    def __repr__(self) -> str:
        return f"ConfigLoader(config_path='{self.config_path}')"
