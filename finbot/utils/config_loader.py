"""
Configuration Loader for FINBOT
--------------------------------
Handles reading YAML or JSON configuration files safely.
If no config file exists, returns a default empty configuration.
"""

import yaml
import json
from pathlib import Path


class ConfigLoader:
    """Unified configuration loader for FINBOT."""

    def __init__(self, config_path: str | None = None):
        """
        Initialize configuration loader.

        Args:
            config_path (str | None): Path to config file (YAML/JSON). 
                                      If None, it auto-searches common locations.
        """
        # Auto-detect config file if not provided
        if config_path is None:
            candidates = [
                Path(__file__).resolve().parents[1] / "config.yaml",   # finbot/config.yaml
                Path(__file__).resolve().parents[2] / "config.yaml",   # project root
            ]
            for candidate in candidates:
                if candidate.exists():
                    config_path = candidate
                    break

        self.config_path = Path(config_path) if config_path else None
        self.config = self._load_config()

    # ---------------------------------------------------------------------
    def _load_config(self) -> dict:
        """Attempt to load config file. Return defaults if missing."""
        if not self.config_path or not self.config_path.exists():
            print("[FINBOT] ⚠️ No configuration file found. Using default settings.")
            return {}

        try:
            if self.config_path.suffix in (".yaml", ".yml"):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
            elif self.config_path.suffix == ".json":
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                print(f"[FINBOT] ⚠️ Unsupported config format: {self.config_path.suffix}")
                return {}
        except Exception as e:
            print(f"[FINBOT] ⚠️ Error loading config: {e}")
            return {}

    # ---------------------------------------------------------------------
    def get(self, key: str, default=None):
        """Retrieve a value from config."""
        return self.config.get(key, default)

    def reload(self):
        """Reload config file from disk."""
        self.config = self._load_config()
        return self.config
