# src/utils/config_loader.py
import os
import yaml
from typing import Any, Dict

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load YAML config using an absolute path so it works when ZenML changes cwd.
    """
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Config file not found: {abs_path}")
    with open(abs_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
