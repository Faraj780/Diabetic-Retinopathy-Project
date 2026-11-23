"""Load the YAML config and resolve paths in it."""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: Path | str | None = None) -> dict:
    """Read a config file (the default one if no path is given)."""
    config_path = Path(path) if path else DEFAULT_CONFIG
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not config.get("classes"):
        raise ValueError(f"{config_path} must list the lesion classes under 'classes'")
    return config


def resolve_path(value: Path | str | None) -> Path | None:
    """Turn a config path into an absolute path. Relative paths start at the project folder."""
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path
