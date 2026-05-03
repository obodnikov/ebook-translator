"""Load and merge YAML config with pydantic validation."""

from __future__ import annotations

from pathlib import Path

import yaml

from .models import Config


def load_config(path: Path | None = None) -> Config:
    """Load a YAML config file into the typed `Config` model.

    If `path` is None, returns `Config()` with built-in defaults.
    Missing fields in the file are filled from defaults.
    """
    if path is None:
        return Config()

    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    return Config.model_validate(raw)
