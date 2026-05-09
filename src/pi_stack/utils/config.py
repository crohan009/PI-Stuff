"""YAML config loader.

All entrypoints take a config path; configs live in `configs/`. Keeping a
plain pydantic loader avoids pulling in hydra for a one-person project,
while still giving validation errors at load time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file as a plain dict.

    TODO: when a model is fleshed out, swap the return type for a typed
    pydantic model and validate here. For now, raw dicts keep the scaffold
    flexible.
    """
    with Path(path).open("r") as f:
        return yaml.safe_load(f) or {}
