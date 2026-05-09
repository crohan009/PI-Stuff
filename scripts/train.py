"""Training entrypoint.

Usage::

    uv run python scripts/train.py --config configs/pi0.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pi_stack.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a pi-stack policy.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"[train] loaded config: {args.config}")
    print(f"[train] model name: {config.get('model', {}).get('name', '<unset>')}")
    print("[train] TODO: dispatch to KITrainer / RECAPTrainer based on training.recipe")
    raise NotImplementedError("Trainer dispatch not implemented yet.")


if __name__ == "__main__":
    main()
