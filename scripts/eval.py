"""Evaluation entrypoint.

Usage::

    uv run python scripts/eval.py --config configs/pi0.yaml --suite libero_spatial
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pi_stack.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a pi-stack policy.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--suite", type=str, default="libero_spatial")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"[eval] suite={args.suite} episodes={args.episodes}")
    print(f"[eval] model={config.get('model', {}).get('name', '<unset>')}")
    raise NotImplementedError("Eval loop not implemented yet.")


if __name__ == "__main__":
    main()
