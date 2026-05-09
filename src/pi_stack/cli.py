"""CLI entrypoints declared in pyproject.toml.

These are deliberately thin wrappers around the corresponding scripts so the
implementation lives in `scripts/` (easier to read, edit, copy).
"""

from __future__ import annotations


def train() -> None:
    from scripts.train import main

    main()


def eval_() -> None:
    from scripts.eval import main

    main()


def serve() -> None:
    from scripts.infer_server import main

    main()
