"""Common eval-env protocol.

`scripts/eval.py` dispatches across Libero / Kinetix / MuJoCo / (and OXE
replay) without caring which one is plugged in. Every concrete env wrapper
returns something that satisfies ``BaseEvalEnv``.

Why a tiny custom protocol instead of bare ``gymnasium.Env``: the PI policies
work on action **chunks** (H=50) not single steps, so the loop semantics are
"submit a chunk, advance H steps, observe again." We expose that explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class StepResult:
    obs: dict[str, Any]
    reward: float
    done: bool
    success: bool
    info: dict[str, Any]


@runtime_checkable
class BaseEvalEnv(Protocol):
    """Protocol every sim wrapper must implement."""

    suite: str          # short id, e.g. "libero_spatial"
    embodiment: str     # e.g. "franka_panda", "widowx"
    action_dim: int
    control_hz: int

    def reset(self, *, seed: int | None = None) -> dict[str, Any]:
        """Reset and return the initial observation dict."""
        ...

    def step_chunk(self, actions: np.ndarray) -> StepResult:
        """Execute one action chunk of shape (H, action_dim).

        The wrapper is responsible for replaying the chunk at ``control_hz``
        and aggregating reward/done. Returns the next observation.
        """
        ...

    def close(self) -> None: ...
