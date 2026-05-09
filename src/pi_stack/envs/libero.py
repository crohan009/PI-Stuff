"""Libero benchmark wrapper.

Libero is the standard suite for measuring general VLA performance. Used by
π₀ and follow-ups. Five sub-suites: Spatial, Object, Goal, 10, 90.

Repo: https://github.com/Lifelong-Robot-Learning/LIBERO
Install: see docs/sim-setup.md (it's not on PyPI).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

LiberoSuite = Literal[
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
]

LIBERO_SUITES: tuple[LiberoSuite, ...] = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
)


@dataclass
class LiberoConfig:
    suite: LiberoSuite = "libero_spatial"
    embodiment: str = "franka_panda"
    action_dim: int = 7        # 6 DoF EEF + 1 gripper
    control_hz: int = 20
    max_steps: int = 600
    image_resolution: int = 224


class LiberoEnv:
    """Skeleton wrapper that satisfies ``BaseEvalEnv``.

    TODO:
      - lazy-import `libero.libero` (heavy native deps)
      - construct the suite's task list and pick by index
      - chunk replay loop in `step_chunk`
    """

    def __init__(self, config: LiberoConfig) -> None:
        self.config = config
        self.suite = config.suite
        self.embodiment = config.embodiment
        self.action_dim = config.action_dim
        self.control_hz = config.control_hz

    def reset(self, *, seed: int | None = None):
        raise NotImplementedError

    def step_chunk(self, actions):
        raise NotImplementedError

    def close(self) -> None:
        pass


def make_libero_env(suite: LiberoSuite = "libero_spatial") -> LiberoEnv:
    return LiberoEnv(LiberoConfig(suite=suite))
