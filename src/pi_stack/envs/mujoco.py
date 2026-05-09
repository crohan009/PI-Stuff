"""MuJoCo baseline tabletop manipulation envs.

Cited as a baseline across PI papers. We use raw `mujoco` (3.x) rather than
dm_control to keep the dependency surface narrow.

Install: ``uv sync --extra sim`` (mujoco wheels ship binaries).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MuJoCoConfig:
    name: str = "tabletop_pick"
    embodiment: str = "franka_panda"
    action_dim: int = 7
    control_hz: int = 20
    max_steps: int = 500


class MuJoCoEnv:
    """Skeleton wrapper that satisfies ``BaseEvalEnv``.

    TODO: load an MJCF scene per `name`, expose camera observations,
    implement chunked replay.
    """

    def __init__(self, config: MuJoCoConfig) -> None:
        self.config = config
        self.suite = f"mujoco_{config.name}"
        self.embodiment = config.embodiment
        self.action_dim = config.action_dim
        self.control_hz = config.control_hz

    def reset(self, *, seed: int | None = None):
        raise NotImplementedError

    def step_chunk(self, actions):
        raise NotImplementedError

    def close(self) -> None:
        pass


def make_mujoco_env(name: str = "tabletop_pick") -> MuJoCoEnv:
    return MuJoCoEnv(MuJoCoConfig(name=name))
