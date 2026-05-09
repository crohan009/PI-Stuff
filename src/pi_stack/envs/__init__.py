"""Simulation environment wrappers — Libero, Kinetix, MuJoCo.

OXE is a dataset, not a sim, so it lives in :mod:`pi_stack.data.oxe`.
"""

from pi_stack.envs.base import BaseEvalEnv, StepResult
from pi_stack.envs.kinetix import KinetixEnv, make_kinetix_env
from pi_stack.envs.libero import LiberoEnv, make_libero_env
from pi_stack.envs.mujoco import MuJoCoEnv, make_mujoco_env

__all__ = [
    "BaseEvalEnv",
    "StepResult",
    "LiberoEnv",
    "KinetixEnv",
    "MuJoCoEnv",
    "make_libero_env",
    "make_kinetix_env",
    "make_mujoco_env",
    "make_env",
]


def make_env(suite: str, **kwargs) -> BaseEvalEnv:
    """Dispatch to the right wrapper by suite name.

    Recognized prefixes:
        - "libero_*"   → :func:`make_libero_env`
        - "kinetix_*"  → :func:`make_kinetix_env`
        - "mujoco_*"   → :func:`make_mujoco_env`
    """
    if suite.startswith("libero_"):
        return make_libero_env(suite, **kwargs)  # type: ignore[arg-type]
    if suite.startswith("kinetix_"):
        scenario = suite.removeprefix("kinetix_")
        return make_kinetix_env(scenario, **kwargs)  # type: ignore[arg-type]
    if suite.startswith("mujoco_"):
        name = suite.removeprefix("mujoco_")
        return make_mujoco_env(name, **kwargs)
    raise ValueError(
        f"Unknown suite '{suite}'. Expected libero_*, kinetix_*, or mujoco_*."
    )
