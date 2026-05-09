"""MuJoCo baseline tabletop manipulation envs.

Mentioned in multiple PI papers as a baseline for tabletop tasks.

TODO:
  - thin wrapper around `mujoco` and (optionally) `dm_control` Suite
  - expose `make_mujoco_env(name)` returning a gymnasium.Env
"""

from __future__ import annotations


def make_mujoco_env(name: str = "tabletop_pick"):
    raise NotImplementedError
