"""Libero benchmark wrapper.

Libero (-Spatial, -Object, -Goal, -10, -90) is the standard suite for
measuring general VLA performance. Used by π₀ and follow-ups.

Repo: https://github.com/Lifelong-Robot-Learning/LIBERO

TODO:
  - install libero from source (it's not on PyPI as of writing)
  - expose `make_libero_env(suite: str)` returning a gymnasium.Env
"""

from __future__ import annotations


def make_libero_env(suite: str = "libero_spatial"):
    raise NotImplementedError
