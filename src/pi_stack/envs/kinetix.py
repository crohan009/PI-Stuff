"""Kinetix wrapper — dynamic / stochastic tasks for testing RTC.

Kinetix is the simulator the RTC paper uses to validate the algorithm on
force-based, dynamic tasks (throwing, catching, balancing) that punish any
inference latency.

Repo: https://github.com/FlairOx/Kinetix

TODO:
  - install Kinetix from source
  - expose `make_kinetix_env(scenario: str)`
"""

from __future__ import annotations


def make_kinetix_env(scenario: str = "throw"):
    raise NotImplementedError
