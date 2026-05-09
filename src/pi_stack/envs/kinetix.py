"""Kinetix wrapper — dynamic / stochastic tasks for testing RTC.

Kinetix is the simulator the RTC paper uses to validate the algorithm on
force-based tasks (throwing, catching, balancing) where any inference
latency shows up immediately as a missed catch or a dropped throw.

Repo: https://github.com/FlairOx/Kinetix
Install: see docs/sim-setup.md.

Use this suite specifically when validating ``pi_stack.inference.rtc``.
Static benchmarks (Libero) won't expose latency bugs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

KinetixScenario = Literal["throw", "catch", "balance", "match"]

KINETIX_SCENARIOS: tuple[KinetixScenario, ...] = ("throw", "catch", "balance", "match")


@dataclass
class KinetixConfig:
    scenario: KinetixScenario = "throw"
    embodiment: str = "kinetix_arm"
    action_dim: int = 6
    control_hz: int = 50      # high — that's the whole point
    max_steps: int = 400
    inject_latency_ms: int = 0   # set >300 to stress-test RTC


class KinetixEnv:
    """Skeleton wrapper that satisfies ``BaseEvalEnv``."""

    def __init__(self, config: KinetixConfig) -> None:
        self.config = config
        self.suite = f"kinetix_{config.scenario}"
        self.embodiment = config.embodiment
        self.action_dim = config.action_dim
        self.control_hz = config.control_hz

    def reset(self, *, seed: int | None = None):
        raise NotImplementedError

    def step_chunk(self, actions):
        raise NotImplementedError

    def close(self) -> None:
        pass


def make_kinetix_env(scenario: KinetixScenario = "throw") -> KinetixEnv:
    return KinetixEnv(KinetixConfig(scenario=scenario))
