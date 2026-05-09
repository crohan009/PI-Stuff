"""Smoke tests — confirm the package imports and the per-paper modules wire up.

Real tests arrive when each module gains real code.
"""

from __future__ import annotations

import importlib

PAPER_MODULES = [
    "pi_stack",
    "pi_stack.models.pi0",
    "pi_stack.models.pi05",
    "pi_stack.models.pi06",
    "pi_stack.models.pi07",
    "pi_stack.models.hi_robot",
    "pi_stack.models.action_expert",
    "pi_stack.models.backbones",
    "pi_stack.tokenization.fast",
    "pi_stack.training.flow_matching",
    "pi_stack.training.ki",
    "pi_stack.training.recap",
    "pi_stack.inference.rtc",
    "pi_stack.inference.server",
    "pi_stack.memory.mem",
    "pi_stack.rlt.rl_token",
    "pi_stack.data.chunking",
    "pi_stack.data.synthetic",
    "pi_stack.data.human_to_robot",
    "pi_stack.envs.libero",
    "pi_stack.envs.kinetix",
    "pi_stack.envs.mujoco",
    "pi_stack.utils.config",
]


def test_all_paper_modules_import() -> None:
    for name in PAPER_MODULES:
        importlib.import_module(name)


def test_version_exposed() -> None:
    import pi_stack

    assert isinstance(pi_stack.__version__, str) and pi_stack.__version__


def test_pi07_inherits_through_arc() -> None:
    """π₀.₇ should be reachable as a subclass of π*₀.₆ → π₀.₅ → π₀."""
    from pi_stack.models.pi0 import Pi0Policy
    from pi_stack.models.pi07 import Pi07Policy

    assert issubclass(Pi07Policy, Pi0Policy)
