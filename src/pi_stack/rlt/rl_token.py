"""RL Token (RLT) — precise manipulation via online RL (Mar 2026).

Frozen VLA + tiny actor-critic MLP attached to a single compressed
"RL token" extracted from the VLA's internal features. Refines the last
millimeter of contact-rich tasks (screw installation, charger insertion)
in a few hours of robot time. Paper reports 20% → 65% success rates on
hard insertion tasks with this method alone.

Paper: papers/2026-03-19_rlt_precise-manipulation-online-rl.pdf
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RLTConfig:
    rl_token_dim: int = 256
    actor_hidden: int = 256
    critic_hidden: int = 256
    learning_rate: float = 3e-4
    freeze_vla: bool = True


class RLTHead:
    """Tiny actor-critic head riding on top of a frozen VLA.

    TODO:
      - extract RL token from a chosen VLA layer (config-driven)
      - implement actor MLP -> action residual / Δa applied to VLA output
      - implement critic MLP -> Q(s, a)
      - online RL update (SAC-style is the natural choice on continuous Δa)
    """

    def __init__(self, config: RLTConfig) -> None:
        self.config = config

    def act(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError
