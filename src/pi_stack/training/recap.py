"""RECAP — Reinforcement learning with advantage-conditioned policies (Nov 2025).

Trains the VLA on its own deployment experience by conditioning action
prediction on advantage estimates from a distributional value function.
This sidesteps the instability of vanilla policy-gradient on giant VLAs
while still allowing the policy to improve beyond demonstration quality.

Paper: papers/2025-11-17_pistar06_vla-learns-from-experience.pdf
Model card: papers/2025-11-17_pistar06_model-card.pdf
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RECAPConfig:
    discount: float = 0.99
    advantage_bins: int = 51            # categorical/distributional head
    advantage_token_embed_dim: int = 64
    target_update_period: int = 500


class RECAPTrainer:
    """Skeleton RECAP training loop.

    TODO:
      - distributional value head V_φ(s) returning a categorical over returns
      - compute advantage = Q_φ(s,a) - V_φ(s) (or n-step variant)
      - bin advantage and embed as a conditioning token for the policy
      - alternate value updates and advantage-conditioned BC-on-experience
    """

    def __init__(self, config: RECAPConfig) -> None:
        self.config = config

    def update_value(self, *args, **kwargs):
        raise NotImplementedError

    def update_policy(self, *args, **kwargs):
        raise NotImplementedError
