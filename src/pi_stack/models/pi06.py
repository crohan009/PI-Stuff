"""π*₀.₆ — VLA that learns from experience (Nov 2025).

First π model trained with reinforcement learning via RECAP
(advantage-conditioned policies). The architecture itself is close to π₀.₅
on the Gemma 3 backbone; the novelty is in the RL training loop, which
lives in `pi_stack.training.recap`.

Papers:
    papers/2025-11-17_pistar06_vla-learns-from-experience.pdf
    papers/2025-11-17_pistar06_model-card.pdf
"""

from __future__ import annotations

from dataclasses import dataclass

from pi_stack.models.backbones import GEMMA3_4B, BackboneSpec
from pi_stack.models.pi05 import Pi05Config, Pi05Policy


@dataclass
class Pi06Config(Pi05Config):
    backbone: BackboneSpec = GEMMA3_4B
    advantage_conditioning: bool = True
    value_distribution_bins: int = 51  # categorical/distributional value head


class Pi06Policy(Pi05Policy):
    """π*₀.₆ — Pi05 backbone swap + advantage-conditioning input channel.

    TODO:
      - add advantage-token input pathway (consumed by the action expert)
      - expose hooks for the RECAP trainer to inject advantage estimates
    """
