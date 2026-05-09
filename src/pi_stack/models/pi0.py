"""π₀ — first generalist VLA (Oct 2024).

PaliGemma 3B backbone + flow-matching action expert. Late-fusion transformer
that consumes images + language + proprioceptive state and emits a chunk of
H continuous actions at up to 50 Hz.

Paper: papers/2024-10-31_pi0_first-generalist-policy.pdf
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pi_stack.models.action_expert import ActionExpertConfig
from pi_stack.models.backbones import PALIGEMMA_3B, BackboneSpec


@dataclass
class Pi0Config:
    backbone: BackboneSpec = PALIGEMMA_3B
    action_expert: ActionExpertConfig = field(default_factory=ActionExpertConfig)
    state_dim: int = 14
    image_resolution: int = 224
    control_hz: int = 50


class Pi0Policy:
    """Skeleton π₀ policy.

    TODO:
      - load backbone + processor (see `models.backbones.load_backbone`)
      - inject state encoder
      - wire action expert in late-fusion mode
      - implement `predict_chunk(obs, language)` returning (H, action_dim)
    """

    def __init__(self, config: Pi0Config) -> None:
        self.config = config

    def predict_chunk(self, *args, **kwargs):
        raise NotImplementedError
