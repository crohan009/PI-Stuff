"""π*₀.₆ — VLA that learns from experience (Nov 2025).

First π model trained with reinforcement learning via RECAP
(advantage-conditioned policies). The architecture itself is close to π₀.₅
on the Gemma 3 backbone; the novelty is in the RL training loop, which
lives in :mod:`pi_stack.training.recap`.

Papers:
    papers/2025-11-17_pistar06_vla-learns-from-experience.pdf
    papers/2025-11-17_pistar06_model-card.pdf

Adds the **advantage-token input pathway** to the action expert's context.
At training time RECAP injects an advantage token (bucketized return minus
baseline); at deployment the caller injects the top-bucket token to ask
for above-average behavior. The mechanics live in
``pi_stack.training.recap.AdvantageConditioner``; here we just provide the
hook to consume the token.
"""

from __future__ import annotations

from dataclasses import dataclass

from pi_stack.models.backbones import GEMMA3_4B, TINY, BackboneSpec
from pi_stack.models.pi05 import Pi05Config, Pi05Policy


@dataclass
class Pi06Config(Pi05Config):
    backbone: BackboneSpec = TINY            # swap to GEMMA3_4B for real runs
    advantage_conditioning: bool = True
    advantage_token_dim: int = 64
    value_distribution_bins: int = 51        # informational — RECAP owns this


class Pi06Policy(Pi05Policy):
    """π*₀.₆ — adds an advantage-conditioning input pathway.

    The advantage token is consumed as one extra context token prepended to
    the VLM features fed into the action expert. RECAP's
    ``AdvantageConditioner`` produces the token; this class just routes it.
    """

    def __init__(self, config: Pi06Config | None = None, *, backbone=None) -> None:
        import torch.nn as nn

        super().__init__(config or Pi06Config(), backbone=backbone)
        cfg: Pi06Config = self.config

        # Project the advantage token (from RECAP) into the VLM hidden size
        # so it can be concatenated as a single context token.
        self.advantage_proj = (
            nn.Linear(cfg.advantage_token_dim, self.hidden_size)
            if cfg.advantage_conditioning
            else None
        )

    def parameters(self):
        params = super().parameters()
        if self.advantage_proj is not None:
            params += list(self.advantage_proj.parameters())
        return params

    def to(self, device) -> "Pi06Policy":
        super().to(device)
        if self.advantage_proj is not None:
            self.advantage_proj.to(device)
        return self

    # --- Forward path ----------------------------------------------------

    def encode_context(self, images, state, language_ids, *, advantage_token=None):
        """Prepend the advantage token (if provided) to the π₀.₅ context."""
        import torch

        ctx = super().encode_context(images, state, language_ids)
        if advantage_token is not None and self.advantage_proj is not None:
            adv = self.advantage_proj(advantage_token).unsqueeze(1)   # (B, 1, hidden)
            ctx = torch.cat([adv, ctx], dim=1)
        return ctx

    def predict_chunk(
        self,
        images,
        state,
        language_ids,
        *,
        advantage_token=None,
        prefix=None,
        flow_steps=None,
    ):
        ctx = self.encode_context(images, state, language_ids, advantage_token=advantage_token)
        return self.action_expert.sample_chunk(ctx, prefix=prefix, flow_steps=flow_steps)
