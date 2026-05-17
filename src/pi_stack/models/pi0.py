"""π₀ — first generalist VLA (Oct 2024).

PaliGemma 3B backbone + flow-matching action expert. Late-fusion transformer
that consumes images + language + proprioceptive state and emits a chunk of
H continuous actions at up to 50 Hz.

Paper: papers/2024-10-31_pi0_first-generalist-policy.pdf

This module assembles the policy:

  observations → [vision encoder] ⊕ [language encoder] ⊕ [state encoder]
              → late-fusion VLM
              → context features (B, T_ctx, F)
              → action expert (flow matching)
              → action chunk (B, H, D)

A ``TinyBackbone`` is used when tests / notebooks run locally; the real
PaliGemma backbone is swapped in by passing a ``backbone`` callable to the
constructor (see ``pi_stack.models.backbones.load_backbone``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pi_stack.models.action_expert import ActionExpert, ActionExpertConfig
from pi_stack.models.backbones import TINY, BackboneSpec


@dataclass
class Pi0Config:
    backbone: BackboneSpec = TINY
    action_expert: ActionExpertConfig = field(
        default_factory=lambda: ActionExpertConfig(
            hidden_size=128, num_layers=2, num_heads=4, action_dim=7, horizon=20, flow_steps=5,
        )
    )
    state_dim: int = 14
    image_resolution: int = 64       # 224 for real PaliGemma; 64 for tiny
    control_hz: int = 50


class Pi0Policy:
    """π₀ policy. Assembles backbone, state encoder, and action expert.

    Inputs to ``predict_chunk``:
      - ``images``      : (B, C, H_img, W_img) — single camera; multi-cam
        wrapping is a wrapper concern
      - ``state``       : (B, state_dim) proprioception
      - ``language_ids``: (B, T_lang) int64 token ids (FAST tokens or LM
        tokens, depending on caller)

    Output: ``(B, action_expert.horizon, action_expert.action_dim)``
    """

    def __init__(self, config: Pi0Config | None = None, *, backbone=None) -> None:
        import torch
        import torch.nn as nn

        from pi_stack.models.backbones import TinyBackbone, load_backbone

        self.config = config or Pi0Config()
        cfg = self.config

        # Backbone — provided externally (real PaliGemma) or via tiny default.
        if backbone is None:
            if cfg.backbone.name == "tiny":
                backbone = TinyBackbone(
                    hidden_size=cfg.backbone.hidden_size,
                    vocab_size=cfg.backbone.vocab_size,
                    image_resolution=cfg.image_resolution,
                )
            else:
                backbone = load_backbone(cfg.backbone)
        self.backbone = backbone
        self.hidden_size = getattr(backbone, "hidden_size", cfg.backbone.hidden_size)

        # State encoder — proprioception → hidden-size token.
        class _StateEnc(nn.Module):
            def __init__(self_inner) -> None:
                super().__init__()
                self_inner.proj = nn.Sequential(
                    nn.Linear(cfg.state_dim, self.hidden_size), nn.SiLU(),
                    nn.Linear(self.hidden_size, self.hidden_size),
                )

            def forward(self_inner, state):
                # (B, hidden) → (B, 1, hidden) — added as one extra context token
                return self_inner.proj(state).unsqueeze(1)

        self.state_encoder = _StateEnc()

        # Action expert with ctx_dim matching the backbone's hidden size.
        self.action_expert = ActionExpert(cfg.action_expert, ctx_dim=self.hidden_size)

    # --- Submodule helpers ------------------------------------------------

    def parameters(self):
        params = list(self.backbone.parameters()) + list(self.state_encoder.parameters()) + list(self.action_expert.parameters())
        return params

    def to(self, device) -> "Pi0Policy":
        self.backbone.to(device)
        self.state_encoder.to(device)
        self.action_expert.to(device)
        return self

    # --- Forward path -----------------------------------------------------

    def encode_context(self, images, state, language_ids):
        """Produce VLM context features ``(B, T_ctx, hidden)``.

        T_ctx layout: ``[image_tokens | language_tokens | state_token]``.
        """
        import torch

        _logits, vlm_features = self.backbone(language_ids, images=images)
        state_token = self.state_encoder(state)
        ctx = torch.cat([vlm_features, state_token], dim=1)
        return ctx

    def predict_chunk(
        self,
        images,
        state,
        language_ids,
        *,
        prefix=None,
        flow_steps=None,
    ):
        """Sample an action chunk via flow matching."""
        ctx = self.encode_context(images, state, language_ids)
        return self.action_expert.sample_chunk(ctx, prefix=prefix, flow_steps=flow_steps)

    def vlm_logits(self, images, language_ids):
        """Return the VLM's next-token logits — used by the KI discrete loss."""
        logits, _ = self.backbone(language_ids, images=images)
        return logits
