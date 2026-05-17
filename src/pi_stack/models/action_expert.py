"""Modular flow-matching action expert that attends to VLM activations.

Introduced in π₀; reused (and "knowledge-insulated" via stop-gradient) by KI;
extended for steerable conditioning in π₀.₇.

Sizes seen in the literature: 300M (π₀ small) → 860M (π₀.₆ / π₀.₇ large).
Action chunking horizon H ≈ 50 across the arc.

Paper refs:
    papers/2024-10-31_pi0_first-generalist-policy.pdf — §3 architecture
    papers/2025-05-28_ki_train-fast-run-fast-generalize-better.pdf
    papers/2026-04-16_pi07_steerable-model-emergent-capabilities.pdf

This module provides a real (small) implementation that satisfies the
interface ``v_θ(a_t, t, ctx)`` — usable with both the tiny in-repo backbone
and a real PaliGemma/Gemma3 VLM (the ``ctx_dim`` arg picks the right size).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class ActionExpertConfig:
    hidden_size: int = 1024
    num_layers: int = 16
    num_heads: int = 16
    action_dim: int = 14         # bimanual default; override per embodiment
    horizon: int = 50            # H — chunk length
    flow_steps: int = 5          # 5-10 denoising steps at inference
    cross_attend_to_vlm: bool = True
    time_embed_dim: int = 128


def _sinusoidal_time_embed(t: "Tensor", dim: int) -> "Tensor":
    """Sinusoidal positional encoding of the diffusion time t ∈ [0,1].

    Same as standard diffusion time embeddings — half sin, half cos over
    geometrically-spaced frequencies.
    """
    import torch

    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    args = t.unsqueeze(-1) * freqs.unsqueeze(0)   # (B, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class ActionExpert:
    """Flow-matching velocity transformer.

    Behaves like an ``nn.Module``: implement ``forward(a_t, t, ctx)`` returning
    a velocity prediction. ``sample_chunk(ctx, ...)`` runs the inference-time
    Euler integration (with optional RTC inpainting prefix).

    Architecture:
      - Project action ``a_t`` + time embedding to hidden_size.
      - ``num_layers`` of cross-attention to ``ctx`` (VLM features).
      - Project back to ``action_dim`` for velocity output.
    """

    def __init__(self, config: ActionExpertConfig, ctx_dim: int | None = None) -> None:
        import torch
        import torch.nn as nn

        self.config = config
        self.ctx_dim = ctx_dim or config.hidden_size

        class _Net(nn.Module):
            def __init__(self_inner) -> None:
                super().__init__()
                self_inner.action_in = nn.Linear(config.action_dim, config.hidden_size)
                self_inner.time_proj = nn.Linear(config.time_embed_dim, config.hidden_size)
                self_inner.ctx_proj = (
                    nn.Linear(self.ctx_dim, config.hidden_size)
                    if self.ctx_dim != config.hidden_size
                    else nn.Identity()
                )
                decoder_layer = nn.TransformerDecoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.num_heads,
                    dim_feedforward=config.hidden_size * 4,
                    batch_first=True,
                    activation="gelu",
                )
                self_inner.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)
                self_inner.action_out = nn.Linear(config.hidden_size, config.action_dim)

            def forward(self_inner, a_t: "Tensor", t: "Tensor", ctx: "Tensor") -> "Tensor":
                # (B, H, hidden)
                action_h = self_inner.action_in(a_t)
                # (B, time_embed)
                t_emb = _sinusoidal_time_embed(t, config.time_embed_dim)
                t_h = self_inner.time_proj(t_emb).unsqueeze(1)   # broadcast over H
                tgt = action_h + t_h
                memory = self_inner.ctx_proj(ctx)
                out = self_inner.decoder(tgt=tgt, memory=memory)
                return self_inner.action_out(out)

        self.net = _Net()

    def parameters(self):
        return self.net.parameters()

    def to(self, device) -> "ActionExpert":
        self.net.to(device)
        return self

    def __call__(self, a_t: "Tensor", t: "Tensor", ctx: "Tensor") -> "Tensor":
        return self.net(a_t, t, ctx)

    def velocity(self, a_t: "Tensor", t: "Tensor", ctx: "Tensor") -> "Tensor":
        return self.net(a_t, t, ctx)

    def sample_chunk(
        self,
        ctx: "Tensor",
        *,
        prefix: "Tensor | None" = None,
        flow_steps: int | None = None,
    ) -> "Tensor":
        """Sample a (B, horizon, action_dim) chunk via flow matching."""
        from pi_stack.training.flow_matching import euler_sample

        return euler_sample(
            self.net,
            ctx,
            horizon=self.config.horizon,
            action_dim=self.config.action_dim,
            flow_steps=flow_steps or self.config.flow_steps,
            prefix=prefix,
        )
