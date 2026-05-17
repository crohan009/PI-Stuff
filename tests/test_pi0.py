"""Tests for π₀ — backbone + state encoder + action expert assembly."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pi_stack.models.action_expert import ActionExpertConfig  # noqa: E402
from pi_stack.models.backbones import TINY  # noqa: E402
from pi_stack.models.pi0 import Pi0Config, Pi0Policy  # noqa: E402


def _tiny_config() -> Pi0Config:
    return Pi0Config(
        backbone=TINY,
        action_expert=ActionExpertConfig(
            hidden_size=64, num_layers=1, num_heads=4,
            action_dim=7, horizon=12, flow_steps=2, time_embed_dim=64,
        ),
        state_dim=10,
        image_resolution=32,
    )


def _sample_inputs(B: int, cfg: Pi0Config):
    return {
        "images": torch.randn(B, 3, cfg.image_resolution, cfg.image_resolution),
        "state": torch.randn(B, cfg.state_dim),
        "language_ids": torch.randint(0, TINY.vocab_size, (B, 6)),
    }


def test_pi0_predict_chunk_shape() -> None:
    cfg = _tiny_config()
    policy = Pi0Policy(cfg)
    out = policy.predict_chunk(**_sample_inputs(3, cfg))
    assert out.shape == (3, cfg.action_expert.horizon, cfg.action_expert.action_dim)
    assert out.dtype == torch.float32


def test_pi0_rtc_inpainting_prefix_held_fixed() -> None:
    cfg = _tiny_config()
    policy = Pi0Policy(cfg)
    inputs = _sample_inputs(2, cfg)
    P = 3
    prefix = torch.full((2, P, cfg.action_expert.action_dim), 0.42)
    chunk = policy.predict_chunk(**inputs, prefix=prefix)
    # The first P actions should match the prefix exactly (re-pinned each step).
    torch.testing.assert_close(chunk[:, :P, :], prefix)


def test_pi0_gradient_flow_through_vlm_and_expert() -> None:
    cfg = _tiny_config()
    policy = Pi0Policy(cfg)
    inputs = _sample_inputs(2, cfg)
    ctx = policy.encode_context(**inputs)
    # Drive a loss through the action expert.
    a_t = torch.randn(2, cfg.action_expert.horizon, cfg.action_expert.action_dim, requires_grad=True)
    t = torch.rand(2)
    v_pred = policy.action_expert(a_t, t, ctx)
    v_pred.sum().backward()
    # Backbone params should have gradient because we didn't insulate.
    backbone_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in policy.backbone.parameters()
    )
    expert_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in policy.action_expert.parameters()
    )
    assert backbone_has_grad
    assert expert_has_grad


def test_pi0_vlm_logits_match_backbone_vocab() -> None:
    cfg = _tiny_config()
    policy = Pi0Policy(cfg)
    logits = policy.vlm_logits(
        images=torch.randn(1, 3, cfg.image_resolution, cfg.image_resolution),
        language_ids=torch.randint(0, TINY.vocab_size, (1, 4)),
    )
    # T = image_patches + language_tokens; vocab matches backbone.
    assert logits.size(-1) == TINY.vocab_size
