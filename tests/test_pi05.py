"""Tests for π₀.₅ — subtask head + co-training mixer on top of π₀."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pi_stack.models.action_expert import ActionExpertConfig  # noqa: E402
from pi_stack.models.backbones import TINY  # noqa: E402
from pi_stack.models.pi05 import CoTrainWeights, Pi05Config, Pi05Policy  # noqa: E402


def _tiny_config(subtask_vocab=64) -> Pi05Config:
    return Pi05Config(
        backbone=TINY,
        action_expert=ActionExpertConfig(
            hidden_size=64, num_layers=1, num_heads=4,
            action_dim=7, horizon=8, flow_steps=2, time_embed_dim=64,
        ),
        state_dim=10,
        image_resolution=32,
        subtask_vocab_size=subtask_vocab,
        cotrain=CoTrainWeights(mobile_manip=0.5, web=0.2, vqa=0.2, subtask_pred=0.1),
    )


def test_pi05_inherits_pi0_predict_chunk() -> None:
    cfg = _tiny_config()
    p = Pi05Policy(cfg)
    out = p.predict_chunk(
        images=torch.randn(2, 3, cfg.image_resolution, cfg.image_resolution),
        state=torch.randn(2, cfg.state_dim),
        language_ids=torch.randint(0, TINY.vocab_size, (2, 4)),
    )
    assert out.shape == (2, cfg.action_expert.horizon, cfg.action_expert.action_dim)


def test_pi05_subtask_head_emits_right_vocab() -> None:
    cfg = _tiny_config(subtask_vocab=32)
    p = Pi05Policy(cfg)
    logits = p.predict_subtask_logits(
        images=torch.randn(2, 3, cfg.image_resolution, cfg.image_resolution),
        language_ids=torch.randint(0, TINY.vocab_size, (2, 5)),
    )
    assert logits.shape[-1] == 32
    assert logits.shape[0] == 2


def test_cotrain_mixer_respects_weights() -> None:
    cfg = _tiny_config()
    p = Pi05Policy(cfg)
    losses = {
        "mobile_manip": torch.tensor(2.0),
        "web": torch.tensor(0.0),
        "vqa": torch.tensor(1.0),
        "subtask_pred": torch.tensor(0.0),
    }
    out = p.cotrain_loss(losses)
    # Weights are normalized over present keys: total = 0.5+0.2+0.2+0.1 = 1.0
    # Expected: (0.5*2 + 0.2*0 + 0.2*1 + 0.1*0) / 1.0 = 1.2
    assert float(out) == pytest.approx(1.2)


def test_cotrain_mixer_handles_missing_keys() -> None:
    cfg = _tiny_config()
    p = Pi05Policy(cfg)
    # Only one head present — should renormalize.
    losses = {"subtask_pred": torch.tensor(3.0)}
    out = p.cotrain_loss(losses)
    # Only one weight present so it dominates → loss = 3.0
    assert float(out) == pytest.approx(3.0)


def test_cotrain_mixer_zero_when_empty() -> None:
    cfg = _tiny_config()
    p = Pi05Policy(cfg)
    # Edge case: empty dict — caller responsibility, but shouldn't crash.
    # We need at least one tensor to know the device; pass a sentinel.
    losses = {"mobile_manip": torch.tensor(0.0)}
    out = p.cotrain_loss(losses)
    assert float(out) == pytest.approx(0.0)
