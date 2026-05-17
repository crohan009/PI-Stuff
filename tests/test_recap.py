"""Tests for the RECAP trainer (π*₀.₆ recipe).

The three pieces to verify:
1. C51 categorical projection produces valid probability distributions.
2. Distributional value head can learn a known constant return.
3. AdvantageConditioner bucketizes + embeds correctly.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from pi_stack.training.recap import (  # noqa: E402
    AdvantageConditioner,
    DistributionalValueHead,
    RECAPConfig,
    RECAPTrainer,
)


def _make_value_head(feature_dim: int = 8) -> DistributionalValueHead:
    return DistributionalValueHead(
        feature_dim=feature_dim,
        config=RECAPConfig(
            advantage_bins=21,
            v_min=-5.0,
            v_max=5.0,
            advantage_token_embed_dim=16,
            target_update_period=50,
        ),
    )


def test_value_head_probs_are_valid_distribution() -> None:
    head = _make_value_head()
    feats = torch.randn(4, 8)
    p = head.probs(feats)
    assert p.shape == (4, 21)
    torch.testing.assert_close(p.sum(-1), torch.ones(4))
    assert (p >= 0).all()


def test_bellman_projection_preserves_probability_mass() -> None:
    head = _make_value_head()
    B = 6
    rewards = torch.linspace(-1.0, 1.0, B)
    next_feats = torch.randn(B, 8)
    dones = torch.zeros(B)
    target = head.bellman_projection(rewards, next_feats, dones)
    assert target.shape == (B, head.config.advantage_bins)
    # Each row must be a valid probability distribution.
    torch.testing.assert_close(target.sum(-1), torch.ones(B), atol=1e-5, rtol=0)
    assert (target >= 0).all()


def test_bellman_projection_handles_terminal() -> None:
    """For terminal states, target should concentrate near the reward."""
    head = _make_value_head()
    B = 4
    rewards = torch.tensor([2.0, -1.5, 0.0, 1.0])
    next_feats = torch.randn(B, 8)
    dones = torch.ones(B)
    target = head.bellman_projection(rewards, next_feats, dones)
    # E[z | target] should equal r exactly (within projection rounding).
    e_z = (target * head.support).sum(-1)
    torch.testing.assert_close(e_z, rewards, atol=head.delta_z / 2, rtol=0)


def test_value_head_learns_constant_target() -> None:
    """Single-state dummy task: target return is +1.0 for terminal transitions."""
    torch.manual_seed(0)
    head = _make_value_head()
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    feats = torch.randn(8, 8)
    rewards = torch.ones(8)
    next_feats = feats  # doesn't matter, terminal
    dones = torch.ones(8)
    for _ in range(200):
        target = head.bellman_projection(rewards, next_feats, dones)
        loss = head.cross_entropy_loss(feats, target)
        opt.zero_grad(); loss.backward(); opt.step()
    learned = head.value(feats).detach()
    assert (learned - 1.0).abs().mean() < 0.5, (
        f"value head didn't converge to 1.0; mean={learned.mean().item():.3f}"
    )


def test_advantage_conditioner_bucketizes_and_embeds() -> None:
    cfg = RECAPConfig(advantage_bins=10, advantage_token_embed_dim=8, a_min=-1.0, a_max=1.0)
    cond = AdvantageConditioner(cfg)
    advantages = torch.linspace(-2.0, 2.0, 6)   # spans below/above the range
    idx = cond.bucketize(advantages)
    assert idx.shape == (6,)
    assert idx.min() >= 0
    assert idx.max() < cfg.advantage_bins
    tok = cond.token(advantages)
    assert tok.shape == (6, cfg.advantage_token_embed_dim)


def test_advantage_conditioner_top_bucket_token() -> None:
    cfg = RECAPConfig(advantage_bins=10, advantage_token_embed_dim=8)
    cond = AdvantageConditioner(cfg)
    tok = cond.top_bucket_token(batch_size=3, device=torch.device("cpu"))
    assert tok.shape == (3, cfg.advantage_token_embed_dim)
    # All three rows should be identical (same top-bucket index).
    assert (tok[0] == tok[1]).all()
    assert (tok[1] == tok[2]).all()


def test_recap_trainer_step() -> None:
    """One alternating value + policy step runs end-to-end."""
    torch.manual_seed(0)
    head = _make_value_head()
    cond = AdvantageConditioner(head.config)

    # Tiny policy stand-in: a linear layer over the conditioning token.
    policy_proj = nn.Linear(head.config.advantage_token_embed_dim, 7)

    def policy_loss_fn(batch: dict, adv_token: torch.Tensor) -> torch.Tensor:
        pred = policy_proj(adv_token)
        return ((pred - batch["target_action"]) ** 2).mean()

    # The trainer's policy_opt only optimizes the conditioner; for this test
    # we just want the step to run cleanly with the policy plugged in.
    trainer = RECAPTrainer(head, cond, policy_loss_fn)
    B = 8
    batch = {
        "state_features": torch.randn(B, 8),
        "reward": torch.randn(B),
        "next_state_features": torch.randn(B, 8),
        "done": torch.zeros(B),
        "return_to_go": torch.randn(B),
        "target_action": torch.randn(B, 7),
    }
    metrics_v = trainer.update_value(batch)
    metrics_p = trainer.update_policy(batch)
    assert "value_loss" in metrics_v
    assert "policy_loss" in metrics_p
    assert "mean_advantage" in metrics_p
