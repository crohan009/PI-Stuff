"""Tests for the RLT actor-critic head + minimal SAC trainer.

The head sits on a frozen VLA's "RL token" output and produces small
action residuals. We verify:

- `act()` returns finite action residuals bounded by the configured scale.
- One SAC update step runs without NaNs and produces sane gradient norms.
- Target networks track online networks via Polyak averaging.
- The trainer can drive a critic to fit a known fixed-target tabular task.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import numpy as np  # noqa: E402

from pi_stack.rlt.rl_token import (  # noqa: E402
    ReplayBuffer,
    RLTConfig,
    RLTHead,
    RLTTrainer,
)


def _make_head(action_dim: int = 4) -> RLTHead:
    return RLTHead(
        RLTConfig(
            rl_token_dim=8,
            state_dim=4,
            action_dim=action_dim,
            actor_hidden=32,
            critic_hidden=32,
            action_residual_scale=0.05,
        )
    )


def test_act_returns_bounded_residual() -> None:
    head = _make_head()
    rl_token = torch.randn(7, head.config.rl_token_dim)
    state = torch.randn(7, head.config.state_dim)
    action, log_prob = head.act(rl_token, state)
    assert action.shape == (7, head.config.action_dim)
    assert log_prob.shape == (7,)
    # tanh-squashed and scaled — bounded by action_residual_scale.
    assert action.abs().max().item() <= head.config.action_residual_scale + 1e-6


def test_deterministic_act_is_stable_across_calls() -> None:
    head = _make_head()
    rl_token = torch.randn(3, head.config.rl_token_dim)
    state = torch.randn(3, head.config.state_dim)
    a1, _ = head.act(rl_token, state, deterministic=True)
    a2, _ = head.act(rl_token, state, deterministic=True)
    torch.testing.assert_close(a1, a2)


def test_sac_step_runs_without_nans() -> None:
    torch.manual_seed(0)
    head = _make_head()
    trainer = RLTTrainer(head)
    B = 16
    batch = {
        "rl_token": torch.randn(B, head.config.rl_token_dim),
        "state": torch.randn(B, head.config.state_dim),
        "action": torch.empty(B, head.config.action_dim).uniform_(-0.05, 0.05),
        "reward": torch.randn(B),
        "next_rl_token": torch.randn(B, head.config.rl_token_dim),
        "next_state": torch.randn(B, head.config.state_dim),
        "done": torch.zeros(B),
    }
    metrics = trainer.update(batch)
    for k, v in metrics.items():
        assert np.isfinite(v), f"NaN in metric {k}: {v}"


def test_target_networks_polyak_track_online_after_many_updates() -> None:
    torch.manual_seed(0)
    head = _make_head()
    trainer = RLTTrainer(head)
    B = 8
    batch = {
        "rl_token": torch.randn(B, head.config.rl_token_dim),
        "state": torch.randn(B, head.config.state_dim),
        "action": torch.empty(B, head.config.action_dim).uniform_(-0.05, 0.05),
        "reward": torch.randn(B),
        "next_rl_token": torch.randn(B, head.config.rl_token_dim),
        "next_state": torch.randn(B, head.config.state_dim),
        "done": torch.zeros(B),
    }
    # Many updates — target should pull strongly toward online net.
    for _ in range(200):
        trainer.update(batch)
    diffs = []
    for p, p_t in zip(head.q1.parameters(), head.q1_target.parameters()):
        diffs.append((p - p_t).abs().mean().item())
    # Target should be close, but not identical (tau < 1).
    assert max(diffs) < 0.5
    assert max(diffs) > 0


def test_critic_can_fit_constant_target() -> None:
    """Drive the critic toward a *constant* Q target and verify it converges.

    Sanity check that the SAC update mechanics actually decrease MSE on a
    trivial supervised problem (constant 1.0 reward, terminal episodes →
    Q* = 1.0 everywhere).
    """
    torch.manual_seed(0)
    head = _make_head()
    trainer = RLTTrainer(head)
    B = 32
    rl = torch.randn(B, head.config.rl_token_dim)
    s = torch.randn(B, head.config.state_dim)
    a = torch.empty(B, head.config.action_dim).uniform_(-0.05, 0.05)
    batch = {
        "rl_token": rl,
        "state": s,
        "action": a,
        "reward": torch.ones(B),
        "next_rl_token": rl,
        "next_state": s,
        "done": torch.ones(B),     # terminal → next-state Q ignored
    }
    initial = head.q1(rl, s, a).detach()
    for _ in range(300):
        trainer.update(batch)
    final = head.q1(rl, s, a).detach()
    # Q* = reward = 1 because done=1 zeroes out bootstrap.
    assert (final - 1.0).abs().mean() < (initial - 1.0).abs().mean()
    assert (final - 1.0).abs().mean() < 0.3, (
        f"critic didn't approach target Q=1; mean error {(final - 1.0).abs().mean().item():.3f}"
    )


def test_replay_buffer_roundtrip() -> None:
    buf = ReplayBuffer(capacity=10, rl_token_dim=4, state_dim=3, action_dim=2)
    for i in range(15):
        buf.add(
            rl_token=torch.full((4,), float(i)),
            state=torch.zeros(3),
            action=torch.zeros(2),
            reward=torch.tensor(float(i)),
            next_rl_token=torch.zeros(4),
            next_state=torch.zeros(3),
            done=False,
        )
    assert len(buf) == 10   # capacity-bounded
    sample = buf.sample(32)
    assert sample["rl_token"].shape == (32, 4)
    assert sample["reward"].shape == (32,)
