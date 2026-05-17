"""Tests for the Knowledge Insulation (KI) primitives.

The key property to verify: with `insulate()` applied at the VLM↔expert
interface, the action-expert loss must NOT produce gradients on the VLM
parameters. The discrete-token loss must still update the VLM normally.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pi_stack.training.ki import (  # noqa: E402
    KIConfig,
    flow_matching_target,
    insulate,
    ki_loss,
)


def _tiny_vlm(vocab: int = 16, feat: int = 8) -> torch.nn.Module:
    """Stand-in for a VLM: token-emb + linear -> (logits, features)."""

    class TinyVLM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(vocab, feat)
            self.head = torch.nn.Linear(feat, vocab)

        def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            h = self.embed(ids)
            return self.head(h), h

    return TinyVLM()


def _tiny_expert(feat: int = 8, action_dim: int = 4) -> torch.nn.Module:
    """Stand-in for the flow-matching action expert."""

    class TinyExpert(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(feat + action_dim + 1, action_dim)

        def forward(
            self, a_t: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor
        ) -> torch.Tensor:
            # ctx: (B, T, F) — pool over time
            ctx_pooled = ctx.mean(dim=1, keepdim=True).expand(-1, a_t.size(1), -1)
            t_feat = t.view(-1, 1, 1).expand(-1, a_t.size(1), 1)
            return self.proj(torch.cat([a_t, t_feat, ctx_pooled], dim=-1))

    return TinyExpert()


def test_insulate_blocks_gradient() -> None:
    x = torch.randn(2, 4, requires_grad=True)
    y = insulate(x, enabled=True)
    # The whole point: y is severed from x in the autograd graph.
    assert not y.requires_grad
    assert y.grad_fn is None
    # Any downstream loss has no path back to x.
    out = (y * 2).sum()
    assert not out.requires_grad


def test_insulate_disabled_passes_gradient() -> None:
    x = torch.randn(2, 4, requires_grad=True)
    y = insulate(x, enabled=False)
    assert y.requires_grad
    (y * 2).sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_flow_matching_target_shapes() -> None:
    actions = torch.randn(3, 50, 7)
    noise = torch.randn_like(actions)
    t = torch.rand(3)
    a_t, target = flow_matching_target(actions, noise, t)
    assert a_t.shape == actions.shape
    assert target.shape == actions.shape
    # At t=0, a_t should equal noise; at t=1, a_t should equal actions.
    a_t0, _ = flow_matching_target(actions, noise, torch.zeros(3))
    a_t1, _ = flow_matching_target(actions, noise, torch.ones(3))
    assert torch.allclose(a_t0, noise)
    assert torch.allclose(a_t1, actions)


def test_ki_loss_returns_components() -> None:
    B, T, V, H, D = 2, 10, 16, 12, 4
    logits = torch.randn(B, T, V, requires_grad=True)
    targets = torch.randint(0, V, (B, T))
    pred = torch.randn(B, H, D, requires_grad=True)
    target = torch.randn(B, H, D)

    out = ki_loss(logits, targets, pred, target, KIConfig())
    assert set(out.keys()) == {"loss", "discrete", "continuous"}
    assert out["loss"].requires_grad
    assert not out["discrete"].requires_grad   # detached for logging
    assert not out["continuous"].requires_grad


def test_end_to_end_ki_blocks_expert_gradient_from_vlm() -> None:
    """The full KI flow: backbone weights must NOT receive gradient from
    the action-expert loss, even though they're connected by autograd."""
    torch.manual_seed(0)
    B, T, V, H, D = 4, 6, 16, 8, 4
    feat = 8

    vlm = _tiny_vlm(vocab=V, feat=feat)
    expert = _tiny_expert(feat=feat, action_dim=D)

    ids = torch.randint(0, V, (B, T))
    actions = torch.randn(B, H, D)
    noise = torch.randn_like(actions)
    t = torch.rand(B)

    # Forward — KI insulation in place.
    logits, features = vlm(ids)
    ctx = insulate(features, enabled=True)
    a_t, target = flow_matching_target(actions, noise, t)
    v_pred = expert(a_t, t, ctx)

    # Action-expert-only loss (no discrete term).
    expert_only = torch.nn.functional.mse_loss(v_pred, target)
    expert_only.backward()

    # VLM params must have NO gradient.
    for name, p in vlm.named_parameters():
        assert p.grad is None or p.grad.abs().sum() == 0, (
            f"VLM param {name} received gradient through the action expert "
            f"despite KI insulation"
        )
    # Expert params must have gradient.
    grad_norms = [p.grad.abs().sum().item() for p in expert.parameters() if p.grad is not None]
    assert any(g > 0 for g in grad_norms), "expert got no gradient"


def test_discrete_loss_still_updates_vlm() -> None:
    """KI must not break the VLM's own training signal."""
    torch.manual_seed(0)
    B, T, V = 4, 6, 16
    vlm = _tiny_vlm(vocab=V, feat=8)
    ids = torch.randint(0, V, (B, T))
    targets = torch.randint(0, V, (B, T))

    logits, _ = vlm(ids)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, V), targets.reshape(-1)
    )
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in vlm.parameters())
