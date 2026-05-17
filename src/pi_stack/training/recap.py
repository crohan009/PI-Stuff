"""RECAP — Reinforcement learning with advantage-conditioned policies (Nov 2025).

The π*₀.₆ training recipe. Sidesteps the instability of vanilla policy
gradient on giant VLAs by:

1. Learning a **distributional value function** ``V_φ`` (C51-style
   categorical over return bins) — stable supervised training, doesn't
   require backprop through the policy.
2. Computing **advantages** ``A = G_t - V_φ(s_t)`` from on-policy returns.
3. **Conditioning the policy on bucketized advantages** — the action expert
   receives an extra "advantage token" that signals "this trajectory was
   above/below average for its state."
4. **Behavior cloning on experience**: train the policy to imitate the
   actions it took, conditioned on their advantage tokens. At deployment,
   feed a high advantage token to "ask" the model for above-average
   behavior.

The whole loop only requires *one* off-policy ingredient — the value
function. Policy updates remain BC-style and therefore stable even on
3B+ parameter VLAs.

Paper: papers/2025-11-17_pistar06_vla-learns-from-experience.pdf
Model card: papers/2025-11-17_pistar06_model-card.pdf

This module provides three composable pieces:

- :class:`DistributionalValueHead` — C51 categorical value with the
  Bellman projection.
- :class:`AdvantageConditioner` — bucketize a continuous advantage and
  embed it as a token to feed the policy.
- :class:`RECAPTrainer` — alternates value updates and advantage-
  conditioned BC updates.

The policy itself is whatever you want — RECAP is recipe, not architecture.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import Tensor


@dataclass
class RECAPConfig:
    discount: float = 0.99
    advantage_bins: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    advantage_token_embed_dim: int = 64
    target_update_period: int = 500
    value_lr: float = 3e-4
    policy_lr: float = 1e-4

    # Range used to bucketize advantages into the conditioning token.
    # Wider than v_min/v_max because advantages are returns minus baseline.
    a_min: float = -5.0
    a_max: float = 5.0


# --- Distributional value head ------------------------------------------


class DistributionalValueHead:
    """C51 categorical value function with the Bellman projection.

    Models ``p(z | s)`` over ``n_bins`` evenly-spaced return support points
    in ``[v_min, v_max]``. Expected value is ``E[z] = Σ p_i · z_i``.
    """

    def __init__(self, feature_dim: int, config: RECAPConfig | None = None) -> None:
        import torch
        import torch.nn as nn

        self.config = config or RECAPConfig()
        cfg = self.config
        self.support = torch.linspace(cfg.v_min, cfg.v_max, cfg.advantage_bins)
        self.delta_z = (cfg.v_max - cfg.v_min) / (cfg.advantage_bins - 1)

        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, cfg.advantage_bins),
        )
        self.target_net = copy.deepcopy(self.net).requires_grad_(False)

    def to(self, device: "torch.device | str") -> "DistributionalValueHead":
        self.net.to(device)
        self.target_net.to(device)
        self.support = self.support.to(device)
        return self

    def parameters(self):
        return self.net.parameters()

    def probs(self, features: "Tensor") -> "Tensor":
        """``(B, n_bins)`` probability mass over the return support."""
        import torch
        return torch.softmax(self.net(features), dim=-1)

    def value(self, features: "Tensor") -> "Tensor":
        """Scalar expected value ``(B,)``."""
        probs = self.probs(features)
        return (probs * self.support).sum(-1)

    def target_value(self, features: "Tensor") -> "Tensor":
        import torch
        with torch.no_grad():
            probs = torch.softmax(self.target_net(features), dim=-1)
        return (probs * self.support).sum(-1)

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.net.state_dict())

    def bellman_projection(
        self,
        rewards: "Tensor",
        next_features: "Tensor",
        dones: "Tensor",
    ) -> "Tensor":
        """Project ``r + γ·z'`` onto the categorical support (C51 Algorithm 1).

        Args:
            rewards: ``(B,)``
            next_features: ``(B, F)``
            dones: ``(B,)`` float in {0, 1}

        Returns:
            ``(B, n_bins)`` target probability distribution to regress against
            with cross-entropy.
        """
        import torch

        B = rewards.size(0)
        n_bins = self.config.advantage_bins
        v_min, v_max = self.config.v_min, self.config.v_max
        discount = self.config.discount

        with torch.no_grad():
            # Target distribution over next-state return.
            next_probs = torch.softmax(self.target_net(next_features), dim=-1)   # (B, n_bins)
            # Project support: tz = r + γ·z (zero out next-state for terminal).
            tz = rewards.unsqueeze(-1) + (1.0 - dones).unsqueeze(-1) * discount * self.support.unsqueeze(0)
            tz = tz.clamp(v_min, v_max)
            # Map onto bin indices.
            b = (tz - v_min) / self.delta_z   # fractional bin index
            lower = b.floor().long()
            upper = b.ceil().long()
            # Handle corner: when lower==upper, mass goes to lower; nudge upper to keep gradient flowing.
            lower = lower.clamp(0, n_bins - 1)
            upper = upper.clamp(0, n_bins - 1)

            target = torch.zeros_like(next_probs)
            # Distribute mass between lower and upper proportionally.
            lower_mass = next_probs * (upper.float() - b)
            upper_mass = next_probs * (b - lower.float())
            # When b is exactly on a bin, lower_mass + upper_mass = 0 → put all mass in lower.
            on_bin = (lower == upper)
            lower_mass = torch.where(on_bin, next_probs, lower_mass)

            # Use scatter_add to accumulate masses into target distribution.
            target.scatter_add_(1, lower, lower_mass)
            target.scatter_add_(1, upper, upper_mass)

        return target

    def cross_entropy_loss(
        self,
        features: "Tensor",
        target_probs: "Tensor",
    ) -> "Tensor":
        """KL-divergence-equivalent loss against the projected target."""
        import torch
        import torch.nn.functional as F

        log_probs = F.log_softmax(self.net(features), dim=-1)
        return -(target_probs * log_probs).sum(-1).mean()


# --- Advantage conditioner ----------------------------------------------


class AdvantageConditioner:
    """Bucketize continuous advantages and embed them as a learnable token.

    At training time we feed the *empirical* bucket (where this trajectory's
    advantage landed); at deployment we feed the top bucket to ask for
    above-average behavior. The bucket boundaries are linearly spaced in
    ``[a_min, a_max]``.
    """

    def __init__(self, config: RECAPConfig | None = None) -> None:
        import torch
        import torch.nn as nn

        self.config = config or RECAPConfig()
        cfg = self.config
        self.embed = nn.Embedding(cfg.advantage_bins, cfg.advantage_token_embed_dim)
        self.boundaries = torch.linspace(cfg.a_min, cfg.a_max, cfg.advantage_bins + 1)

    def to(self, device: "torch.device | str") -> "AdvantageConditioner":
        self.embed.to(device)
        self.boundaries = self.boundaries.to(device)
        return self

    def parameters(self):
        return self.embed.parameters()

    def bucketize(self, advantages: "Tensor") -> "Tensor":
        """Map advantages to integer bin indices ``(B,)`` int64."""
        import torch
        # bucketize returns indices in 0..n_boundaries; clamp to valid bin range.
        idx = torch.bucketize(advantages, self.boundaries) - 1
        return idx.clamp(0, self.config.advantage_bins - 1)

    def token(self, advantages: "Tensor") -> "Tensor":
        """Embed advantages as ``(B, embed_dim)`` conditioning tokens."""
        return self.embed(self.bucketize(advantages))

    def top_bucket_token(self, batch_size: int, device: "torch.device | str") -> "Tensor":
        """Conditioning token for "above-average behavior" — used at deployment."""
        import torch
        idx = torch.full(
            (batch_size,),
            self.config.advantage_bins - 1,
            dtype=torch.long,
            device=device,
        )
        return self.embed(idx)


# --- RECAP trainer ------------------------------------------------------


class RECAPTrainer:
    """Alternating value-learning + advantage-conditioned BC.

    The policy is whatever you pass in — RECAP only requires that it accepts
    an advantage-token kwarg. Behavior-cloning loss is the policy's own
    log-likelihood under the actions it took.

    Callable contract for `policy_loss_fn`:
        ``policy_loss_fn(batch, advantage_token) -> scalar Tensor``
    """

    def __init__(
        self,
        value_head: DistributionalValueHead,
        conditioner: AdvantageConditioner,
        policy_loss_fn,
        config: RECAPConfig | None = None,
    ) -> None:
        import torch

        self.config = config or value_head.config
        self.value_head = value_head
        self.conditioner = conditioner
        self.policy_loss_fn = policy_loss_fn
        self._steps = 0

        self.value_opt = torch.optim.Adam(
            list(value_head.parameters()), lr=self.config.value_lr
        )
        self.policy_opt = torch.optim.Adam(
            list(conditioner.parameters()), lr=self.config.policy_lr
        )

    def update_value(self, batch: dict) -> dict[str, float]:
        """One step of distributional value learning.

        Batch keys: ``state_features``, ``reward``, ``next_state_features``,
        ``done``.
        """
        target_probs = self.value_head.bellman_projection(
            batch["reward"], batch["next_state_features"], batch["done"]
        )
        loss = self.value_head.cross_entropy_loss(batch["state_features"], target_probs)
        self.value_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.value_opt.step()
        self._steps += 1
        if self._steps % self.config.target_update_period == 0:
            self.value_head.sync_target()
        return {"value_loss": float(loss.detach())}

    def update_policy(self, batch: dict) -> dict[str, float]:
        """One advantage-conditioned BC step.

        Batch keys: ``state_features``, ``return_to_go``, plus whatever the
        policy_loss_fn needs. RECAPTrainer computes the advantage and the
        token, then hands both off to the user's loss function.
        """
        import torch

        with torch.no_grad():
            baseline = self.value_head.value(batch["state_features"])
            advantage = batch["return_to_go"] - baseline

        adv_token = self.conditioner.token(advantage)
        loss = self.policy_loss_fn(batch, adv_token)
        self.policy_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.policy_opt.step()
        return {
            "policy_loss": float(loss.detach()),
            "mean_advantage": float(advantage.mean().detach()),
        }
