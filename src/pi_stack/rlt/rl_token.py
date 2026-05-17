"""RL Token (RLT) — precise manipulation via online RL (Mar 2026).

Frozen VLA + tiny actor-critic MLP attached to a single compressed
"RL token" extracted from the VLA's internal features. Refines the last
millimeter of contact-rich tasks (screw installation, charger insertion)
in a few hours of robot time. Paper reports 20% → 65% success rates on
hard insertion tasks with this method alone.

Paper: papers/2026-03-19_rlt_precise-manipulation-online-rl.pdf

The recipe is **SAC** on a tiny actor-critic head whose only input is the
RL token + low-D state. SAC was chosen because:

- Continuous-action insertion has no obvious discrete-action structure.
- Off-policy means we can refine without re-collecting demos.
- Entropy regularization handles the exploration vs. precision trade-off.

This module ships a minimal SAC implementation (~150 LOC). It's intentionally
self-contained — no `stable-baselines3` dep — so we can hook it directly
into any policy's RL-token pathway. The frozen VLA's job is to expose
``(rl_token, base_action)``; the RLT head emits a residual action
``Δa`` that's added to the base.

Wire-up::

    head = RLTHead(action_dim=7, state_dim=14, rl_token_dim=256)
    trainer = RLTTrainer(head)
    # Per env step:
    rl_token, base_action = vla.extract(obs)
    delta_a, log_prob = head.act(rl_token, state)
    action = base_action + delta_a
    # ...env.step, store in buffer...
    trainer.update(batch)
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import Tensor


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


@dataclass
class RLTConfig:
    rl_token_dim: int = 256
    state_dim: int = 14
    action_dim: int = 7
    actor_hidden: int = 256
    critic_hidden: int = 256

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4

    discount: float = 0.99
    tau: float = 0.005           # target net Polyak smoothing
    init_alpha: float = 0.2       # initial entropy temperature
    autotune_alpha: bool = True
    target_entropy: float | None = None   # default: -action_dim

    action_residual_scale: float = 0.05   # cap on |Δa| — paper uses small residuals
    freeze_vla: bool = True       # informational; RLT only sees rl_token
    device: str = "cpu"


# --- Networks ------------------------------------------------------------


def _make_mlp(in_dim: int, out_dim: int, hidden: int):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.SiLU(),
        nn.Linear(hidden, hidden), nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


def _build_actor(cfg: RLTConfig):
    import torch.nn as nn

    class Actor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(cfg.rl_token_dim + cfg.state_dim, cfg.actor_hidden), nn.SiLU(),
                nn.Linear(cfg.actor_hidden, cfg.actor_hidden), nn.SiLU(),
            )
            self.mu_head = nn.Linear(cfg.actor_hidden, cfg.action_dim)
            self.log_std_head = nn.Linear(cfg.actor_hidden, cfg.action_dim)

        def forward(self, rl_token, state):
            import torch
            h = self.trunk(torch.cat([rl_token, state], dim=-1))
            mu = self.mu_head(h)
            log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
            return mu, log_std

    return Actor()


def _build_critic(cfg: RLTConfig):
    import torch.nn as nn

    class Critic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q = _make_mlp(
                cfg.rl_token_dim + cfg.state_dim + cfg.action_dim,
                1,
                cfg.critic_hidden,
            )

        def forward(self, rl_token, state, action):
            import torch
            return self.q(torch.cat([rl_token, state, action], dim=-1)).squeeze(-1)

    return Critic()


# --- The head -----------------------------------------------------------


class RLTHead:
    """Actor + twin critics + target critics. All small MLPs.

    The reparameterized squashed-Gaussian policy is standard SAC. Actions
    are tanh-squashed and then scaled by ``action_residual_scale`` so the
    head produces *small* corrections on top of the frozen VLA's output.
    """

    def __init__(self, config: RLTConfig | None = None) -> None:
        import torch

        self.config = config or RLTConfig()
        device = torch.device(self.config.device)
        self.actor = _build_actor(self.config).to(device)
        self.q1 = _build_critic(self.config).to(device)
        self.q2 = _build_critic(self.config).to(device)
        self.q1_target = copy.deepcopy(self.q1).requires_grad_(False)
        self.q2_target = copy.deepcopy(self.q2).requires_grad_(False)

        target_ent = (
            self.config.target_entropy
            if self.config.target_entropy is not None
            else -float(self.config.action_dim)
        )
        self.target_entropy = target_ent

        self.log_alpha = torch.tensor(
            math.log(self.config.init_alpha),
            device=device,
            dtype=torch.float32,
            requires_grad=self.config.autotune_alpha,
        )

    @property
    def alpha(self) -> "Tensor":
        return self.log_alpha.exp()

    def parameters(self):
        """All trainable parameters as a flat list, useful for `to(device)`."""
        params = list(self.actor.parameters()) + list(self.q1.parameters()) + list(self.q2.parameters())
        if self.config.autotune_alpha:
            params.append(self.log_alpha)
        return params

    def _sample_action(self, rl_token: "Tensor", state: "Tensor"):
        """Reparameterized squashed-Gaussian sample. Returns (action, log_prob)."""
        import torch
        mu, log_std = self.actor(rl_token, state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        x = normal.rsample()
        squashed = torch.tanh(x)
        # Log-prob with tanh correction (standard SAC trick).
        log_prob = normal.log_prob(x).sum(-1) - torch.log(1 - squashed.pow(2) + 1e-6).sum(-1)
        action = squashed * self.config.action_residual_scale
        return action, log_prob

    def act(self, rl_token: "Tensor", state: "Tensor", *, deterministic: bool = False):
        """Return (action_residual, log_prob).

        ``deterministic=True`` collapses to the mean — use it at eval time.
        """
        import torch
        if deterministic:
            mu, _ = self.actor(rl_token, state)
            action = torch.tanh(mu) * self.config.action_residual_scale
            return action, torch.zeros(action.size(0), device=action.device)
        return self._sample_action(rl_token, state)


# --- Trainer ------------------------------------------------------------


class RLTTrainer:
    """Minimal SAC updater. Hand it a batch dict and it returns a metrics dict.

    Batch keys:
      - ``rl_token``        : (B, T_token)
      - ``state``           : (B, D_state)
      - ``action``          : (B, D_action)   — Δa actually executed
      - ``reward``          : (B,)
      - ``next_rl_token``   : (B, T_token)
      - ``next_state``      : (B, D_state)
      - ``done``            : (B,) float in {0.0, 1.0}
    """

    def __init__(self, head: RLTHead) -> None:
        import torch

        self.head = head
        cfg = head.config
        self.actor_opt = torch.optim.Adam(head.actor.parameters(), lr=cfg.actor_lr)
        self.q1_opt = torch.optim.Adam(head.q1.parameters(), lr=cfg.critic_lr)
        self.q2_opt = torch.optim.Adam(head.q2.parameters(), lr=cfg.critic_lr)
        self.alpha_opt = (
            torch.optim.Adam([head.log_alpha], lr=cfg.alpha_lr)
            if cfg.autotune_alpha
            else None
        )

    def _polyak(self) -> None:
        import torch
        with torch.no_grad():
            for source, target in (
                (self.head.q1, self.head.q1_target),
                (self.head.q2, self.head.q2_target),
            ):
                for p, p_t in zip(source.parameters(), target.parameters()):
                    p_t.data.mul_(1.0 - self.head.config.tau)
                    p_t.data.add_(self.head.config.tau * p.data)

    def update(self, batch: dict) -> dict[str, float]:
        import torch

        head = self.head
        cfg = head.config

        rl_token = batch["rl_token"]
        state = batch["state"]
        action = batch["action"]
        reward = batch["reward"]
        next_rl = batch["next_rl_token"]
        next_state = batch["next_state"]
        done = batch["done"]

        # --- Critic update -------------------------------------------------
        with torch.no_grad():
            next_action, next_logp = head._sample_action(next_rl, next_state)
            target_q1 = head.q1_target(next_rl, next_state, next_action)
            target_q2 = head.q2_target(next_rl, next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - head.alpha * next_logp
            target = reward + (1.0 - done) * cfg.discount * target_q

        q1_pred = head.q1(rl_token, state, action)
        q2_pred = head.q2(rl_token, state, action)
        q1_loss = ((q1_pred - target) ** 2).mean()
        q2_loss = ((q2_pred - target) ** 2).mean()

        self.q1_opt.zero_grad(set_to_none=True)
        q1_loss.backward()
        self.q1_opt.step()
        self.q2_opt.zero_grad(set_to_none=True)
        q2_loss.backward()
        self.q2_opt.step()

        # --- Actor update --------------------------------------------------
        new_action, log_prob = head._sample_action(rl_token, state)
        q1_pi = head.q1(rl_token, state, new_action)
        q2_pi = head.q2(rl_token, state, new_action)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (head.alpha.detach() * log_prob - q_pi).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # --- Entropy temperature -------------------------------------------
        alpha_loss_val = 0.0
        if self.alpha_opt is not None:
            alpha_loss = -(head.log_alpha * (log_prob.detach() + head.target_entropy)).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha_loss_val = float(alpha_loss.detach())

        # --- Polyak averaging ----------------------------------------------
        self._polyak()

        return {
            "q1_loss": float(q1_loss.detach()),
            "q2_loss": float(q2_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "alpha": float(head.alpha.detach()),
            "alpha_loss": alpha_loss_val,
            "mean_logp": float(log_prob.detach().mean()),
        }


# --- Replay buffer ------------------------------------------------------


class ReplayBuffer:
    """Tiny ring-buffer replay for RLT. Stores everything as torch tensors.

    The paper's RLT runs are short (hours) so a few hundred-thousand
    transitions max — no need for prioritized sampling or RLDS support.
    """

    def __init__(
        self,
        capacity: int,
        rl_token_dim: int,
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
    ) -> None:
        import torch

        self.capacity = capacity
        self.device = torch.device(device)
        self._rl_token = torch.zeros(capacity, rl_token_dim, device=self.device)
        self._state = torch.zeros(capacity, state_dim, device=self.device)
        self._action = torch.zeros(capacity, action_dim, device=self.device)
        self._reward = torch.zeros(capacity, device=self.device)
        self._next_rl_token = torch.zeros(capacity, rl_token_dim, device=self.device)
        self._next_state = torch.zeros(capacity, state_dim, device=self.device)
        self._done = torch.zeros(capacity, device=self.device)
        self._idx = 0
        self._size = 0

    def add(
        self,
        rl_token,
        state,
        action,
        reward,
        next_rl_token,
        next_state,
        done,
    ) -> None:
        i = self._idx
        self._rl_token[i] = rl_token
        self._state[i] = state
        self._action[i] = action
        self._reward[i] = reward
        self._next_rl_token[i] = next_rl_token
        self._next_state[i] = next_state
        self._done[i] = float(done)
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        import torch
        idx = torch.randint(0, self._size, (batch_size,), device=self.device)
        return {
            "rl_token": self._rl_token[idx],
            "state": self._state[idx],
            "action": self._action[idx],
            "reward": self._reward[idx],
            "next_rl_token": self._next_rl_token[idx],
            "next_state": self._next_state[idx],
            "done": self._done[idx],
        }

    def __len__(self) -> int:
        return self._size
