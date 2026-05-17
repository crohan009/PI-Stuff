"""Knowledge Insulation (KI) — train fast, run fast, generalize better (May 2025).

Recipe (paper §3):
  1. Train the VLM backbone on **discrete FAST tokens** — autoregressive
     cross-entropy. Preserves the pre-trained semantic knowledge because the
     loss surface is the one the VLM was originally trained on.
  2. Train the **continuous action expert** on the same chunks via flow
     matching. The expert reads VLM activations as conditioning.
  3. Insert a **stop-gradient** at the VLM ↔ expert interface. The expert's
     gradients NEVER flow back into the VLM.

Result: autoregressive VLA training converges ≈ 7.5× faster than pure
diffusion training and the resulting model retains stronger language
grounding.

Paper: papers/2025-05-28_ki_train-fast-run-fast-generalize-better.pdf

This module provides the two primitives that make KI a *recipe* and not a
specific architecture:

- ``insulate(features)`` — apply at the VLM→expert handoff.
- ``ki_loss(...)`` — combine the discrete and continuous losses with the
  prescribed weights.

The full training loop (gradient steps, optimizers, schedulers) lives in
the caller — KI doesn't know what backbone or action expert you're using,
only that the gradient must not cross the insulation boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import Tensor


@dataclass
class KIConfig:
    discrete_loss_weight: float = 1.0
    continuous_loss_weight: float = 1.0
    stop_gradient_at_interface: bool = True


def insulate(features: "Tensor", *, enabled: bool = True) -> "Tensor":
    """KI stop-gradient at the VLM ↔ action-expert interface.

    Call this on VLM features BEFORE feeding them into the action expert.
    With ``enabled=True`` (the KI recipe), gradients computed by the action
    expert's loss will not flow back into the VLM — its weights are only
    updated by the discrete-token loss path.

    Pass ``enabled=False`` for ablations (everything flows through, recovers
    the pre-KI training regime).
    """
    return features.detach() if enabled else features


def ki_loss(
    discrete_logits: "Tensor",
    discrete_targets: "Tensor",
    continuous_pred: "Tensor",
    continuous_target: "Tensor",
    config: KIConfig | None = None,
    *,
    discrete_ignore_index: int = -100,
) -> dict[str, "Tensor"]:
    """Compute the KI dual loss.

    Args:
        discrete_logits: ``(B, T, V)`` — next-token logits from the VLM over
            the FAST vocabulary.
        discrete_targets: ``(B, T)`` int64 — FAST token ids. Use
            ``discrete_ignore_index`` (default -100) to mask padding.
        continuous_pred: ``(B, H, D)`` — action-expert velocity predictions
            (the flow-matching network output ``v_θ(a_t, t, ctx)``).
        continuous_target: ``(B, H, D)`` — flow-matching target
            (``a_1 - a_0`` for the straight-line interpolant).

    Returns:
        Dict with keys: ``'loss'`` (the combined scalar to ``.backward()``),
        and the detached components ``'discrete'`` and ``'continuous'`` for
        logging.

    The caller is responsible for having applied ``insulate()`` to the VLM
    features that produced ``continuous_pred``. ``ki_loss`` only combines
    the two loss terms; the gradient blocking happens in the forward pass.
    """
    import torch  # local to keep the module importable without torch
    from torch.nn import functional as F

    cfg = config or KIConfig()

    discrete = F.cross_entropy(
        discrete_logits.reshape(-1, discrete_logits.size(-1)),
        discrete_targets.reshape(-1),
        ignore_index=discrete_ignore_index,
        reduction="mean",
    )
    continuous = F.mse_loss(continuous_pred, continuous_target)

    total = (
        cfg.discrete_loss_weight * discrete + cfg.continuous_loss_weight * continuous
    )
    return {
        "loss": total,
        "discrete": discrete.detach(),
        "continuous": continuous.detach(),
    }


def flow_matching_target(
    actions: "Tensor", noise: "Tensor", t: "Tensor"
) -> tuple["Tensor", "Tensor"]:
    """Construct the flow-matching interpolant + target for a batch.

    The flow-matching loss (used by π₀'s action expert and re-used here) is
    ``E_t [|| v_θ(a_t, t, ctx) - (a_1 - a_0) ||²]`` where ``a_t = (1-t)·a_0 + t·a_1``.

    Args:
        actions: ``(B, H, D)`` ground-truth actions. Plays the role of ``a_1``.
        noise: ``(B, H, D)`` standard normal noise. Plays the role of ``a_0``.
        t: ``(B,)`` or ``(B, 1, 1)`` interpolation time in ``[0, 1]``.

    Returns:
        ``(a_t, target)`` — both ``(B, H, D)``. Feed ``a_t`` and ``t`` into
        the action expert; regress against ``target``.
    """
    if t.dim() == 1:
        t = t.view(-1, 1, 1)
    a_t = (1.0 - t) * noise + t * actions
    target = actions - noise
    return a_t, target


class KITrainer:
    """Reference KI training step — composes the pieces above.

    This is intentionally tiny. Real trainers (DDP, mixed precision, grad
    accumulation, EMA, etc.) wrap this. The point of this class is to
    capture the *recipe* clearly enough that you can read one method and
    understand what KI does.

    Caller contract:
      - ``backbone(inputs, target_ids)`` returns a dict with keys
        ``'logits'`` ``(B, T, V)`` and ``'features'`` ``(B, T, F)``.
      - ``action_expert(a_t, t, vlm_features)`` returns ``(B, H, D)``
        velocity predictions.
    """

    def __init__(
        self,
        backbone,
        action_expert,
        fast_tokenizer,
        config: KIConfig | None = None,
    ) -> None:
        self.backbone = backbone
        self.action_expert = action_expert
        self.fast = fast_tokenizer
        self.config = config or KIConfig()

    def step(self, batch: dict) -> dict[str, "Tensor"]:
        """Run one KI training step.

        Args:
            batch: dict with at least
                - ``'inputs'``: opaque pass-through to the backbone
                - ``'actions'``: ``(B, H, D)`` float tensor of demo actions
                - ``'fast_token_ids'``: ``(B, T)`` int64 — pre-tokenized FAST ids

        Returns:
            The dict from :func:`ki_loss`.
        """
        import torch

        # 1) Backbone forward — discrete-token loss path.
        backbone_out = self.backbone(batch["inputs"], target_ids=batch["fast_token_ids"])
        logits = backbone_out["logits"]
        features = backbone_out["features"]

        # 2) KI stop-gradient at the VLM ↔ expert interface.
        ctx = insulate(features, enabled=self.config.stop_gradient_at_interface)

        # 3) Flow-matching forward through the action expert.
        actions = batch["actions"]
        noise = torch.randn_like(actions)
        t = torch.rand(actions.size(0), device=actions.device)
        a_t, target = flow_matching_target(actions, noise, t)
        v_pred = self.action_expert(a_t, t, ctx)

        # 4) Combined loss — caller calls .backward() on the 'loss' tensor.
        return ki_loss(
            discrete_logits=logits,
            discrete_targets=batch["fast_token_ids"],
            continuous_pred=v_pred,
            continuous_target=target,
            config=self.config,
        )
