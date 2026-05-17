"""Flow-matching objective for the action expert.

Variant of diffusion training. The expert learns the velocity field
``v_θ(a_t, t, context)`` such that a straight-line interpolant between noise
``a_0 ∼ N(0, I)`` and the demonstrated action chunk ``a_1`` is recovered. At
inference, sample noise and integrate ``v_θ`` for ``flow_steps`` Euler
updates.

Introduced in π₀ (papers/2024-10-31_pi0_first-generalist-policy.pdf §3),
referenced throughout the arc.

The ``euler_sample`` function supports **prefix inpainting** — fixing the
first ``P`` actions to a known sequence while sampling the rest. This is
what RTC (papers/2025-06-09_rtc_real-time-action-chunking-large-models.pdf)
uses to start the next chunk where the current one leaves off.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import torch
    from torch import Tensor


@dataclass
class FlowMatchingConfig:
    sigma_min: float = 1e-4
    flow_steps: int = 5            # inference-time Euler steps
    loss: str = "mse"              # mse | huber


def flow_matching_loss(
    velocity_pred: "Tensor",
    actions: "Tensor",
    noise: "Tensor",
    t: "Tensor",
    *,
    loss: str = "mse",
) -> "Tensor":
    """Per-step velocity regression loss.

    Args:
        velocity_pred: ``(B, H, D)`` from ``v_θ(a_t, t, ctx)``.
        actions: ``(B, H, D)`` ground-truth actions (``a_1``).
        noise: ``(B, H, D)`` standard normal noise (``a_0``).
        t: ``(B,)`` or ``(B, 1, 1)`` interpolation time in ``[0, 1]``. The
           caller is expected to have already used ``t`` to build ``a_t``;
           we only need ``actions - noise`` as the regression target.

    Returns:
        Scalar loss tensor.
    """
    import torch
    import torch.nn.functional as F

    target = actions - noise
    if loss == "mse":
        return F.mse_loss(velocity_pred, target)
    if loss == "huber":
        return F.huber_loss(velocity_pred, target)
    raise ValueError(f"unknown loss '{loss}'")


def euler_sample(
    velocity_fn: Callable[..., "Tensor"],
    ctx: "Tensor",
    *,
    horizon: int,
    action_dim: int,
    flow_steps: int = 5,
    prefix: "Tensor | None" = None,
    generator: "torch.Generator | None" = None,
) -> "Tensor":
    """Inference-time integration of ``v_θ`` with ``flow_steps`` Euler updates.

    Args:
        velocity_fn: callable ``(a_t, t, ctx) -> (B, H, D) velocity``.
        ctx: ``(B, T_ctx, F)`` context tokens (VLM features) shared across
             integration steps.
        horizon: ``H`` — number of action steps per chunk.
        action_dim: ``D`` — per-step action dimension.
        flow_steps: number of Euler steps over ``t ∈ [0, 1]`` (5-10 typical).
        prefix: optional ``(B, P, D)`` inpainting prefix. If provided, the
            first ``P`` actions of the sampled chunk are forced to equal
            ``prefix`` after every Euler step. This is the RTC contract.
        generator: optional torch RNG for reproducible noise.

    Returns:
        ``(B, H, D)`` sampled action chunk.
    """
    import torch

    device = ctx.device
    batch_size = ctx.size(0)
    if generator is None:
        a_t = torch.randn(batch_size, horizon, action_dim, device=device)
    else:
        a_t = torch.randn(
            batch_size, horizon, action_dim,
            device=device, generator=generator,
        )

    if prefix is not None:
        prefix_len = prefix.size(1)
        a_t[:, :prefix_len, :] = prefix

    dt = 1.0 / flow_steps
    for step in range(flow_steps):
        t_scalar = (step + 0.5) * dt   # mid-point t for this slab
        t = torch.full((batch_size,), t_scalar, device=device, dtype=a_t.dtype)
        v = velocity_fn(a_t, t, ctx)
        a_t = a_t + dt * v
        # RTC inpainting: re-pin the prefix after each step.
        if prefix is not None:
            a_t[:, :prefix_len, :] = prefix

    return a_t
