"""Flow-matching objective for the action expert.

Variant of diffusion training. The expert learns the velocity field
v_θ(a_t, t, context) such that a straight-line interpolant between noise
a_0 ∼ N(0, I) and the demonstrated action chunk a_1 is recovered. At
inference, sample noise and integrate v_θ for `flow_steps` Euler updates.

Introduced in π₀ (papers/2024-10-31_pi0_first-generalist-policy.pdf §3),
referenced throughout the arc.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FlowMatchingConfig:
    sigma_min: float = 1e-4
    flow_steps: int = 5            # inference-time Euler steps
    loss: str = "mse"              # mse | huber


def flow_matching_loss(*args, **kwargs):
    """Per-step velocity regression loss.

    TODO: implement
        a_t = (1 - t) * a_0 + t * a_1
        target = a_1 - a_0
        loss = ||v_θ(a_t, t, ctx) - target||²
    """
    raise NotImplementedError


def euler_sample(*args, **kwargs):
    """Inference-time integration of v_θ with `flow_steps` Euler updates.

    TODO: implement straight-line ODE solver over t ∈ [0, 1].
    """
    raise NotImplementedError
