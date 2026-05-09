"""Modular flow-matching action expert that attends to VLM activations.

Introduced in π₀; reused (and "knowledge-insulated" via stop-gradient) by KI;
extended for steerable conditioning in π₀.₇.

Sizes seen in the literature: 300M (π₀ small) → 860M (π₀.₆ / π₀.₇ large).
Action chunking horizon H ≈ 50 across the arc.

Paper refs:
    papers/2024-10-31_pi0_first-generalist-policy.pdf — §3 architecture
    papers/2025-05-28_ki_train-fast-run-fast-generalize-better.pdf
    papers/2026-04-16_pi07_steerable-model-emergent-capabilities.pdf
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ActionExpertConfig:
    hidden_size: int = 1024
    num_layers: int = 16
    num_heads: int = 16
    action_dim: int = 14         # bimanual default; override per embodiment
    horizon: int = 50            # H — chunk length
    flow_steps: int = 5          # 5–10 denoising steps at inference
    cross_attend_to_vlm: bool = True


class ActionExpert:
    """Skeleton for the flow-matching action transformer.

    TODO:
      - implement velocity-field network conditioned on (s, a_t, t, vlm_kv)
      - implement chunked sampling loop with `flow_steps` Euler updates
      - support stop-gradient at the VLM↔expert interface (KI recipe)
    """

    def __init__(self, config: ActionExpertConfig) -> None:
        self.config = config

    def velocity(self, *args, **kwargs):
        raise NotImplementedError

    def sample_chunk(self, *args, **kwargs):
        raise NotImplementedError
