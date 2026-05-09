"""Knowledge Insulation (KI) — train fast, run fast, generalize better (May 2025).

Recipe:
  1. Train the VLM backbone on **discrete FAST tokens** (autoregressive next-token
     loss) — preserves the pre-trained semantic knowledge.
  2. Train the **continuous action expert** on the same chunks via flow matching.
  3. Insert a **stop-gradient** at the VLM ↔ expert interface so the expert's
     gradients cannot degrade the VLM.

Result: autoregressive VLA training converges ≈ 7.5× faster than pure-diffusion
training and the resulting model retains stronger language grounding.

Paper: papers/2025-05-28_ki_train-fast-run-fast-generalize-better.pdf
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KIConfig:
    discrete_loss_weight: float = 1.0   # autoregressive FAST-token loss
    continuous_loss_weight: float = 1.0  # flow-matching loss
    stop_gradient_at_interface: bool = True


class KITrainer:
    """Skeleton training loop for the KI recipe.

    TODO:
      - tokenize actions with `pi_stack.tokenization.fast.FASTTokenizer`
      - forward backbone, compute next-token loss on discrete actions
      - apply stop-gradient at VLM activations passed to the action expert
      - forward action expert, compute flow-matching loss
      - sum losses with KIConfig weights
    """

    def __init__(self, config: KIConfig) -> None:
        self.config = config

    def step(self, *args, **kwargs):
        raise NotImplementedError
