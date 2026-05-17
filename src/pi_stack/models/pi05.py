"""π₀.₅ — open-world generalization (Apr 2025).

Refines Hi Robot's hierarchy into a single unified model that handles both
high-level subtask prediction and low-level action generation. Co-trains on
heterogeneous tasks: mobile manipulator data + web data + VQA + semantic
subtask prediction. Demonstrated 10-15 minute tasks in unseen environments.

Paper: papers/2025-04-22_pi05_open-world-generalization.pdf

Adds two pieces on top of π₀:

1. **Subtask language head** — autoregressive next-token prediction on a
   subtask vocabulary (or the same vocabulary as the VLM). The head is a
   linear projection of VLM features, reusing the backbone's own LM head
   when available. Used both at training (cross-entropy over demo subtask
   labels) and at inference (sample-then-execute the subtask).

2. **Co-training loss mixer** — combines action-prediction loss with
   subtask-prediction loss + auxiliary heads (VQA, web text). Weights
   come from the config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pi_stack.models.pi0 import Pi0Config, Pi0Policy

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class CoTrainWeights:
    mobile_manip: float = 0.4
    web: float = 0.2
    vqa: float = 0.2
    subtask_pred: float = 0.2

    def total(self) -> float:
        return self.mobile_manip + self.web + self.vqa + self.subtask_pred


@dataclass
class Pi05Config(Pi0Config):
    cotrain: CoTrainWeights = field(default_factory=CoTrainWeights)
    predict_subtask_tokens: bool = True
    subtask_vocab_size: int | None = None   # None → use backbone vocab


class Pi05Policy(Pi0Policy):
    """Unified hierarchical policy. Predicts subtask tokens AND action chunks.

    The subtask head is a thin Linear over VLM features. At inference time,
    the caller picks which mode to use (subtask sampling vs action chunk
    sampling) — they share the same forward through the backbone.
    """

    def __init__(self, config: Pi05Config | None = None, *, backbone=None) -> None:
        import torch.nn as nn

        super().__init__(config or Pi05Config(), backbone=backbone)
        cfg: Pi05Config = self.config
        vocab = cfg.subtask_vocab_size or cfg.backbone.vocab_size

        # Subtask head — independent of the backbone's own LM head so we can
        # train it without disturbing the VLM's vocabulary distribution.
        self.subtask_head = nn.Linear(self.hidden_size, vocab)

    # --- Submodule helpers ------------------------------------------------

    def parameters(self):
        return super().parameters() + list(self.subtask_head.parameters())

    def to(self, device) -> "Pi05Policy":
        super().to(device)
        self.subtask_head.to(device)
        return self

    # --- Forward paths ----------------------------------------------------

    def predict_subtask_logits(self, images, language_ids):
        """Return next-token logits over the subtask vocabulary, shape (B, T, V)."""
        _logits, features = self.backbone(language_ids, images=images)
        return self.subtask_head(features)

    # --- Co-training loss mixer ------------------------------------------

    def cotrain_loss(self, losses: dict[str, "Tensor"]) -> "Tensor":
        """Combine per-task losses with config weights.

        Keys recognized: ``'mobile_manip'``, ``'web'``, ``'vqa'``,
        ``'subtask_pred'``. Missing keys are skipped. Weights are normalized
        by the sum of weights for the keys that are actually present, so
        callers don't have to renormalize when a head is skipped on a
        given batch.
        """
        import torch

        weights = self.config.cotrain
        present = {}
        total_w = 0.0
        for k, w in (
            ("mobile_manip", weights.mobile_manip),
            ("web", weights.web),
            ("vqa", weights.vqa),
            ("subtask_pred", weights.subtask_pred),
        ):
            if k in losses:
                present[k] = (w, losses[k])
                total_w += w
        if total_w == 0:
            return torch.zeros((), device=next(iter(losses.values())).device)
        return sum((w / total_w) * loss for (w, loss) in present.values())
