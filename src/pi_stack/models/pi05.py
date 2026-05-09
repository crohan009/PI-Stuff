"""π₀.₅ — open-world generalization (Apr 2025).

Refines Hi Robot's hierarchy into a single unified model that handles both
high-level subtask prediction and low-level action generation. Co-trains on
heterogeneous tasks: mobile manipulator data + web data + VQA + semantic
subtask prediction. Demonstrated 10–15 minute tasks in unseen environments.

Paper: papers/2025-04-22_pi05_open-world-generalization.pdf
"""

from __future__ import annotations

from dataclasses import dataclass

from pi_stack.models.pi0 import Pi0Config, Pi0Policy


@dataclass
class Pi05Config(Pi0Config):
    # Co-training mix weights — tune per dataset budget.
    cotrain_mobile_manip: float = 0.4
    cotrain_web: float = 0.2
    cotrain_vqa: float = 0.2
    cotrain_subtask_pred: float = 0.2

    predict_subtask_tokens: bool = True


class Pi05Policy(Pi0Policy):
    """Unified hierarchical policy. Predicts subtask tokens AND action chunks.

    TODO:
      - extend Pi0 head with a subtask language head (autoregressive)
      - implement co-training loss mixer
      - hand subtask predictions back to the action expert as conditioning
    """

    def __init__(self, config: Pi05Config) -> None:
        super().__init__(config)
        self.config: Pi05Config = config

    def predict_subtask(self, *args, **kwargs):
        raise NotImplementedError
