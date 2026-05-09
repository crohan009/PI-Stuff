"""Hi Robot — hierarchical System 1 / System 2 reasoning (Feb 2025).

A high-level VLM (System 2) parses complex prompts and mid-task user feedback
into atomic commands. A low-level VLA (System 1, typically π₀) executes them.
Trained with synthetic data: see `pi_stack.data.synthetic` for the
"Data-Generator VLM" pipeline that segments teleoperated demonstrations into
atomic subtasks.

Paper: papers/2025-02-26_hi-robot_listen-and-think-harder.pdf
"""

from __future__ import annotations

from dataclasses import dataclass

from pi_stack.models.backbones import PALIGEMMA_3B, BackboneSpec
from pi_stack.models.pi0 import Pi0Policy


@dataclass
class HiRobotConfig:
    high_level_backbone: BackboneSpec = PALIGEMMA_3B
    max_subtask_tokens: int = 64
    user_interject_window_s: float = 1.0  # how often System 2 re-plans


class HiRobotPolicy:
    """Hierarchical wrapper over a low-level VLA.

    TODO:
      - load high-level VLM
      - implement `replan(observation, prompt)` -> atomic subtask string
      - drive the low-level VLA via ``low_level.predict_chunk(..., language=subtask)``
    """

    def __init__(self, config: HiRobotConfig, low_level: Pi0Policy) -> None:
        self.config = config
        self.low_level = low_level

    def step(self, *args, **kwargs):
        raise NotImplementedError
