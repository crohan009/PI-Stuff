"""π₀.₇ — steerable generalist with emergent capabilities (Apr 2026).

Multi-modal context conditioning: detailed language instructions, generated
subgoal images, and episode metadata (speed, quality). Built on the Gemma 3
backbone and the MEM dual-memory architecture. Compositional generalization
to new appliances out of the box (e.g., espresso machines).

Paper: papers/2026-04-16_pi07_steerable-model-emergent-capabilities.pdf
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pi_stack.memory.mem import MEMConfig
from pi_stack.models.pi06 import Pi06Config, Pi06Policy


@dataclass
class Pi07Config(Pi06Config):
    use_subgoal_images: bool = True
    use_episode_metadata: bool = True  # speed / quality / persona tokens
    memory: MEMConfig = field(default_factory=MEMConfig)


class Pi07Policy(Pi06Policy):
    """Steerable generalist. Extends π*₀.₆ with multi-modal context channels.

    TODO:
      - subgoal-image encoder (probably SigLIP) feeding context tokens
      - metadata token embedding (speed/quality/style)
      - integrate `pi_stack.memory.mem.MEM` for long-horizon recall
    """
