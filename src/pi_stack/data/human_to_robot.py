"""Human-to-Robot transfer — emergence of transfer in VLAs (Dec 2025).

Egocentric human video can be co-trained with robot teleop without explicit
alignment (no domain adaptation, no hand re-targeting), and manipulation
strategies emerge that transfer to the robot. Unlocks massive human-video
corpora for generalist robot training.

Paper: papers/2025-12-16_human-to-robot_emergence-of-transfer-in-vlas.pdf
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HumanVideoConfig:
    fps: int = 4
    image_resolution: int = 224
    treat_as_action_free: bool = True   # human videos have no action labels


def load_egocentric_dataset(*args, **kwargs):
    """Stream an egocentric human-video dataset for co-training.

    TODO: candidate sources include Ego4D, EPIC-Kitchens. Yield (frames, language)
    pairs without action targets; the loss head is set up to mask actions.
    """
    raise NotImplementedError
