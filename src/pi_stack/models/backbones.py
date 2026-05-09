"""Pre-trained VLM / vision backbones used across the π series.

- PaliGemma 3B — π₀, π₀.₅, Hi Robot
- Gemma 3 4B    — π*₀.₆, π₀.₇
- SigLIP 400M   — vision encoder, frequently combined with the above
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class BackboneSpec:
    """Identifier + HF repo for a supported backbone."""

    name: Literal["paligemma", "gemma3", "siglip"]
    hf_repo: str
    param_count: str  # human-readable, e.g. "3B"


PALIGEMMA_3B = BackboneSpec("paligemma", "google/paligemma-3b-pt-224", "3B")
GEMMA3_4B = BackboneSpec("gemma3", "google/gemma-3-4b-pt", "4B")
SIGLIP_400M = BackboneSpec("siglip", "google/siglip-base-patch16-224", "400M")


def load_backbone(spec: BackboneSpec, *, device: str | None = None):
    """Load a backbone from HuggingFace.

    TODO: implement once `transformers` is wired in. Should return a tuple of
    ``(model, processor)`` and route to the correct AutoClass per backbone.
    """
    raise NotImplementedError(
        "Backbone loading is not implemented yet. "
        "Install the 'ml' extra and fill in via `transformers.Auto*`."
    )
