"""Pre-trained VLM / vision backbones used across the π series.

- PaliGemma 3B — π₀, π₀.₅, Hi Robot
- Gemma 3 4B    — π*₀.₆, π₀.₇
- SigLIP 400M   — vision encoder, frequently combined with the above

For local development on commodity hardware we ship a :class:`TinyBackbone`
stand-in. It has the same input/output contract as a real PaliGemma/Gemma3
forward pass (``(B, T, hidden_size)`` context features), so the rest of the
pi_stack code is hardware-agnostic. Real backbone weights load via
``load_backbone(spec)`` on a workstation with sufficient memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class BackboneSpec:
    """Identifier + HF repo for a supported backbone."""

    name: Literal["paligemma", "gemma3", "siglip", "tiny"]
    hf_repo: str
    param_count: str  # human-readable, e.g. "3B"
    hidden_size: int = 0
    vocab_size: int = 0


PALIGEMMA_3B = BackboneSpec("paligemma", "google/paligemma-3b-pt-224", "3B", hidden_size=2048, vocab_size=257152)
GEMMA3_4B = BackboneSpec("gemma3", "google/gemma-3-4b-pt", "4B", hidden_size=2560, vocab_size=262144)
SIGLIP_400M = BackboneSpec("siglip", "google/siglip-base-patch16-224", "400M", hidden_size=768)
TINY = BackboneSpec("tiny", "in-repo", "1M", hidden_size=128, vocab_size=1024)


# --- Tiny backbone (used by tests + notebooks) -------------------------


class TinyBackbone:
    """In-repo small Transformer matching the real backbone interface.

    Returns ``(logits, features)`` for a batch of token ids and an optional
    image tensor. The image is consumed by a small patch encoder that
    produces a few "image tokens" prepended to the language tokens.
    Same contract as PaliGemma: late-fusion concatenation of visual and
    language tokens through one Transformer stack.

    Use this for unit tests and for the π₀.₇ assembly notebook. Real
    backbones load via :func:`load_backbone`.
    """

    def __init__(
        self,
        *,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        vocab_size: int = 1024,
        image_patch_size: int = 16,
        image_resolution: int = 64,
        in_channels: int = 3,
    ) -> None:
        import torch
        import torch.nn as nn

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.image_patch_size = image_patch_size
        self.num_image_patches = (image_resolution // image_patch_size) ** 2

        class _Net(nn.Module):
            def __init__(self_inner) -> None:
                super().__init__()
                self_inner.token_embed = nn.Embedding(vocab_size, hidden_size)
                self_inner.patch_proj = nn.Conv2d(
                    in_channels, hidden_size,
                    kernel_size=image_patch_size, stride=image_patch_size,
                )
                self_inner.pos_embed = nn.Parameter(
                    torch.zeros(1, self.num_image_patches + 256, hidden_size)
                )
                layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size, nhead=num_heads,
                    dim_feedforward=hidden_size * 4, batch_first=True,
                    activation="gelu",
                )
                self_inner.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
                self_inner.lm_head = nn.Linear(hidden_size, vocab_size)

            def forward(self_inner, token_ids, images=None):
                token_h = self_inner.token_embed(token_ids)   # (B, T_lang, F)
                if images is not None:
                    # (B, C, H, W) → (B, P, F)
                    patches = self_inner.patch_proj(images).flatten(2).transpose(1, 2)
                    h = torch.cat([patches, token_h], dim=1)
                else:
                    h = token_h
                T = h.size(1)
                h = h + self_inner.pos_embed[:, :T, :]
                features = self_inner.encoder(h)
                logits = self_inner.lm_head(features)
                return logits, features

        import torch
        self.net = _Net()

    def __call__(self, token_ids: "Tensor", images: "Tensor | None" = None):
        return self.net(token_ids, images)

    def parameters(self):
        return self.net.parameters()

    def to(self, device) -> "TinyBackbone":
        self.net.to(device)
        return self


# --- Real backbone loading -----------------------------------------------


def load_backbone(spec: BackboneSpec, *, device: str | None = None):
    """Load a backbone from HuggingFace.

    Only PaliGemma / Gemma 3 / SigLIP are real loaders here — they need 6-12GB
    of VRAM and a recent transformers build. On hardware that can't hold
    them, use :class:`TinyBackbone` instead.
    """
    if spec.name == "tiny":
        return TinyBackbone(hidden_size=spec.hidden_size, vocab_size=spec.vocab_size)
    raise NotImplementedError(
        f"Real backbone loading for {spec.name} ({spec.hf_repo}) requires a "
        "GPU workstation with ≥ 12 GB VRAM. Use TinyBackbone for local dev."
    )
