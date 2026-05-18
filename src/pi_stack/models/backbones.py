"""Pre-trained VLM / vision backbones used across the π series.

- PaliGemma 3B — π₀, π₀.₅, Hi Robot
- Gemma 3 4B    — π*₀.₆, π₀.₇
- SigLIP 400M   — vision encoder, frequently combined with the above

For local development on commodity hardware we ship a :class:`TinyBackbone`
stand-in. It has the same input/output contract as a real PaliGemma/Gemma3
forward pass (``(logits, features)``), so the rest of the pi_stack code is
hardware-agnostic.

The real PaliGemma loader lives here too. It wraps
``transformers.PaliGemmaForConditionalGeneration`` into the same minimal
contract — see :class:`PaliGemmaAdapter`. Caller is responsible for
tokenization via the adapter's ``processor`` attribute.
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

    def encode_image_features(self, images: "Tensor") -> "Tensor":
        """``(B, C, H, W) → (B, num_patches, hidden_size)``.

        Convenience for callers (Pi07Policy.subgoal pathway) that want image
        tokens *without* going through the full Transformer stack.
        """
        return self.net.patch_proj(images).flatten(2).transpose(1, 2)

    def parameters(self):
        return self.net.parameters()

    def to(self, device) -> "TinyBackbone":
        self.net.to(device)
        return self


# --- Real backbone — PaliGemma ----------------------------------------


class PaliGemmaAdapter:
    """Wraps HF ``PaliGemmaForConditionalGeneration`` into the pi_stack contract.

    PaliGemma's HF interface returns a full ``ModelOutput`` from a complex
    call signature; we narrow it down to the same ``(logits, features)``
    that :class:`TinyBackbone` returns so :class:`pi_stack.models.pi0.Pi0Policy`
    can swap backbones without code changes.

    Attributes:
        model:      the HF ``PaliGemmaForConditionalGeneration`` instance
        processor:  the matching HF ``PaliGemmaProcessor`` (use it to
                    tokenize text + encode images into ``(input_ids, pixel_values)``)
        hidden_size, vocab_size: text-model dims, mirrored from
                    ``model.config.text_config`` for downstream auto-sizing

    Notes:
        - ``token_ids`` passed to ``__call__`` must include the special
          ``<image>`` placeholder tokens at the positions where image
          features should be spliced in. The ``processor`` produces these.
        - ``images`` are PaliGemma's ``pixel_values`` — (B, C, H, W) after
          the processor's image transform (normalize + resize to 224 px).
        - The adapter automatically casts ``images`` to the model dtype
          (typically bf16) so callers don't have to worry about it.
    """

    def __init__(self, model, processor) -> None:
        self.model = model
        self.processor = processor

        cfg = model.config
        text_cfg = getattr(cfg, "text_config", cfg)
        self.hidden_size = text_cfg.hidden_size
        self.vocab_size = text_cfg.vocab_size

    @property
    def _dtype(self):
        return next(self.model.parameters()).dtype

    @property
    def _device(self):
        return next(self.model.parameters()).device

    def __call__(self, token_ids: "Tensor", images: "Tensor | None" = None):
        kwargs = dict(
            input_ids=token_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        if images is not None:
            kwargs["pixel_values"] = images.to(self._dtype)
        out = self.model(**kwargs)
        # hidden_states is a tuple (n_layers+1,); -1 is the final layer.
        return out.logits, out.hidden_states[-1]

    def encode_image_features(self, images: "Tensor") -> "Tensor":
        """``(B, C, H, W) → (B, num_patches, hidden_size)`` in the text-model space.

        Runs the vision tower + multi-modal projector, skipping the LM. Used
        by :class:`pi_stack.models.pi07.Pi07Policy` for subgoal images.
        """
        x = images.to(self._dtype)
        vision_out = self.model.vision_tower(x)
        vision_features = vision_out.last_hidden_state
        return self.model.multi_modal_projector(vision_features)

    def parameters(self):
        return self.model.parameters()

    def to(self, device) -> "PaliGemmaAdapter":
        self.model.to(device)
        return self

    def preprocess(self, text: str | list[str], images, *, return_tensors: str = "pt"):
        """Tokenize text + encode images into the model's expected inputs.

        Returns a dict typically with ``input_ids`` (B, T) and ``pixel_values``
        (B, C, H, W). Pass these directly into ``predict_chunk(...)``.
        """
        return self.processor(text=text, images=images, return_tensors=return_tensors)


# --- Real backbone loading -----------------------------------------------


def load_backbone(spec: BackboneSpec, *, device: str | None = None, dtype=None, **load_kwargs):
    """Load a backbone from HuggingFace.

    Args:
        spec: which backbone (see ``PALIGEMMA_3B`` etc. constants above).
        device: optional ``str`` or ``torch.device`` — calls ``.to(device)``
            on the model after load. Common values: ``'cuda'``, ``'mps'``.
        dtype: optional torch dtype. Defaults to ``torch.bfloat16`` for
            real models — they ship in bf16 and the savings are large.
        **load_kwargs: passed through to ``from_pretrained()``.

    Returns:
        - ``TinyBackbone`` for ``TINY``
        - ``PaliGemmaAdapter`` for ``PALIGEMMA_3B``
        - raises ``NotImplementedError`` for Gemma 3 / SigLIP (TBD)
    """
    if spec.name == "tiny":
        return TinyBackbone(
            hidden_size=spec.hidden_size or 128,
            vocab_size=spec.vocab_size or 1024,
        )

    if spec.name == "paligemma":
        import torch
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        dtype = dtype if dtype is not None else torch.bfloat16
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            spec.hf_repo,
            torch_dtype=dtype,
            **load_kwargs,
        )
        if device is not None:
            model = model.to(device)
        processor = AutoProcessor.from_pretrained(spec.hf_repo)
        return PaliGemmaAdapter(model, processor)

    raise NotImplementedError(
        f"Real backbone loading for {spec.name} ({spec.hf_repo}) is not wired "
        f"yet. PaliGemma is the experimental baseline (use PALIGEMMA_3B); "
        f"Gemma 3 / SigLIP land when we need them."
    )
