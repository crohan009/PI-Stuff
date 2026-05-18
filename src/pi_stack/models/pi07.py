"""π₀.₇ — steerable generalist with emergent capabilities (Apr 2026).

Multi-modal context conditioning: detailed language instructions, generated
subgoal images, and episode metadata (speed, quality). Built on the Gemma 3
backbone and the MEM dual-memory architecture. Compositional generalization
to new appliances out of the box (e.g., espresso machines).

Paper: papers/2026-04-16_pi07_steerable-model-emergent-capabilities.pdf

Adds three context channels on top of π*₀.₆:

1. **Subgoal image encoder** — SigLIP (or the tiny patch encoder in the
   in-repo backbone) compresses one or more generated subgoal images into
   tokens prepended to the VLM context.
2. **Episode metadata tokens** — a small embedding table over discrete
   metadata buckets: speed (slow/normal/fast), quality (rough/normal/precise),
   persona/style. Steered by the caller to bias the action distribution.
3. **MEM recall integration** — long-term language memory entries can be
   appended as auxiliary text tokens (or, when MEM is upgraded, as their
   own learned-embedding tokens).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

from pi_stack.memory.mem import MEM, MEMConfig
from pi_stack.models.pi06 import Pi06Config, Pi06Policy

if TYPE_CHECKING:
    from torch import Tensor


class Speed(IntEnum):
    slow = 0
    normal = 1
    fast = 2


class Quality(IntEnum):
    rough = 0
    normal = 1
    precise = 2


class Persona(IntEnum):
    neutral = 0
    careful = 1
    bold = 2


@dataclass
class Pi07Config(Pi06Config):
    use_subgoal_images: bool = True
    use_episode_metadata: bool = True
    use_mem: bool = True
    n_subgoal_images: int = 1
    memory: MEMConfig = field(default_factory=MEMConfig)
    metadata_speed_buckets: int = 3
    metadata_quality_buckets: int = 3
    metadata_persona_buckets: int = 3


class Pi07Policy(Pi06Policy):
    """π₀.₇ — steerable generalist.

    Extra ``predict_chunk`` kwargs:
      - ``subgoal_images``   : (B, K, C, H, W) — K generated subgoal images
      - ``metadata``         : dict of int buckets {'speed', 'quality', 'persona'}
      - ``memory_tokens``    : (B, T_mem, hidden) — pre-encoded MEM recall tokens
    """

    def __init__(
        self,
        config: Pi07Config | None = None,
        *,
        backbone=None,
        memory: MEM | None = None,
    ) -> None:
        import torch.nn as nn

        super().__init__(config or Pi07Config(), backbone=backbone)
        cfg: Pi07Config = self.config

        # Subgoal image encoder — reuses the backbone's patch projection
        # for simplicity (real impl swaps in SigLIP). On the tiny backbone
        # this means we just call the same patch_proj on subgoal images.
        if cfg.use_subgoal_images:
            self.subgoal_proj = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
        else:
            self.subgoal_proj = None

        # Episode metadata tokens — three small embedding tables.
        if cfg.use_episode_metadata:
            self.speed_embed = nn.Embedding(cfg.metadata_speed_buckets, self.hidden_size)
            self.quality_embed = nn.Embedding(cfg.metadata_quality_buckets, self.hidden_size)
            self.persona_embed = nn.Embedding(cfg.metadata_persona_buckets, self.hidden_size)
        else:
            self.speed_embed = self.quality_embed = self.persona_embed = None

        # Memory token projection — assumes recalled-memory tokens already
        # live in the policy's hidden space; this is a per-token MLP for any
        # last-mile adjustments. The caller is responsible for actually
        # building the memory tokens (typically via a small encoder over
        # the embedding stored on each MEM entry).
        self.mem_proj = (
            nn.Linear(self.hidden_size, self.hidden_size)
            if cfg.use_mem
            else None
        )

        # Optional MEM store (used by callers that want everything bundled).
        self.memory = memory

    def parameters(self):
        params = super().parameters()
        if self.subgoal_proj is not None:
            params += list(self.subgoal_proj.parameters())
        if self.speed_embed is not None:
            params += list(self.speed_embed.parameters())
            params += list(self.quality_embed.parameters())
            params += list(self.persona_embed.parameters())
        if self.mem_proj is not None:
            params += list(self.mem_proj.parameters())
        return params

    def to(self, device) -> "Pi07Policy":
        super().to(device)
        for m in (self.subgoal_proj, self.speed_embed, self.quality_embed, self.persona_embed, self.mem_proj):
            if m is not None:
                m.to(device)
        return self

    # --- Context assembly -------------------------------------------------

    def _encode_subgoal_images(self, subgoal_images: "Tensor") -> "Tensor":
        """``(B, K, C, H, W) → (B, K*P, hidden)``  -- one bag of image tokens.

        Delegates to ``backbone.encode_image_features`` so both TinyBackbone
        (Conv2d patch projection) and PaliGemmaAdapter (vision_tower +
        multi_modal_projector) work without special-casing.
        """
        if subgoal_images is None or self.subgoal_proj is None:
            return None
        B, K, C, H, W = subgoal_images.shape
        flat = subgoal_images.reshape(B * K, C, H, W)
        patches = self.backbone.encode_image_features(flat)
        patches = patches.reshape(B, K * patches.size(1), -1)
        return self.subgoal_proj(patches)

    def _metadata_tokens(self, metadata: dict | None, batch_size: int, device) -> "Tensor | None":
        import torch

        if metadata is None or self.speed_embed is None:
            return None
        speed = metadata.get("speed", Speed.normal)
        quality = metadata.get("quality", Quality.normal)
        persona = metadata.get("persona", Persona.neutral)
        ids = lambda v: torch.tensor([int(v)] * batch_size, device=device, dtype=torch.long)
        tokens = torch.stack(
            [
                self.speed_embed(ids(speed)),
                self.quality_embed(ids(quality)),
                self.persona_embed(ids(persona)),
            ],
            dim=1,
        )   # (B, 3, hidden)
        return tokens

    def encode_context(
        self,
        images,
        state,
        language_ids,
        *,
        advantage_token=None,
        subgoal_images=None,
        metadata=None,
        memory_tokens=None,
    ):
        import torch

        ctx = super().encode_context(
            images, state, language_ids, advantage_token=advantage_token
        )

        extras = []
        sg = self._encode_subgoal_images(subgoal_images)
        if sg is not None:
            extras.append(sg)
        md = self._metadata_tokens(metadata, batch_size=ctx.size(0), device=ctx.device)
        if md is not None:
            extras.append(md)
        if memory_tokens is not None and self.mem_proj is not None:
            extras.append(self.mem_proj(memory_tokens))

        if extras:
            ctx = torch.cat([*extras, ctx], dim=1)
        return ctx

    def predict_chunk(
        self,
        images,
        state,
        language_ids,
        *,
        advantage_token=None,
        subgoal_images=None,
        metadata=None,
        memory_tokens=None,
        prefix=None,
        flow_steps=None,
    ):
        ctx = self.encode_context(
            images,
            state,
            language_ids,
            advantage_token=advantage_token,
            subgoal_images=subgoal_images,
            metadata=metadata,
            memory_tokens=memory_tokens,
        )
        return self.action_expert.sample_chunk(ctx, prefix=prefix, flow_steps=flow_steps)

    # --- MEM helpers ------------------------------------------------------

    def recall_memory_tokens(self, query: str, k: int = 4) -> "Tensor | None":
        """Convenience: query the attached MEM and return projected tokens.

        Returns ``(1, k_actual, hidden)`` — one batch row. Callers with
        batched obs should broadcast. Returns None if no MEM is attached
        or the query has no hits.
        """
        import torch

        if self.memory is None:
            return None
        entries = self.memory.recall(query, k=k)
        if not entries:
            return None
        # Pad each embedding up to hidden_size with zeros — keeps the path
        # framework-free (no extra learned compressor needed).
        rows = []
        for e in entries:
            emb = torch.from_numpy(e.embedding)
            if emb.numel() < self.hidden_size:
                emb = torch.nn.functional.pad(emb, (0, self.hidden_size - emb.numel()))
            else:
                emb = emb[: self.hidden_size]
            rows.append(emb)
        return torch.stack(rows, dim=0).unsqueeze(0).float()
