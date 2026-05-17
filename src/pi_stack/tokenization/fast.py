"""FAST — Frequency-space Action Sequence Tokenization (Jan 2025).

Compresses 1-second chunks of high-frequency robot actions into ~30-60 dense
tokens by applying a Discrete Cosine Transform (DCT) and then Byte-Pair
Encoding (BPE) over the truncated frequency coefficients. Enables
autoregressive VLA training that converges ≈ 5× faster than diffusion-based
models while matching their performance.

Paper: papers/2025-01-16_fast_efficient-robot-action-tokenization.pdf

This module wraps the official HF processor (`physical-intelligence/fast`)
as a black box. The full FAST+ tokenizer is a `UniversalActionProcessor`
that handles the DCT + BPE pipeline internally.

Usage::

    from pi_stack.tokenization.fast import FASTTokenizer
    tok = FASTTokenizer()
    chunks = np.random.randn(4, 50, 14).astype(np.float32)   # (B, H, D)
    tokens = tok.encode(chunks)                              # list of lists
    recon = tok.decode(tokens, horizon=50, action_dim=14)    # (B, 50, 14)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

HF_REPO_ID = "physical-intelligence/fast"


@dataclass
class FASTConfig:
    """Config for the FAST tokenizer wrapper.

    Most fields are defaults used when not explicitly passed to `encode` /
    `decode`. The official tokenizer learned its codebook on a fixed
    action-space distribution; calling it with very different dimensions
    still works, but compression quality may degrade.
    """

    horizon: int = 50
    action_dim: int = 14
    use_official_tokenizer: bool = True


class FASTTokenizer:
    """Thin wrapper over the HF `physical-intelligence/fast` processor.

    Lazy-loads the processor on first use so importing this module is cheap
    (no network, no model weights).
    """

    def __init__(self, config: FASTConfig | None = None) -> None:
        self.config = config or FASTConfig()
        self._processor: Any | None = None

    @property
    def processor(self) -> Any:
        if self._processor is None:
            if not self.config.use_official_tokenizer:
                raise NotImplementedError(
                    "Local DCT+BPE fallback is not implemented. "
                    "Set config.use_official_tokenizer=True to use the HF tokenizer."
                )
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(
                HF_REPO_ID, trust_remote_code=True
            )
        return self._processor

    @staticmethod
    def _as_batched(actions: np.ndarray | list) -> np.ndarray:
        arr = np.asarray(actions, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim != 3:
            raise ValueError(
                f"FAST expects (H, D) or (B, H, D) input; got shape {arr.shape}"
            )
        return arr

    def encode(self, actions: np.ndarray | list) -> list[list[int]]:
        """Encode action chunks to FAST tokens.

        Accepts a single chunk of shape `(H, D)` or a batch `(B, H, D)`.
        Returns a list of `B` token lists. Token lists are variable length —
        FAST chooses how many tokens it needs per chunk based on entropy.
        """
        batched = self._as_batched(actions)
        out = self.processor(batched)
        # The HF processor returns list[list[int]]; ensure plain Python ints
        # for downstream serialization friendliness.
        return [list(map(int, seq)) for seq in out]

    def decode(
        self,
        tokens: list[list[int]],
        *,
        horizon: int | None = None,
        action_dim: int | None = None,
    ) -> np.ndarray:
        """Decode tokens back to a `(B, H, D)` float32 array.

        `horizon` and `action_dim` fall back to the values stored on
        `self.config` if not provided.
        """
        h = horizon if horizon is not None else self.config.horizon
        d = action_dim if action_dim is not None else self.config.action_dim
        recon = self.processor.decode(tokens, time_horizon=h, action_dim=d)
        return np.asarray(recon, dtype=np.float32)

    def round_trip_mse(self, actions: np.ndarray) -> float:
        """Encode-then-decode a chunk and return the per-element MSE.

        Useful for verifying that the tokenizer preserves enough information
        for the action space at hand. Paper reports MSE comparable to or
        below diffusion-VAE baselines.
        """
        batched = self._as_batched(actions)
        tokens = self.encode(batched)
        recon = self.decode(tokens, horizon=batched.shape[1], action_dim=batched.shape[2])
        return float(np.mean((batched - recon) ** 2))

    def compression_ratio(self, actions: np.ndarray) -> float:
        """Tokens-per-float ratio — lower is better compression.

        Paper-equivalent: ratio of total tokens emitted to total float
        coefficients in the input. A value of 0.1 means FAST emits one
        token per 10 input floats.
        """
        batched = self._as_batched(actions)
        tokens = self.encode(batched)
        n_tokens = sum(len(seq) for seq in tokens)
        n_floats = batched.size
        return n_tokens / n_floats
