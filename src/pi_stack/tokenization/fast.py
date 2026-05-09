"""FAST — Frequency-space Action Sequence Tokenization (Jan 2025).

Compresses 1-second chunks of high-frequency robot actions into ~30 dense
tokens by applying a Discrete Cosine Transform (DCT) and then Byte-Pair
Encoding (BPE) over the truncated frequency coefficients. Enables
autoregressive VLA training that converges ≈ 5× faster than diffusion-based
models while matching their performance.

Paper: papers/2025-01-16_fast_efficient-robot-action-tokenization.pdf

Usage:
    The official tokenizer ships on HuggingFace as a black box::

        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
        tokens = proc(actions=action_chunk)              # encode
        recon = proc.decode(tokens, time_horizon=H, action_dim=D)
"""

from __future__ import annotations

from dataclasses import dataclass

HF_REPO_ID = "physical-intelligence/fast"


@dataclass
class FASTConfig:
    horizon: int = 50            # H — actions per chunk
    action_dim: int = 14
    target_tokens_per_second: int = 30   # paper reports ~30–60 tok/s
    use_official_tokenizer: bool = True  # set False to roll your own DCT+BPE


class FASTTokenizer:
    """Thin wrapper over the official HF tokenizer.

    TODO:
      - lazy-load `AutoProcessor.from_pretrained(HF_REPO_ID, trust_remote_code=True)`
      - expose `encode(actions: np.ndarray) -> list[int]`
      - expose `decode(tokens: list[int], H: int, D: int) -> np.ndarray`
      - if `use_official_tokenizer=False`, implement DCT + BPE pipeline locally
        (scipy.fft.dct, sklearn-style BPE trained on a calibration set)
    """

    def __init__(self, config: FASTConfig | None = None) -> None:
        self.config = config or FASTConfig()
        self._processor = None  # populated on first encode/decode

    def encode(self, actions):
        raise NotImplementedError

    def decode(self, tokens, *, horizon: int | None = None, action_dim: int | None = None):
        raise NotImplementedError
