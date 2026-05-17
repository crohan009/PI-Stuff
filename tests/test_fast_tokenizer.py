"""Tests for the FAST tokenizer wrapper.

These tests download the HF `physical-intelligence/fast` processor on first
run (~1 MB), so they're skipped if `transformers` is missing.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("transformers")
pytest.importorskip("sentencepiece")

from pi_stack.tokenization.fast import FASTTokenizer  # noqa: E402


def _sine_chunk(batch: int = 1, horizon: int = 50, dim: int = 14) -> np.ndarray:
    """Smooth, low-frequency, unit-amplitude input — easy for FAST.

    Each action dim is a sine wave with a per-dim phase shift so we exercise
    the multi-dim codepath without amplitude scaling (which would scale MSE
    accordingly and make the threshold a moving target).
    """
    t = np.linspace(0.0, 2 * np.pi, horizon, dtype=np.float32)
    phases = np.linspace(0.0, np.pi, dim, dtype=np.float32)
    base = np.sin(t[:, None] + phases[None, :])
    return np.broadcast_to(base, (batch, horizon, dim)).copy()


def test_single_chunk_roundtrip_low_mse() -> None:
    tok = FASTTokenizer()
    chunk = _sine_chunk(batch=1, horizon=50, dim=14)
    mse = tok.round_trip_mse(chunk)
    assert mse < 1e-3, f"sine chunk MSE too high: {mse}"


def test_batched_input_accepted() -> None:
    tok = FASTTokenizer()
    batch = _sine_chunk(batch=4, horizon=50, dim=14)
    tokens = tok.encode(batch)
    assert isinstance(tokens, list)
    assert len(tokens) == 4
    assert all(isinstance(seq, list) and all(isinstance(t, int) for t in seq) for seq in tokens)


def test_decode_shape_matches_inputs() -> None:
    tok = FASTTokenizer()
    batch = _sine_chunk(batch=3, horizon=50, dim=14)
    tokens = tok.encode(batch)
    recon = tok.decode(tokens, horizon=50, action_dim=14)
    assert recon.shape == (3, 50, 14)
    assert recon.dtype == np.float32


def test_compression_ratio_under_one() -> None:
    """FAST is supposed to compress, not expand."""
    tok = FASTTokenizer()
    batch = _sine_chunk(batch=2, horizon=50, dim=14)
    ratio = tok.compression_ratio(batch)
    # 50 * 14 = 700 floats per chunk; FAST typically emits ~60-100 tokens.
    assert ratio < 0.5, f"compression worse than 2x: ratio={ratio}"


def test_rejects_bad_shape() -> None:
    tok = FASTTokenizer()
    with pytest.raises(ValueError):
        tok.encode(np.zeros((10,), dtype=np.float32))   # 1-D
    with pytest.raises(ValueError):
        tok.encode(np.zeros((2, 3, 4, 5), dtype=np.float32))  # 4-D


def test_local_fallback_raises_until_implemented() -> None:
    from pi_stack.tokenization.fast import FASTConfig

    tok = FASTTokenizer(FASTConfig(use_official_tokenizer=False))
    with pytest.raises(NotImplementedError):
        _ = tok.processor
