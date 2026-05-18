"""Tests for the PaliGemmaAdapter + Pi0Policy.from_pretrained factory.

Two flavors:

1. **Mocked unit tests** (always run). Use a fake PaliGemma model that
   exposes just the bits we touch — no HF download, no GPU needed.

2. **Real-load integration test** (skipped unless ``PI_STACK_RUN_REAL_PALIGEMMA=1``).
   Actually loads PaliGemma 3B from HuggingFace and runs one forward pass.
   ~6 GB download, ~10-30 s on CPU/MPS, < 5 s on a GPU pod.

The mocks intentionally mirror the *narrow* HF interface our adapter uses
(``model.config.text_config.{hidden_size,vocab_size}``,
 ``model.vision_tower(...).last_hidden_state``,
 ``model.multi_modal_projector(...)``,
 ``model(input_ids=..., pixel_values=..., output_hidden_states=True)``).
If a future ``transformers`` release reshapes those, both these tests AND
real PaliGemma forwards will fail in the same way — we'll see it.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


# --- Fake PaliGemma -----------------------------------------------------


def _fake_paligemma(hidden_size: int = 32, vocab_size: int = 256, num_patches: int = 4):
    """A tiny callable that mimics the HF PaliGemmaForConditionalGeneration
    interface our adapter touches.

    Total params: ~4-8 K. Forward returns a fake ``ModelOutput``-like.
    """
    import torch.nn as nn

    class FakeVisionTower(nn.Module):
        def __init__(self_inner) -> None:
            super().__init__()
            self_inner.proj = nn.Conv2d(3, hidden_size, kernel_size=8, stride=8)

        def forward(self_inner, x):
            feats = self_inner.proj(x).flatten(2).transpose(1, 2)   # (B, P, hidden)
            return SimpleNamespace(last_hidden_state=feats)

    class FakeProjector(nn.Module):
        def __init__(self_inner) -> None:
            super().__init__()
            self_inner.fc = nn.Linear(hidden_size, hidden_size)

        def forward(self_inner, x):
            return self_inner.fc(x)

    class FakeLM(nn.Module):
        def __init__(self_inner) -> None:
            super().__init__()
            self_inner.embed = nn.Embedding(vocab_size, hidden_size)
            self_inner.head = nn.Linear(hidden_size, vocab_size)

        def forward(self_inner, ids):
            h = self_inner.embed(ids)
            return h, self_inner.head(h)

    class FakePaliGemma(nn.Module):
        def __init__(self_inner) -> None:
            super().__init__()
            self_inner.vision_tower = FakeVisionTower()
            self_inner.multi_modal_projector = FakeProjector()
            self_inner.lm = FakeLM()
            self_inner.config = SimpleNamespace(
                text_config=SimpleNamespace(
                    hidden_size=hidden_size,
                    vocab_size=vocab_size,
                ),
            )

        def forward(self_inner, *, input_ids, pixel_values=None, **_):
            # Mimic late-fusion: prepend image patches to language tokens.
            h, logits = self_inner.lm(input_ids)
            if pixel_values is not None:
                vis = self_inner.multi_modal_projector(
                    self_inner.vision_tower(pixel_values).last_hidden_state
                )
                h = torch.cat([vis, h], dim=1)
                # Re-project to logits over the combined sequence.
                logits = self_inner.lm.head(h)
            # HF convention: hidden_states is a tuple of layer outputs.
            return SimpleNamespace(logits=logits, hidden_states=(h,))

    return FakePaliGemma()


def _fake_processor():
    """Trivial stand-in for PaliGemmaProcessor.__call__."""

    class FakeProc:
        def __call__(self_, text=None, images=None, return_tensors="pt"):
            # Pretend the text "pick" tokenizes to a fixed (1, 5) of ids.
            return {
                "input_ids": torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long),
                "pixel_values": torch.zeros(1, 3, 16, 16),   # tiny but well-shaped
            }

    return FakeProc()


# --- Mocked unit tests --------------------------------------------------


def test_adapter_exposes_text_config_dims() -> None:
    from pi_stack.models.backbones import PaliGemmaAdapter

    adapter = PaliGemmaAdapter(_fake_paligemma(hidden_size=48, vocab_size=128),
                                _fake_processor())
    assert adapter.hidden_size == 48
    assert adapter.vocab_size == 128


def test_adapter_call_returns_logits_features() -> None:
    from pi_stack.models.backbones import PaliGemmaAdapter

    adapter = PaliGemmaAdapter(_fake_paligemma(hidden_size=32, vocab_size=64),
                                _fake_processor())
    ids = torch.randint(0, 64, (2, 5))
    images = torch.randn(2, 3, 16, 16)
    logits, features = adapter(ids, images=images)
    # Late-fusion: 4 image patches (16/8 * 16/8) + 5 language = 9 tokens
    assert features.shape == (2, 9, 32)
    assert logits.shape == (2, 9, 64)


def test_adapter_call_without_images() -> None:
    from pi_stack.models.backbones import PaliGemmaAdapter

    adapter = PaliGemmaAdapter(_fake_paligemma(), _fake_processor())
    ids = torch.randint(0, 256, (1, 7))
    logits, features = adapter(ids, images=None)
    # No image tokens — just the 7 language tokens.
    assert features.size(1) == 7


def test_adapter_encode_image_features_in_text_space() -> None:
    from pi_stack.models.backbones import PaliGemmaAdapter

    adapter = PaliGemmaAdapter(_fake_paligemma(hidden_size=24), _fake_processor())
    images = torch.randn(2, 3, 16, 16)
    patches = adapter.encode_image_features(images)
    assert patches.shape == (2, 4, 24)


def test_adapter_preprocess_passes_through_processor() -> None:
    from pi_stack.models.backbones import PaliGemmaAdapter

    adapter = PaliGemmaAdapter(_fake_paligemma(), _fake_processor())
    inputs = adapter.preprocess(text="anything", images=None)
    assert "input_ids" in inputs
    assert "pixel_values" in inputs


def test_pi0policy_accepts_paligemma_adapter() -> None:
    """End-to-end: Pi0Policy + fake PaliGemma + flow-matching chunk sample."""
    from pi_stack.models.action_expert import ActionExpertConfig
    from pi_stack.models.backbones import PALIGEMMA_3B, PaliGemmaAdapter
    from pi_stack.models.pi0 import Pi0Config, Pi0Policy

    spec = PALIGEMMA_3B   # Spec metadata is just dims; we override the actual model.
    adapter = PaliGemmaAdapter(
        _fake_paligemma(hidden_size=spec.hidden_size, vocab_size=spec.vocab_size, num_patches=4),
        _fake_processor(),
    )
    policy = Pi0Policy(
        Pi0Config(
            backbone=spec,
            action_expert=ActionExpertConfig(
                hidden_size=64, num_layers=1, num_heads=4,
                action_dim=7, horizon=10, flow_steps=2, time_embed_dim=32,
            ),
            state_dim=14,
            image_resolution=16,   # match fake vision tower's patch_size=8 → 4 patches
        ),
        backbone=adapter,
    )
    chunk = policy.predict_chunk(
        images=torch.randn(1, 3, 16, 16),
        state=torch.randn(1, 14),
        language_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
    )
    assert chunk.shape == (1, 10, 7)


def test_pi0_subgoal_pathway_uses_encode_image_features() -> None:
    """Verify Pi07Policy talks to the adapter via the abstract method,
    not by reaching into backbone.net.patch_proj (which PaliGemma doesn't have).
    """
    from pi_stack.models.action_expert import ActionExpertConfig
    from pi_stack.models.backbones import PALIGEMMA_3B, PaliGemmaAdapter
    from pi_stack.models.pi07 import Pi07Config, Pi07Policy

    adapter = PaliGemmaAdapter(
        _fake_paligemma(hidden_size=PALIGEMMA_3B.hidden_size, vocab_size=PALIGEMMA_3B.vocab_size),
        _fake_processor(),
    )
    cfg = Pi07Config(
        backbone=PALIGEMMA_3B,
        action_expert=ActionExpertConfig(
            hidden_size=64, num_layers=1, num_heads=4,
            action_dim=7, horizon=8, flow_steps=2, time_embed_dim=32,
        ),
        state_dim=14,
        image_resolution=16,
    )
    policy = Pi07Policy(cfg, backbone=adapter)
    # The subgoal path goes through adapter.encode_image_features — must not raise.
    chunk = policy.predict_chunk(
        images=torch.randn(1, 3, 16, 16),
        state=torch.randn(1, 14),
        language_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        subgoal_images=torch.randn(1, 1, 3, 16, 16),
    )
    assert chunk.shape == (1, 8, 7)


def test_from_pretrained_uses_load_backbone(monkeypatch) -> None:
    """Patch load_backbone to return our fake adapter; verify the factory
    threads the right config through."""
    from pi_stack.models import backbones as backbones_mod
    from pi_stack.models.action_expert import ActionExpertConfig
    from pi_stack.models.backbones import PALIGEMMA_3B, PaliGemmaAdapter
    from pi_stack.models.pi0 import Pi0Config, Pi0Policy

    fake_adapter = PaliGemmaAdapter(
        _fake_paligemma(hidden_size=PALIGEMMA_3B.hidden_size, vocab_size=PALIGEMMA_3B.vocab_size),
        _fake_processor(),
    )

    called = {}

    def fake_load(spec, *, device=None, dtype=None, **kw):
        called["spec"] = spec
        called["device"] = device
        called["dtype"] = dtype
        return fake_adapter

    monkeypatch.setattr(backbones_mod, "load_backbone", fake_load)
    # Also patch the import inside pi0.from_pretrained.
    monkeypatch.setattr("pi_stack.models.pi0.load_backbone", fake_load, raising=False)

    cfg = Pi0Config(
        backbone=PALIGEMMA_3B,
        action_expert=ActionExpertConfig(
            hidden_size=64, num_layers=1, num_heads=4,
            action_dim=7, horizon=6, flow_steps=2, time_embed_dim=32,
        ),
        state_dim=14, image_resolution=16,
    )
    policy = Pi0Policy.from_pretrained(PALIGEMMA_3B, config=cfg, device=None, dtype=None)
    assert called["spec"] == PALIGEMMA_3B
    assert isinstance(policy.backbone, PaliGemmaAdapter)


# --- Real-load integration test (env-gated) -----------------------------


_RUN_REAL = os.environ.get("PI_STACK_RUN_REAL_PALIGEMMA") == "1"


@pytest.mark.skipif(
    not _RUN_REAL,
    reason="set PI_STACK_RUN_REAL_PALIGEMMA=1 to actually download + load PaliGemma 3B",
)
def test_real_paligemma_load_and_forward() -> None:
    """Real load test. Only runs when explicitly requested.

    Downloads ~6 GB on first call (cached at ~/.cache/huggingface afterward).
    Takes seconds on a GPU pod, minutes on CPU/MPS.
    """
    from PIL import Image

    from pi_stack.models.backbones import PALIGEMMA_3B, load_backbone

    adapter = load_backbone(PALIGEMMA_3B)
    inputs = adapter.preprocess(text="caption en", images=Image.new("RGB", (224, 224)))
    logits, features = adapter(inputs["input_ids"], images=inputs["pixel_values"])
    assert logits.size(-1) == PALIGEMMA_3B.vocab_size
    assert features.size(-1) == PALIGEMMA_3B.hidden_size
