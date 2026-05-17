"""Tests for π*₀.₆ + π₀.₇ — advantage conditioning, subgoal images,
metadata tokens, MEM recall integration."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pi_stack.memory.mem import MEM, MEMConfig  # noqa: E402
from pi_stack.models.action_expert import ActionExpertConfig  # noqa: E402
from pi_stack.models.backbones import TINY  # noqa: E402
from pi_stack.models.pi06 import Pi06Config, Pi06Policy  # noqa: E402
from pi_stack.models.pi07 import (  # noqa: E402
    Persona,
    Pi07Config,
    Pi07Policy,
    Quality,
    Speed,
)


def _tiny06_config() -> Pi06Config:
    return Pi06Config(
        backbone=TINY,
        action_expert=ActionExpertConfig(
            hidden_size=64, num_layers=1, num_heads=4,
            action_dim=6, horizon=10, flow_steps=2, time_embed_dim=64,
        ),
        state_dim=10,
        image_resolution=32,
        advantage_conditioning=True,
        advantage_token_dim=64,
    )


def _tiny07_config() -> Pi07Config:
    base = _tiny06_config()
    return Pi07Config(
        **{f.name: getattr(base, f.name) for f in base.__dataclass_fields__.values() if not f.name.startswith("_")},
        use_subgoal_images=True,
        use_episode_metadata=True,
        use_mem=True,
        n_subgoal_images=1,
        memory=MEMConfig(embedding_dim=64),
    )


# --- π*₀.₆ tests --------------------------------------------------------


def test_pi06_predict_chunk_with_advantage_token() -> None:
    cfg = _tiny06_config()
    p = Pi06Policy(cfg)
    B = 2
    adv = torch.randn(B, cfg.advantage_token_dim)
    out = p.predict_chunk(
        images=torch.randn(B, 3, cfg.image_resolution, cfg.image_resolution),
        state=torch.randn(B, cfg.state_dim),
        language_ids=torch.randint(0, TINY.vocab_size, (B, 4)),
        advantage_token=adv,
    )
    assert out.shape == (B, cfg.action_expert.horizon, cfg.action_expert.action_dim)


def test_pi06_advantage_token_changes_context_size() -> None:
    """The advantage token must actually land in the context."""
    cfg = _tiny06_config()
    p = Pi06Policy(cfg)
    B = 1
    common = {
        "images": torch.randn(B, 3, cfg.image_resolution, cfg.image_resolution),
        "state": torch.randn(B, cfg.state_dim),
        "language_ids": torch.randint(0, TINY.vocab_size, (B, 4)),
    }
    ctx_no_adv = p.encode_context(**common)
    ctx_with_adv = p.encode_context(**common, advantage_token=torch.randn(B, cfg.advantage_token_dim))
    assert ctx_with_adv.size(1) == ctx_no_adv.size(1) + 1


# --- π₀.₇ tests ---------------------------------------------------------


def test_pi07_full_predict_chunk_with_all_context() -> None:
    cfg = _tiny07_config()
    p = Pi07Policy(cfg)
    B = 2
    out = p.predict_chunk(
        images=torch.randn(B, 3, cfg.image_resolution, cfg.image_resolution),
        state=torch.randn(B, cfg.state_dim),
        language_ids=torch.randint(0, TINY.vocab_size, (B, 4)),
        advantage_token=torch.randn(B, cfg.advantage_token_dim),
        subgoal_images=torch.randn(B, cfg.n_subgoal_images, 3, cfg.image_resolution, cfg.image_resolution),
        metadata={"speed": Speed.fast, "quality": Quality.precise, "persona": Persona.careful},
    )
    assert out.shape == (B, cfg.action_expert.horizon, cfg.action_expert.action_dim)


def test_pi07_metadata_token_changes_action() -> None:
    """Different metadata buckets should yield different (deterministic) contexts."""
    cfg = _tiny07_config()
    p = Pi07Policy(cfg)
    B = 1
    common = dict(
        images=torch.randn(B, 3, cfg.image_resolution, cfg.image_resolution),
        state=torch.randn(B, cfg.state_dim),
        language_ids=torch.randint(0, TINY.vocab_size, (B, 4)),
    )
    ctx_slow = p.encode_context(**common, metadata={"speed": Speed.slow})
    ctx_fast = p.encode_context(**common, metadata={"speed": Speed.fast})
    # Sizes match; values differ.
    assert ctx_slow.shape == ctx_fast.shape
    assert not torch.allclose(ctx_slow, ctx_fast)


def test_pi07_subgoal_images_add_context_tokens() -> None:
    cfg = _tiny07_config()
    p = Pi07Policy(cfg)
    B = 1
    common = dict(
        images=torch.randn(B, 3, cfg.image_resolution, cfg.image_resolution),
        state=torch.randn(B, cfg.state_dim),
        language_ids=torch.randint(0, TINY.vocab_size, (B, 4)),
    )
    ctx_none = p.encode_context(**common)
    ctx_subgoal = p.encode_context(
        **common,
        subgoal_images=torch.randn(B, 2, 3, cfg.image_resolution, cfg.image_resolution),
    )
    assert ctx_subgoal.size(1) > ctx_none.size(1)


def test_pi07_recall_memory_tokens_from_mem() -> None:
    cfg = _tiny07_config()
    # MEM embedding_dim is independent — recall pads/truncates to policy.hidden_size.
    mem = MEM(MEMConfig(embedding_dim=32))
    p = Pi07Policy(cfg, memory=mem)
    # Empty memory → None.
    assert p.recall_memory_tokens("anything") is None
    # Populate.
    for st in ["pick lettuce", "add salt to bowl", "pour olive oil"]:
        mem.add_subtask_summary(st, frames=[np.zeros((1,))], t_start=0, t_end=1)
    tokens = p.recall_memory_tokens("did I add salt yet?", k=3)
    assert tokens is not None
    # Tokens are projected into the policy's backbone hidden size, not the expert's.
    assert tokens.shape == (1, 3, p.hidden_size)


def test_pi07_consumes_memory_tokens_in_context() -> None:
    cfg = _tiny07_config()
    p = Pi07Policy(cfg)
    B = 1
    common = dict(
        images=torch.randn(B, 3, cfg.image_resolution, cfg.image_resolution),
        state=torch.randn(B, cfg.state_dim),
        language_ids=torch.randint(0, TINY.vocab_size, (B, 4)),
    )
    ctx_none = p.encode_context(**common)
    # Memory tokens are in the *policy* hidden space.
    mem_tokens = torch.randn(B, 4, p.hidden_size)
    ctx_mem = p.encode_context(**common, memory_tokens=mem_tokens)
    assert ctx_mem.size(1) == ctx_none.size(1) + 4
