"""Tests for the MEM dual-memory store."""

from __future__ import annotations

import numpy as np
import pytest

from pi_stack.memory.mem import (
    MEM,
    LongTermLanguageMemory,
    MEMConfig,
    ShortTermVideoMemory,
    default_embedder,
)


def _fake_frame(seed: int) -> np.ndarray:
    return np.full((4, 4, 3), seed, dtype=np.uint8)


# --- Short-term ---------------------------------------------------------


def test_short_term_downsamples_to_fps() -> None:
    cfg = MEMConfig(short_term_video_seconds=2.0, short_term_fps=4)
    stm = ShortTermVideoMemory(cfg)
    # Push at 100 Hz; only ~1 in 25 frames should land.
    for i in range(200):
        stm.add(_fake_frame(i), t=i * 0.01)
    # 2 seconds @ 4 fps = 8 frames max.
    assert len(stm) == 8


def test_short_term_evicts_old_frames_fifo() -> None:
    cfg = MEMConfig(short_term_video_seconds=1.0, short_term_fps=4)
    stm = ShortTermVideoMemory(cfg)
    for i in range(20):
        stm.add(_fake_frame(i), t=i * 0.25)
    assert len(stm) == 4   # 1 sec * 4 fps
    timestamps = stm.timestamps()
    # The 4 newest frames span the last second.
    assert timestamps.max() == pytest.approx(19 * 0.25)
    assert timestamps.min() == pytest.approx(16 * 0.25)


def test_short_term_clear() -> None:
    cfg = MEMConfig(short_term_video_seconds=1.0, short_term_fps=2)
    stm = ShortTermVideoMemory(cfg)
    stm.add(_fake_frame(0), t=0.0)
    stm.clear()
    assert len(stm) == 0
    assert stm.recall().shape == (0,)


# --- Long-term ----------------------------------------------------------


def test_long_term_recall_orders_by_relevance() -> None:
    cfg = MEMConfig(embedding_dim=64)
    ltm = LongTermLanguageMemory(cfg)
    # Add summaries with disjoint vocabularies so the hash embedder can rank them.
    ltm.add_subtask("salt", frames=[], t_start=0.0, t_end=1.0)
    ltm.add_subtask("water boiling", frames=[], t_start=1.0, t_end=2.0)
    ltm.add_subtask("plate clean", frames=[], t_start=2.0, t_end=3.0)
    top = ltm.recall("did I already add salt?", k=1)
    assert len(top) == 1
    assert top[0].subtask == "salt"


def test_long_term_fifo_eviction() -> None:
    cfg = MEMConfig(long_term_max_summaries=3)
    ltm = LongTermLanguageMemory(cfg)
    for i in range(5):
        ltm.add_subtask(f"task-{i}", frames=[], t_start=i, t_end=i + 1)
    assert len(ltm) == 3
    # Oldest two should have been evicted; the remaining are 2, 3, 4.
    subtasks = [e.subtask for e in ltm.entries()]
    assert subtasks == ["task-2", "task-3", "task-4"]


def test_default_embedder_is_unit_norm_for_nonempty_text() -> None:
    v = default_embedder("hello world", dim=64)
    assert v.shape == (64,)
    assert v.dtype == np.float32
    assert np.linalg.norm(v) == pytest.approx(1.0)


def test_default_embedder_empty_text_returns_zero() -> None:
    v = default_embedder("", dim=32)
    assert np.allclose(v, 0.0)


def test_default_embedder_is_deterministic() -> None:
    a = default_embedder("the quick brown fox", dim=32)
    b = default_embedder("the quick brown fox", dim=32)
    np.testing.assert_array_equal(a, b)


def test_long_term_with_custom_summarizer() -> None:
    cfg = MEMConfig()
    seen = []

    def fake_summarizer(subtask: str, frames: list[np.ndarray]) -> str:
        seen.append((subtask, len(frames)))
        return f"FAKE: {subtask} ({len(frames)} frames)"

    ltm = LongTermLanguageMemory(cfg, summarizer=fake_summarizer)
    frames = [_fake_frame(i) for i in range(3)]
    entry = ltm.add_subtask("pick lettuce", frames, t_start=0.0, t_end=1.0)
    assert entry.summary == "FAKE: pick lettuce (3 frames)"
    assert seen == [("pick lettuce", 3)]


# --- Facade -------------------------------------------------------------


def test_mem_facade_routes_frames_and_summaries() -> None:
    mem = MEM(MEMConfig(short_term_video_seconds=2.0, short_term_fps=4))
    for i in range(40):
        mem.add_frame(_fake_frame(i), t=i * 0.05)
    assert len(mem.short_term) <= 8   # capped at 2s * 4fps

    mem.add_subtask_summary("turn on stove", t_start=0.0, t_end=2.0)
    assert len(mem.long_term) == 1
    # add_subtask_summary defaulted frames= to the short-term buffer.
    entry = mem.long_term.entries()[0]
    assert entry.subtask == "turn on stove"


def test_mem_reset_clears_both_stores() -> None:
    mem = MEM(MEMConfig())
    mem.add_frame(_fake_frame(0), t=0.0)
    mem.add_subtask_summary("a", frames=[_fake_frame(0)], t_start=0.0, t_end=1.0)
    mem.reset()
    assert len(mem.short_term) == 0
    assert len(mem.long_term) == 0
