"""Tests for the RTC async runner.

The synthetic policy returns a known chunk after a configurable delay so we
can verify:

- The runner never blocks on inference once it's primed (steady state).
- The worker actually runs in a background thread (the `chunks_async` stat).
- The runner falls back to sync sampling only when it should.
- Worker exceptions are surfaced cleanly.
"""

from __future__ import annotations

import time
import threading

import numpy as np
import pytest

from pi_stack.inference.rtc import RTCConfig, RTCRunner


def _make_policy(delay_s: float, action_dim: int, horizon: int):
    """Return a `sample_chunk` callable + a counter of calls."""
    counter = {"calls": 0, "prefix_seen": []}

    def sample(obs, *, prefix=None):
        counter["calls"] += 1
        counter["prefix_seen"].append(None if prefix is None else prefix.shape)
        if delay_s > 0:
            time.sleep(delay_s)
        # Build a chunk so each step holds the call-index — easy to inspect.
        chunk = np.full((horizon, action_dim), float(counter["calls"]), dtype=np.float32)
        # Honor the prefix contract — first len(prefix) actions match.
        if prefix is not None:
            chunk[: len(prefix)] = prefix
        return chunk

    return sample, counter


def test_bootstrap_runs_one_sync_sample() -> None:
    sample, counter = _make_policy(delay_s=0.0, action_dim=4, horizon=10)
    runner = RTCRunner(RTCConfig(chunk_horizon=10, overlap_steps=3), sample)
    runner.step(obs=None)
    assert counter["calls"] == 1
    assert runner.stats["chunks_sync_fallback"] == 1
    runner.close()


def test_runner_emits_chunk_horizon_actions_per_chunk() -> None:
    sample, _ = _make_policy(delay_s=0.0, action_dim=2, horizon=8)
    runner = RTCRunner(RTCConfig(chunk_horizon=8, overlap_steps=2), sample)
    actions = [runner.step(obs=None) for _ in range(8)]
    # Within the first chunk the action value is the call index (1).
    for a in actions:
        np.testing.assert_array_equal(a, np.full(2, 1.0))
    runner.close()


def test_async_worker_overlaps_with_execution() -> None:
    """If chunk_horizon is large enough, the worker finishes before the
    main loop reaches the chunk boundary — the swap should be non-blocking."""
    horizon = 30
    overlap = 5
    # Slack window between spawn (step ~26) and swap (step 30) is
    # ~(horizon - overlap) - 1 ≈ 4 control-loop sleeps. Pick a worker delay
    # well under that so the swap never has to block.
    sample, counter = _make_policy(delay_s=0.002, action_dim=4, horizon=horizon)
    runner = RTCRunner(
        RTCConfig(chunk_horizon=horizon, overlap_steps=overlap),
        sample,
    )
    # Run two full chunks worth of steps.
    for _ in range(horizon * 2):
        runner.step(obs=None)
        time.sleep(0.001)  # 1 ms control loop
    # 1 sync bootstrap + at least 1 async refresh; a 3rd sample may have been
    # spawned near the end of chunk 2 even though it never got consumed.
    assert runner.stats["chunks_sync_fallback"] == 1
    assert runner.stats["chunks_async"] >= 1
    # The property that matters: no swap ever had to block on the worker.
    assert runner.stats["swaps_blocked_on_worker"] == 0
    runner.close()


def test_slow_policy_causes_worker_join_but_no_data_corruption() -> None:
    """If the policy is slower than the chunk window, the runner must
    block on swap (acknowledged in the paper as a graceful degradation),
    but still produce coherent actions."""
    horizon = 6
    overlap = 2
    # Worker takes 200 ms — longer than horizon * 1 ms control loop.
    sample, _ = _make_policy(delay_s=0.2, action_dim=3, horizon=horizon)
    runner = RTCRunner(
        RTCConfig(chunk_horizon=horizon, overlap_steps=overlap),
        sample,
    )
    actions = []
    for _ in range(horizon * 2):
        actions.append(runner.step(obs=None))
        time.sleep(0.001)
    assert len(actions) == horizon * 2
    # Should have blocked on the worker once on the chunk-1 → chunk-2 swap.
    assert runner.stats["swaps_blocked_on_worker"] >= 1
    runner.close()


def test_prefix_contract_is_forwarded() -> None:
    """The runner must pass a non-None prefix on every non-bootstrap call."""
    horizon = 10
    overlap = 3
    sample, counter = _make_policy(delay_s=0.0, action_dim=2, horizon=horizon)
    runner = RTCRunner(
        RTCConfig(chunk_horizon=horizon, overlap_steps=overlap), sample
    )
    for _ in range(horizon * 3):
        runner.step(obs=None)
    # Bootstrap had prefix=None; subsequent calls all carry the prefix shape.
    assert counter["prefix_seen"][0] is None
    for shape in counter["prefix_seen"][1:]:
        assert shape == (overlap, 2), f"expected overlap prefix, got {shape}"
    runner.close()


def test_worker_exception_is_surfaced() -> None:
    horizon = 6
    state = {"calls": 0}

    def flaky_sample(obs, *, prefix=None):
        state["calls"] += 1
        if state["calls"] == 1:
            return np.zeros((horizon, 2), dtype=np.float32)
        raise RuntimeError("policy exploded")

    runner = RTCRunner(
        RTCConfig(chunk_horizon=horizon, overlap_steps=2), flaky_sample
    )
    # Run through the first chunk; worker fires near the end and crashes.
    with pytest.raises(RuntimeError, match="policy exploded"):
        for _ in range(horizon + 1):
            runner.step(obs=None)
    runner.close()


def test_worker_runs_off_main_thread() -> None:
    """Direct verification that the sample callable lands on a non-main thread."""
    horizon = 8
    main_tid = threading.get_ident()
    seen = {"async": False, "sync": False}

    def sample(obs, *, prefix=None):
        tid = threading.get_ident()
        if tid == main_tid:
            seen["sync"] = True
        else:
            seen["async"] = True
        return np.zeros((horizon, 2), dtype=np.float32)

    runner = RTCRunner(
        RTCConfig(chunk_horizon=horizon, overlap_steps=2), sample
    )
    for _ in range(horizon * 2):
        runner.step(obs=None)
    runner.close()
    assert seen["sync"]    # bootstrap
    assert seen["async"]   # at least one chunk refresh
