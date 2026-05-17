"""Real-Time Chunking (RTC) — async inpainting for low-latency VLA control (Jun 2025).

Instead of waiting for chunk *N* to finish executing before requesting chunk
*N+1*, RTC starts inferring *N+1* in a background thread *while* *N*
executes. The new chunk is generated as a flow-matching **inpainting**
problem: the prefix that will overlap with the still-executing actions is
held fixed, and only the tail is sampled. Tolerates inference latency
> 300 ms with no visible pauses.

Paper: papers/2025-06-09_rtc_real-time-action-chunking-large-models.pdf
Follow-ups:
    papers/cited/2025-06-09_black_real-time-action-chunking-flow-policies.pdf
    papers/cited/2025-12-05_black_action-conditioning-real-time-chunking.pdf

This module is policy-agnostic. The caller supplies a `sample_chunk`
callable; the runner manages the asynchronous loop.

Algorithm 1 from the paper (paraphrased):

    current_chunk = sample_chunk(obs_0, prefix=None)       # bootstrap, sync
    while task not done:
        # Background: prepare the next chunk a few steps before we need it.
        when (chunk_horizon - idx) <= overlap_steps and no worker is running:
            prefix = current_chunk[idx : idx + overlap_steps]    # what's still executing
            spawn worker: next_chunk = sample_chunk(obs_now, prefix=prefix)
        # Foreground: execute one step.
        action = current_chunk[idx]; idx += 1
        # If we finished the current chunk, swap in the next one (waiting if needed).
        if idx >= chunk_horizon:
            if worker still running: join (graceful — paper allows this stall)
            current_chunk = next_chunk; idx = 0
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np


@dataclass
class RTCConfig:
    chunk_horizon: int = 50
    overlap_steps: int = 10           # prefix length held fixed for inpainting
    control_hz: int = 50
    spawn_threshold: int | None = None
    """Schedule the next chunk when ``chunk_horizon - idx <= spawn_threshold``.
    Defaults to ``overlap_steps`` if None — matches the paper's recipe."""


SampleChunk = Callable[..., np.ndarray]
"""Callable signature: ``sample_chunk(obs, *, prefix: np.ndarray | None) -> (H, D) array``.

The ``prefix`` argument is the actions that the runner has committed to
executing while inference happens — the sampler must produce a chunk whose
first ``len(prefix)`` actions equal ``prefix`` (or are close enough that
the discontinuity is below the action-noise floor). The paper implements
this as flow-matching inpainting; for our scaffolding, the policy implementor
is responsible for honoring the prefix contract.
"""


class RTCRunner:
    """Asynchronous action runner implementing Algorithm 1.

    Single inflight worker — sufficient for the paper's latency targets and
    avoids the queue-management complexity that more workers would add.

    Typical use::

        runner = RTCRunner(RTCConfig(), sample_chunk=my_policy.sample_chunk)
        obs = env.reset()
        for _ in range(steps):
            action = runner.step(obs)
            obs, _, done, _ = env.step(action)
        runner.close()
    """

    def __init__(self, config: RTCConfig, sample_chunk: SampleChunk) -> None:
        self.config = config
        self._sample_chunk = sample_chunk
        self._current_chunk: Optional[np.ndarray] = None
        self._next_chunk: Optional[np.ndarray] = None
        self._idx = 0
        self._lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._worker_failed: BaseException | None = None
        # Stats for introspection / tests.
        self.stats = {
            "chunks_sampled": 0,
            "chunks_async": 0,
            "chunks_sync_fallback": 0,
            "swaps_blocked_on_worker": 0,
        }

    @property
    def _spawn_threshold(self) -> int:
        return self.config.spawn_threshold or self.config.overlap_steps

    def _spawn_worker(self, obs: Any) -> None:
        """Kick off background sampling for the next chunk."""
        assert self._current_chunk is not None
        prefix_end = min(self._idx + self.config.overlap_steps, self.config.chunk_horizon)
        prefix = np.array(self._current_chunk[self._idx : prefix_end], copy=True)

        def _run() -> None:
            try:
                new = self._sample_chunk(obs, prefix=prefix)
            except BaseException as exc:  # noqa: BLE001 — surface to main thread
                self._worker_failed = exc
                return
            with self._lock:
                self._next_chunk = np.asarray(new)
                self.stats["chunks_sampled"] += 1
                self.stats["chunks_async"] += 1

        self._worker = threading.Thread(target=_run, daemon=True, name="rtc-worker")
        self._worker.start()

    def _swap(self, obs: Any) -> None:
        """Move to the next chunk, joining the worker if it hasn't finished."""
        if self._worker is not None:
            if self._worker.is_alive():
                self.stats["swaps_blocked_on_worker"] += 1
            self._worker.join()
            self._worker = None
            if self._worker_failed is not None:
                err, self._worker_failed = self._worker_failed, None
                raise err
        with self._lock:
            if self._next_chunk is not None:
                self._current_chunk = self._next_chunk
                self._next_chunk = None
            else:
                # Worker never spawned (chunk was too short, or first call) —
                # fall back to a synchronous sample. This is the only blocking
                # path; it shouldn't happen during steady-state operation.
                self._current_chunk = np.asarray(
                    self._sample_chunk(obs, prefix=None)
                )
                self.stats["chunks_sampled"] += 1
                self.stats["chunks_sync_fallback"] += 1
        self._idx = 0

    def step(self, obs: Any) -> np.ndarray:
        """Return the next action.

        Bootstraps synchronously on the first call (we need *some* chunk
        before we can stream). Subsequent calls return in microseconds —
        all the work happens in the background.
        """
        # First call — synchronously produce the bootstrap chunk.
        if self._current_chunk is None:
            self._current_chunk = np.asarray(self._sample_chunk(obs, prefix=None))
            self.stats["chunks_sampled"] += 1
            self.stats["chunks_sync_fallback"] += 1
            self._idx = 0

        # Schedule the next chunk if we're close to the boundary.
        steps_remaining = self.config.chunk_horizon - self._idx
        if (
            steps_remaining <= self._spawn_threshold
            and self._worker is None
            and self._next_chunk is None
        ):
            self._spawn_worker(obs)

        # Emit one action from the current chunk.
        action = np.array(self._current_chunk[self._idx], copy=True)
        self._idx += 1

        # End of chunk — promote the next one.
        if self._idx >= self.config.chunk_horizon:
            self._swap(obs)

        return action

    def close(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=1.0)
            self._worker = None
        # If the worker raised, surface it on close as well.
        if self._worker_failed is not None:
            err, self._worker_failed = self._worker_failed, None
            raise err
