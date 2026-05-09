"""Real-Time Chunking (RTC) — async inpainting for low-latency VLA control (Jun 2025).

Instead of waiting for chunk N to finish executing before requesting chunk N+1,
RTC starts inferring N+1 in a background thread *while* N executes. The new
chunk is generated as a flow-matching inpainting problem: the prefix that
will overlap with the still-executing actions is held fixed, and only the
tail is sampled. Tolerates inference latency >300 ms with no visible pauses.

Paper: papers/2025-06-09_rtc_real-time-action-chunking-large-models.pdf
Follow-up: papers/cited/2025-06-09_black_real-time-action-chunking-flow-policies.pdf
           papers/cited/2025-12-05_black_action-conditioning-real-time-chunking.pdf

Algorithm 1 from the paper (paraphrased):

    while task not done:
        spawn background: next_chunk = policy.sample_chunk(
            obs=current_obs,
            inpaint_prefix=remaining_executing_actions,
        )
        execute current_chunk
        when next_chunk arrives: swap it in
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class RTCConfig:
    chunk_horizon: int = 50
    overlap_steps: int = 10        # prefix length held fixed during inpainting
    max_inflight: int = 1
    control_hz: int = 50


class RTCRunner:
    """Skeleton async runner for RTC.

    The runner is policy-agnostic: it takes a callable that produces the next
    chunk given the current observation and an optional fixed prefix.

    TODO:
      - implement Algorithm 1 with a single background worker
      - implement inpainting-aware sampler hook on the action expert
        (`flow_matching.euler_sample(..., inpaint_prefix=...)`)
      - handle dropped/late chunks gracefully (fall back to repeating last chunk)
    """

    def __init__(
        self,
        config: RTCConfig,
        sample_chunk: Callable[..., Any],
    ) -> None:
        self.config = config
        self._sample_chunk = sample_chunk
        self._lock = threading.Lock()
        self._next_chunk: Any | None = None

    def step(self, *args, **kwargs):
        raise NotImplementedError
