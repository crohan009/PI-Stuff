"""Synthetic data generation for Hi Robot.

Uses a "Data-Generator VLM" to segment teleoperated demonstrations into
atomic subtasks (e.g., "pick up piece of lettuce"), turning expensive
human teleop into cheap (image, language) pairs that train System 2
without manual labeling.

Paper: papers/2025-02-26_hi-robot_listen-and-think-harder.pdf
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SyntheticGenConfig:
    generator_model: str = "claude-opus-4-7"   # any capable VLM works
    min_subtask_length_s: float = 0.5
    max_subtask_length_s: float = 8.0


def segment_demonstration(*args, **kwargs):
    """Run a VLM over a demo and emit a list of (start_t, end_t, subtask) tuples.

    TODO: prompt the VLM with sampled keyframes; parse JSON output.
    """
    raise NotImplementedError
