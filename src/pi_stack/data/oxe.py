"""Open X-Embodiment (OXE) — cross-embodiment dataset loader.

OXE is a dataset, not a simulator: it's an aggregation of teleoperation
demonstrations across 22+ robot embodiments, formatted as RLDS / TFDS. PI
researchers use it for cross-embodiment training and held-out generalization
evaluation.

Repo / docs: https://github.com/google-deepmind/open_x_embodiment

Each OXE episode carries an embodiment tag. The π series handles cross-
embodiment by exposing the embodiment as part of the input — this loader
preserves that tag so downstream code can condition on it.

Install: see docs/sim-setup.md (TFDS + tensorflow-cpu, optional dep).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


# Subset commonly used in PI co-training mixes (the full OXE has 70+).
DEFAULT_EMBODIMENTS = (
    "fractal20220817_data",       # RT-1 / Everyday Robots
    "kuka",
    "bridge",
    "taco_play",
    "jaco_play",
    "berkeley_cable_routing",
    "roboturk",
    "viola",
    "berkeley_autolab_ur5",
    "language_table",
    "stanford_hydra_dataset_converted_externally_to_rlds",
)


@dataclass
class OXEConfig:
    embodiments: tuple[str, ...] = DEFAULT_EMBODIMENTS
    image_resolution: int = 224
    horizon: int = 50
    shuffle_buffer: int = 10_000
    held_out_for_eval: tuple[str, ...] = ()  # embodiments excluded from train


@dataclass
class OXEEpisode:
    """One episode with embodiment metadata preserved."""

    embodiment: str
    frames: object        # numpy array of shape (T, H, W, 3) — typed when impl lands
    actions: object       # numpy array of shape (T, action_dim)
    language: str
    extras: dict[str, object]


def load_oxe(config: OXEConfig | None = None) -> Iterator[OXEEpisode]:
    """Stream OXE episodes from TFDS, filtered by config.

    TODO:
      - lazy-import tensorflow_datasets / tensorflow_cpu
      - resolve gs://gresearch/robotics/<embodiment> URIs
      - emit OXEEpisode with embodiment tag preserved
    """
    raise NotImplementedError


def load_held_out(config: OXEConfig) -> Iterator[OXEEpisode]:
    """Iterate the held-out embodiments for cross-embodiment evaluation."""
    raise NotImplementedError
