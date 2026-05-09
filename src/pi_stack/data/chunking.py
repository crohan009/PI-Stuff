"""Action chunking — slice trajectories into H-step windows for VLA training.

Used everywhere in the arc; horizon H ≈ 50 is the canonical choice.
"""

from __future__ import annotations


def chunk_trajectory(actions, horizon: int = 50, stride: int = 1):
    """Yield overlapping (H, action_dim) windows.

    TODO: implement once a numpy/jax array contract is decided.
    """
    raise NotImplementedError
