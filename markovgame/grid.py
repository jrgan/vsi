"""
Grid helper for Markov-game VSI.
"""
from __future__ import annotations

import numpy as np

from markovgame.markov_game import MarkovGame
from vsi._validation import as_positive_float
from vsi.vertex_rep.grid import Grid


def markovgame_grid(
    game: MarkovGame, *, eps: float, delta: float,
) -> Grid:
    """
    Derive the value-space :class:`Grid` for *game* from a
    target accuracy ``(eps, delta)``.

    Uses ``grid_spacing = (1 - gamma) * min(eps, delta / 2)`` and snaps
    the per-agent value box outward to the enclosing grid lines,
    stepping one extra cell on each face, so the box is described
    entirely by integer grid indices.
    """
    eps = as_positive_float(eps, "eps")
    delta = as_positive_float(delta, "delta")

    gamma = game.gamma
    grid_spacing = (1 - gamma) * min(eps, delta / 2)

    raw_lbs = game.rewards.min(axis=(1, 2)) / (1 - gamma)
    raw_ubs = game.rewards.max(axis=(1, 2)) / (1 - gamma)

    k_min = np.ceil(raw_lbs / grid_spacing).astype(int) - 1
    k_max = np.floor(raw_ubs / grid_spacing).astype(int) + 1

    return Grid(
        dim=raw_lbs.shape[0],
        grid_spacing=grid_spacing,
        k_min=k_min,
        k_max=k_max,
    )
