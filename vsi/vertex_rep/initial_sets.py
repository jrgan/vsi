"""
Caller-facing helpers for constructing VSI initial value sets.

These helpers build the ``init_value_sets`` argument passed to
``IterationRunner.run_iter`` / ``PolicyRunner``.  They are a convenience
for the *caller*: the VSI operator itself (``VertexRepEngine`` and its
variants) never calls them — seeding is the caller's responsibility, not
part of the operator contract.  They live under :mod:`vsi` only so that
examples and experiments can build seeds without depending on the heavier
``experiments`` framework.

The two box helpers take explicit bounds ``(lbs, ubs)`` — a plain pair of
corner vectors — rather than a :class:`vsi.Grid`.  A box is just a bound;
it is independent of the grid the operator runs on.  For a clipped grid,
read the bounds off it with ``grid.lbs`` / ``grid.ubs``:

    sets = uniform_value_sets(box_corners(grid.lbs, grid.ubs), num_states)

:func:`uniform_value_sets` then replicates one point array across states
to form the :class:`VertexSets`.
"""
from __future__ import annotations

import itertools

import numpy as np

from vsi._validation import as_positive_int
from vsi.value_set_engine import Points
from .engine import VertexSets


def _validate_box_bounds(
    lbs: np.ndarray, ubs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert and validate a pair of box corners."""
    lbs = np.asarray(lbs, dtype=float)
    ubs = np.asarray(ubs, dtype=float)
    if lbs.ndim != 1 or ubs.ndim != 1 or lbs.shape != ubs.shape:
        raise ValueError(
            "lbs and ubs must be one-dimensional and have the same shape"
        )
    if len(lbs) == 0:
        raise ValueError("lbs and ubs must not be empty")
    if not np.isfinite(lbs).all() or not np.isfinite(ubs).all():
        raise ValueError("lbs and ubs must contain only finite values")
    if (lbs > ubs).any():
        raise ValueError("lbs must be less than or equal to ubs")
    return lbs, ubs


def box_corners(lbs: np.ndarray, ubs: np.ndarray) -> np.ndarray:
    """Return the ``2**d`` corners of the box ``[lbs, ubs]``.

    Shape ``(2**d, d)``.  The natural seed for the integer-hull engine.
    """
    lbs, ubs = _validate_box_bounds(lbs, ubs)
    return np.array(list(itertools.product(*zip(lbs, ubs))))


def box_centre(lbs: np.ndarray, ubs: np.ndarray) -> np.ndarray:
    """Return the centre of the box ``[lbs, ubs]``.

    Shape ``(1, d)``.  The natural seed for the singleton engine.
    """
    lbs, ubs = _validate_box_bounds(lbs, ubs)
    return np.array([(lbs + ubs) / 2])


def uniform_value_sets(points: Points, num_states: int) -> VertexSets:
    """Build :class:`VertexSets` with the same seed points at every state.

    Replicates one ``(n, d)`` point array across ``num_states`` states
    (a fresh copy per state).  Combined with the box seed helpers
    (:func:`box_corners`, :func:`box_centre`) this builds the initial
    value sets without consulting the engine.
    """
    num_states = as_positive_int(num_states, "num_states")
    points = np.asarray(points, dtype=float)
    is_empty = points.ndim > 0 and points.shape[0] == 0
    if not is_empty and points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (n, d)")
    if not np.isfinite(points).all():
        raise ValueError("points must contain only finite values")
    return VertexSets([points.copy() for _ in range(num_states)])
