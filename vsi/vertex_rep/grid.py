"""
Grid for vertex-representation VSI engines.

Defines :class:`Grid`, the value-space grid a
:class:`vsi.VertexRepEngine` operates on.  In the paper this is an
*infinite* grid of points ``grid_spacing * k`` for integer index
vectors ``k`` (so the grid always passes through the origin).  The
``k_min`` / ``k_max`` fields are a *convenience clip*: optional
per-dimension integer index bounds that confine ``k`` to
``k_min[i] <= k[i] <= k_max[i]``, with a ``None`` bound leaving that
side unbounded.  The state count ``num_states`` is *not* part of this
bundle; it lives as a separate engine attribute.

A fully clipped grid (both bounds set) is the common case: it is the
only one the grid-enumeration baseline supports, and the one the
caller-facing seed helpers in :mod:`vsi.vertex_rep.initial_sets` read
their bounds from.  Half-clipped or unclipped grids require
caller-supplied initial value sets.

Each domain provides a small helper (e.g. ``markovgame_grid``,
``scmdp_grid``) that builds a clipped :class:`Grid` from a problem
instance and a target accuracy, rounding the value box outward to
integer grid indices itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from vsi._validation import as_positive_float, as_positive_int


@dataclass(frozen=True, kw_only=True)
class Grid:
    """
    Value-space grid for a vertex-rep VSI engine.

    The grid points are ``grid_spacing * k`` for integer index vectors
    ``k`` — conceptually an infinite grid through the origin.  Optional
    per-dimension index bounds ``k_min`` and ``k_max`` are a convenience
    clip confining the indices to ``k_min[i] .. k_max[i]``; a ``None``
    bound leaves that side unbounded.

    Integer indices give exact cell counts without having to infer grid
    alignment from floating-point bounds.

    Three regimes:

    * **Bounded** (both ``k_min`` and ``k_max`` set) — a finite box.
      Required by the grid-enumeration baseline, and the regime the
      bounding-box seed helpers read their bounds from.
    * **Half-bounded** (one side set) — e.g. values bounded below only.
    * **Unbounded** (neither set) — the full infinite grid.  The MILP
      integer hull works directly on it; initial value sets must be
      supplied by the caller (``IterationRunner.run(init_value_sets=...)``)
      since there is no box to seed from.

    Attributes:
        dim: Reward dimension ``d``.  Needed to materialise the grid
            even when it is unbounded.
        grid_spacing: Finite, strictly positive side length of a grid cell.
        k_min: Optional integer lower index bound per dimension, shape
            ``(d,)``.  ``None`` leaves the grid unbounded below.
        k_max: Optional integer upper index bound per dimension, shape
            ``(d,)``.  ``None`` leaves the grid unbounded above.
    """
    dim: int
    grid_spacing: float
    k_min: npt.NDArray[np.int64] | None = None
    k_max: npt.NDArray[np.int64] | None = None

    def __post_init__(self) -> None:
        """Validate and normalise the grid definition."""
        dim = as_positive_int(self.dim, "dim")
        grid_spacing = as_positive_float(self.grid_spacing, "grid_spacing")

        def normalise_bound(name, bound):
            if bound is None:
                return None
            values = np.asarray(bound)
            if values.shape != (dim,):
                raise ValueError(
                    f"{name} must have shape ({dim},), got {values.shape}"
                )
            if not np.issubdtype(values.dtype, np.integer):
                raise ValueError(f"{name} must contain integers")
            # Reject oversized integers before int64 conversion can wrap them.
            limits = np.iinfo(np.int64)
            if (values < limits.min).any() or (values > limits.max).any():
                raise ValueError(f"{name} must fit in int64")
            return values.astype(np.int64, copy=True)

        k_min = normalise_bound("k_min", self.k_min)
        k_max = normalise_bound("k_max", self.k_max)
        if k_min is not None and k_max is not None and (k_min > k_max).any():
            raise ValueError("k_min must be less than or equal to k_max")

        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "grid_spacing", grid_spacing)
        object.__setattr__(self, "k_min", k_min)
        object.__setattr__(self, "k_max", k_max)

    @property
    def is_lower_bounded(self) -> bool:
        """Whether a finite lower index bound ``k_min`` is set."""
        return self.k_min is not None

    @property
    def is_upper_bounded(self) -> bool:
        """Whether a finite upper index bound ``k_max`` is set."""
        return self.k_max is not None

    @property
    def is_bounded(self) -> bool:
        """Whether both index bounds are set (a finite box)."""
        return self.k_min is not None and self.k_max is not None

    @property
    def lbs(self) -> np.ndarray:
        """Lower corner ``k_min * grid_spacing``.

        Defined only when the grid is bounded below.
        """
        if self.k_min is None:
            raise AttributeError(
                "lbs is undefined: grid is unbounded below (k_min is None)."
            )
        return self.k_min * self.grid_spacing

    @property
    def ubs(self) -> np.ndarray:
        """Upper corner ``k_max * grid_spacing``.

        Defined only when the grid is bounded above.  All consumers
        evaluate grid coordinates as ``grid_spacing * k``, so this
        coincides bit-for-bit with the top grid point used elsewhere.
        """
        if self.k_max is None:
            raise AttributeError(
                "ubs is undefined: grid is unbounded above (k_max is None)."
            )
        return self.k_max * self.grid_spacing

    @property
    def n_cells(self) -> np.ndarray:
        """Integer cell count ``k_max - k_min`` per dimension.

        Defined only when the grid is fully bounded.
        """
        if self.k_min is None or self.k_max is None:
            raise AttributeError(
                "n_cells is undefined: grid is not fully bounded."
            )
        return self.k_max - self.k_min
