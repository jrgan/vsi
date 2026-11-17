"""
GridEnumEngine — baseline VSI engine enumerating all grid points.

This engine filters every grid point by LP achievability at each
state.  It is intended **as a baseline comparison only**; expect poor
scaling in ``reward_dim`` and ``1 / grid_spacing``.  Prefer
:class:`vsi.VertexRepEngine` for any non-trivial problem.
"""
from __future__ import annotations

import warnings
from typing import Generic

import numpy as np

from .._progress import ProgressSink, NullSink
from vsi.value_set_engine import StepOutcomeT, State, Points
from ..engine import VertexRepEngine, VertexSets
from ..grid import Grid
from ..value_set_lp import ValueSetLPFactory


def box_mesh(grid: Grid) -> np.ndarray:
    """Return every grid point of the clipped box ``[grid.lbs, grid.ubs]``.

    Shape ``(prod(n_cells + 1), d)``.  This is the candidate universe the
    grid-enumeration baseline sweeps — operator machinery, not a seed
    helper — so it lives with the only engine that uses it.  Requires a
    fully bounded :class:`vsi.Grid`.
    """
    if not grid.is_bounded:
        raise ValueError(
            "box_mesh requires a fully bounded Grid (both k_min and k_max "
            "set): it enumerates every grid point, which is only defined "
            "on a finite box."
        )
    d = grid.dim
    lbs = grid.lbs
    grid_spacing = grid.grid_spacing
    # Grid stores the integer cell count directly, so each axis has
    # exactly n_cells[i] + 1 grid points from lbs[i] to ubs[i].
    n_cells = grid.n_cells
    coords = [
        lbs[i] + grid_spacing * np.arange(n_cells[i] + 1)
        for i in range(d)
    ]
    mesh = np.stack(np.meshgrid(*coords, indexing="ij"), axis=-1)
    return mesh.reshape(-1, d)


class GridEnumEngine(VertexRepEngine[StepOutcomeT], Generic[StepOutcomeT]):
    """
    Vertex-rep VSI engine that enumerates every grid point and keeps
    only those that are LP-achievable.

    Baseline-only.  See module docstring.
    """

    _mode_label = "grid points"

    def __init__(
        self,
        lp_factory: ValueSetLPFactory[StepOutcomeT],
        grid: Grid,
        *,
        nested_iteration: bool = True,
    ) -> None:
        """
        Args:
            lp_factory: :class:`vsi.ValueSetLPFactory` bound to the
                domain problem.  Supplies the per-state LP and
                ``num_states``.
            grid: Value-space :class:`vsi.Grid`.
            nested_iteration: If True, candidate points at each
                iteration come from the current value sets; otherwise
                they come from the full initial grid mesh.
        """
        warnings.warn(
            "GridEnumEngine enumerates every grid point via LP and is "
            "intended as a baseline for comparison with VertexRepEngine / "
            "SingletonEngine.  Expect poor scaling in reward_dim and "
            "1/grid_spacing.",
            UserWarning,
            stacklevel=2,
        )
        super().__init__(lp_factory, grid)
        if not grid.is_bounded:
            raise ValueError(
                "GridEnumEngine requires a fully bounded Grid "
                "(both k_min and k_max set): it enumerates every grid "
                "point, which is only defined on a finite box."
            )
        self.nested_iteration = nested_iteration

    # ------------------------------------------------------------------
    # Per-state compute
    # ------------------------------------------------------------------

    def _compute_value_set(
        self,
        s0: State,
        value_sets: VertexSets,
        sink: ProgressSink = NullSink(),
        *,
        lp_factory,
    ) -> Points:
        """
        Filter grid points at *s0* by LP achievability.

        Opens a single :class:`LPSession`, adds a persistent
        target box, and re-solves for each candidate point by updating
        only the RHS.

        Candidate points come from *value_sets* when
        ``self.nested_iteration`` is True, otherwise from the full
        initial grid mesh.
        """
        if self.nested_iteration:
            points = value_sets[s0]
        else:
            points = box_mesh(self.grid)
        d = self.reward_dim

        if len(points) == 0:
            return np.empty((0, d))

        sink.set_label("grid points")
        sink.set_total(len(points))

        with lp_factory.open_lp(
            s0, value_sets, slack=self.grid.grid_spacing,
        ) as lp:
            if not lp.feasible:
                return np.empty((0, d))

            achievable = []
            for p in points:
                lp.set_target(p)
                if lp.is_nonempty():
                    achievable.append(p)
                sink.advance()

            if len(achievable) == 0:
                return np.empty((0, d))
            return np.array(achievable)
