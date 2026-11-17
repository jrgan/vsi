"""
SingletonEngine — single-point vertex-rep VSI engine.

A lightweight :class:`vsi.VertexRepEngine` variant that maintains a
**single feasible point** per state.  The point-selection rule is
configurable via :class:`PointSelection`:

* ``SOME`` — an arbitrary feasible point (cheapest; an LP feasibility
  solve).
* ``NEAREST`` — the feasible point closest to a target (QP); the
  default, with the target defaulting to the origin (minimum-norm
  point).
* ``SUPPORT`` — the feasible point maximising ``normal @ v`` (an LP);
  the normal defaults to the all-ones vector.

It is non-grid: it ignores the grid spacing both when building initial
sets and when expanding values.
"""
from __future__ import annotations

from enum import Enum, auto
from typing import Generic

import numpy as np

from ._progress import ProgressSink, NullSink
from vsi.value_set_engine import StepOutcomeT, State, Points
from .engine import VertexRepEngine, VertexSets
from .grid import Grid
from .value_set_lp import ValueSetLPFactory


class PointSelection(Enum):
    """
    Rule for choosing the single feasible point in
    :class:`SingletonEngine`.

    * ``SOME`` — an arbitrary feasible point
      (:meth:`vsi.vertex_rep.LPSession.some_point`).
    * ``NEAREST`` — the feasible point closest to
      :attr:`SingletonEngine.target`
      (:meth:`vsi.vertex_rep.LPSession.nearest_point`).
    * ``SUPPORT`` — the feasible point maximising
      ``normal @ v`` along :attr:`SingletonEngine.normal`
      (:meth:`vsi.vertex_rep.LPSession.support_point`).
    """
    SOME = auto()
    NEAREST = auto()
    SUPPORT = auto()


class SingletonEngine(VertexRepEngine[StepOutcomeT], Generic[StepOutcomeT]):
    """
    Vertex-rep VSI engine that maintains a **single feasible point**
    per state, chosen according to a configurable
    :class:`PointSelection` rule.
    """

    _mode_label = "singletons"

    def __init__(
        self,
        lp_factory: ValueSetLPFactory[StepOutcomeT],
        *,
        selection: PointSelection = PointSelection.NEAREST,
        target: np.ndarray | None = None,
        normal: np.ndarray | None = None,
    ) -> None:
        """
        Args:
            lp_factory: :class:`vsi.ValueSetLPFactory` bound to the
                domain problem.  Supplies the per-state LP and
                ``num_states``.
            selection: Point-selection rule (see :class:`PointSelection`).
                Defaults to ``NEAREST``.
            target: Target point for ``selection == NEAREST`` (the point
                projected onto the feasible set).  Defaults to the zero
                vector (minimum-norm point).  Ignored by other rules.
            normal: Direction for ``selection == SUPPORT`` (the point
                maximising ``normal @ v``).  Defaults to the all-ones
                vector.  Ignored by other rules.
        """
        # This engine is grid-free: it overrides _compute_value_set and
        # never reads the grid.  Supply a valid dummy grid to initialise
        # the shared vertex-set machinery, then explicitly disable the
        # grid-based expansion slack.
        super().__init__(
            lp_factory, Grid(dim=lp_factory.reward_dim, grid_spacing=1.0),
        )
        self._expansion_slack = 0.0

        d = self.reward_dim
        self.selection = selection
        #: Target point for ``NEAREST`` (default: origin / minimum norm).
        self.target = (
            np.zeros(d) if target is None else np.asarray(target, dtype=float)
        )
        #: Direction for ``SUPPORT`` (default: all-ones vector).
        self.normal = (
            np.ones(d) if normal is None else np.asarray(normal, dtype=float)
        )
        if self.target.shape != (d,):
            raise ValueError(
                f"target must have shape ({d},), got {self.target.shape}"
            )
        if self.normal.shape != (d,):
            raise ValueError(
                f"normal must have shape ({d},), got {self.normal.shape}"
            )

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
        """Compute a single feasible point for state *s0*."""
        d = self.reward_dim
        sink.set_label("solving")
        sink.set_total(1)

        with lp_factory.open_lp(
            s0, value_sets,
        ) as lp:
            if not lp.feasible:
                return np.empty((0, d))
            match self.selection:
                case PointSelection.SOME:
                    point = lp.some_point()
                case PointSelection.NEAREST:
                    point = lp.nearest_point(self.target)
                case PointSelection.SUPPORT:
                    point = lp.support_point(self.normal)
                case _:
                    raise ValueError(
                        f"Unsupported selection: {self.selection!r}"
                    )
            sink.advance()
            if point is None:
                return np.empty((0, d))
            return point.reshape(1, d)
