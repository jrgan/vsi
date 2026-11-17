"""
Generic vertex-representation VSI engine.

Provides a problem-agnostic implementation of the VSI operator for
value sets stored as vertex sets (V-representation) on a grid.

:class:`VertexRepEngine` is the concrete, high-fidelity engine: at
each state it computes the **integer hull** of the achievable-value
polytope via iterative MILP facet separation.  It is constructed from
two orthogonal ingredients:

* ``lp_factory`` — a :class:`vsi.ValueSetLPFactory` bound to the
  domain problem instance (e.g. ``MarkovGame``, ``SCMDP``); it opens
  per-state optimization sessions and reports ``num_states``.
* ``grid`` — a :class:`vsi.Grid` carrying the
  value-space grid (grid spacing, optional index bounds).

The state count and reward dimension are both taken from
``lp_factory`` (``num_states`` and ``reward_dim``); the grid dim
must agree.

Two lighter variants subclass it:

* :class:`vsi.SingletonEngine` — a single feasible point per state.
* :class:`vsi.baselines.GridEnumEngine` — enumerates all grid points
  (baseline comparison only).
"""
from __future__ import annotations

import itertools
import logging
import operator
from collections.abc import Iterable
from dataclasses import dataclass, fields
from typing import Generic

import numpy as np
import scipy.spatial as sp

from vsi._validation import as_nonnegative_float, as_positive_int
from .polytope import (
    polytope_distance,
    convex_hull_vertices,
)
from ._execution import map_states
from ._progress import ProgressSink, NullSink
from vsi.value_set_engine import (
    ValueSetEngine, ExpansionResult,
    StepOutcomeT, State, Points,
)
from .grid import Grid
from .value_set_lp import (
    ValueSetLPFactory,
)

_module_logger = logging.getLogger(__name__)


# ============================================================
# VertexSets — vertex-set representation
# ============================================================

class VertexSets:
    """
    Value sets stored as lists of *vertices* (V-representation).

    Each ``_sets[s]`` is an ``(n_s, d)`` array whose rows are the
    points of the achievable value set for state *s*.

    The interpretation of the points (integer-hull vertices, a single
    feasible point, or enumerated grid points) is determined by the
    engine that produced them.
    """

    def __init__(self, sets: Iterable[Points]) -> None:
        self._sets: list[Points] = list(sets)

    def __getitem__(self, state: State) -> Points:
        return self._sets[state]

    def __len__(self) -> int:
        return len(self._sets)

    def __iter__(self):
        return iter(self._sets)

    def __repr__(self) -> str:
        sizes = [len(s) for s in self._sets]
        return f"VertexSets(sizes={sizes})"


# ============================================================
# Parameters
# ============================================================

@dataclass
class VertexRepEngineParams:
    """Tunable parameters for :class:`VertexRepEngine`."""

    # Tolerance for the facet separation check in integer-hull computation.
    # A new vertex is accepted when the MILP objective exceeds the hull
    # boundary by more than this value.
    facet_sep_tol: float = 1e-6

    # Parallel execution across states for iterate_once.
    vsi_operator_parallel: bool = False
    max_workers: int | None = None  # Thread pool size; None = min(num_states, cpu_count)
    show_progress: bool = True     # Display per-state progress bars.

    def set_param(
        self, name: str | None = None, value=None, **kwargs,
    ) -> None:
        """Set one or more validated parameters."""
        param_names = {field.name for field in fields(self)}

        updates = {}
        if name is not None:
            updates[name] = value
        updates.update(kwargs)

        validated = {}
        for key, param_value in updates.items():
            if key not in param_names:
                raise AttributeError(f"Unknown parameter: {key!r}")

            if key == "facet_sep_tol":
                param_value = as_nonnegative_float(param_value, key)

            elif key == "max_workers" and param_value is not None:
                param_value = as_positive_int(param_value, key)

            validated[key] = param_value

        for key, param_value in validated.items():
            setattr(self, key, param_value)


# ============================================================
# VertexRepEngine
# ============================================================

class VertexRepEngine(
    ValueSetEngine[VertexSets, StepOutcomeT],
    Generic[StepOutcomeT]
):
    """
    Vertex-representation VSI engine.

    At each state it computes the **integer hull** of the
    achievable-value polytope via iterative MILP facet separation.
    This is the recommended high-fidelity strategy and the concrete
    default of the vertex-rep family.

    A :class:`VertexRepEngine` is constructed from two orthogonal
    ingredients (a :class:`vsi.ValueSetLPFactory` bound to the domain
    problem, and a :class:`vsi.Grid`) plus algorithm and execution settings.
    The state count and reward dimension are both
    taken from the factory; the grid dim must agree.

    Lighter variants override :meth:`_compute_value_set` (and may set a
    different :attr:`_expansion_slack`): :class:`vsi.SingletonEngine` and
    :class:`vsi.baselines.GridEnumEngine`.
    """

    #: Human-readable label of the per-state computation produced by
    #: :meth:`_compute_value_set`.  Used for log messages.
    _mode_label: str = "integer hulls"

    def __init__(
        self,
        lp_factory: ValueSetLPFactory[StepOutcomeT],
        grid: Grid,
    ):
        """
        Initialise the vertex-rep VSI engine.

        Args:
            lp_factory: :class:`vsi.ValueSetLPFactory` bound to the
                domain problem (e.g. ``MarkovGame``, ``SCMDP``).
                Supplies per-state sessions and ``num_states``.
            grid: Value-space :class:`vsi.Grid`.
        """
        self.lp_factory = lp_factory
        self.grid = grid
        self.num_states = lp_factory.num_states
        #: Reward dimension ``d``, taken from the problem via the
        #: factory.  The grid must discretise the same space.
        self.reward_dim = lp_factory.reward_dim
        if grid.dim != self.reward_dim:
            raise ValueError(
                f"Grid dimension {grid.dim} does not match the problem "
                f"reward dimension {self.reward_dim}."
            )
        #: Coupling slack passed to the per-state value-set LP in
        #: :meth:`expand_value` (default: the grid spacing, so a value
        #: counts as achievable if some exact point lies within one
        #: grid cell).  Non-grid variants (e.g. singletons) set ``0``.
        self._expansion_slack = grid.grid_spacing

        self._logger = _module_logger
        self.params = VertexRepEngineParams()

    # ------------------------------------------------------------------
    # Value-set iteration
    # ------------------------------------------------------------------

    def iterate_once(self, value_sets: VertexSets) -> VertexSets:
        """Compute one VSI step by applying :meth:`_compute_value_set` per state.

        State scheduling and progress reporting are handled by
        :func:`map_states`, according to the engine's execution settings.
        """
        self._validate_value_sets(value_sets, name="value_sets")
        num_states = self.num_states

        par_label = "in parallel" if self.params.vsi_operator_parallel else "sequentially"
        self._logger.info(
            f"Computing {self._mode_label} {par_label} "
            f"({num_states} states)"
        )

        new_sets = map_states(
            self._compute_value_set,
            value_sets,
            lp_factory=self.lp_factory,
            num_states=num_states,
            parallel=self.params.vsi_operator_parallel,
            max_workers=self.params.max_workers,
            show_progress=self.params.show_progress,
        )

        total = sum(len(v) for v in new_sets)
        self._logger.info(
            f"Computed {total} points across "
            f"{num_states} states"
        )
        return VertexSets(new_sets)

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
        Compute the vertices of the integer hull of the achievable-value
        polytope at state *s0*.

        Steps:
            1. Build the LP, find a seed feasible point, derive grid cell corners.
            2. Promote a fresh build to a MILP via grid indicators.
            3. Iterative facet-based separation until no new vertices.
            4. Convert integer vertices to value space and trim.
        """
        d = self.reward_dim
        spacing = self.grid.grid_spacing
        k_min = self.grid.k_min   # may be None (unbounded below)
        k_max = self.grid.k_max   # may be None (unbounded above)

        if len(value_sets[s0]) == 0:
            return np.empty((0, d))

        # --- Step 1: seed via LP-relaxation feasible point ---
        sink.set_label("seeding")
        discovered: set[tuple[int, ...]] = set()

        with lp_factory.open_lp(
            s0, value_sets, slack=0.0,
        ) as lp:
            if not lp.feasible:
                return np.empty((0, d))
            seed = lp.some_point()
            if seed is None:
                return np.empty((0, d))

        # Snap the seed to its surrounding grid-cell corners to start the
        # hull, clamping each side only where the grid is bounded.
        k_cont = seed / spacing
        k_floor = np.floor(k_cont).astype(int)
        k_ceil = k_floor + 1
        if k_min is not None:
            k_floor = np.maximum(k_floor, k_min)
            k_ceil = np.maximum(k_ceil, k_min)
        if k_max is not None:
            k_floor = np.minimum(k_floor, k_max)
            k_ceil = np.minimum(k_ceil, k_max)
        for corner in itertools.product(*zip(k_floor, k_ceil)):
            discovered.add(tuple(int(c) for c in corner))

        # --- Steps 2-3: MILP + iterative facet-based separation ---
        sink.set_label("building")
        with lp_factory.open_milp(
            s0, value_sets, grid=self.grid,
        ) as milp:
            if not milp.feasible:
                return np.empty((0, d))

            for round_num in itertools.count(1):
                # Re-hull the discovered grid points; every facet is a
                # candidate separating direction.
                pts = np.array(sorted(discovered))
                if d == 1:
                    # Qhull needs at least 2D; an interval has two walls.
                    lo, hi = pts[:, 0].min(), pts[:, 0].max()
                    equations = np.array([[-1.0, lo], [1.0, -hi]])
                else:
                    try:
                        equations = sp.ConvexHull(pts).equations
                    except sp.QhullError as exc:
                        raise ValueError(
                            f"Cannot construct a full-dimensional grid hull at state {s0}. "
                            "The initial grid cell must span the reward space. "
                            "Check for fixed grid coordinates or a seed at a clipping "
                            "boundary; consider expanding the grid bounds."
                        ) from exc

                n_facets = len(equations)
                new_this_round = 0
                sink.set_label(
                    f"round {round_num} ({len(discovered)} points) · facets"
                )
                sink.set_total(n_facets)

                for eq in equations:
                    normal = eq[:d]
                    offset = eq[d]

                    # Push outward along the facet normal; accept any grid
                    # point that lies strictly beyond the hull boundary.
                    result = milp.maximize_grid_index(normal)
                    if result is not None:
                        k, obj_val = result
                        if obj_val + offset > self.params.facet_sep_tol:
                            if k not in discovered:
                                discovered.add(k)
                                new_this_round += 1

                    sink.advance()

                if new_this_round == 0:
                    break

        # --- Step 4: convert grid indices to value space and trim ---
        if not discovered:
            return np.empty((0, d))

        k_pts = np.array(sorted(discovered), dtype=float)
        vertices = k_pts * spacing
        vertices = convex_hull_vertices(vertices)
        return vertices

    # ------------------------------------------------------------------
    # Policy expansion
    # ------------------------------------------------------------------

    def expand_value(
            self,
            value: np.ndarray,
            state: State,
            onward_value_sets: VertexSets,
    ) -> ExpansionResult | None:
        """
        Decompose an achievable value into onward components.

        Returns:
            An ``ExpansionResult`` if feasible, or ``None`` if not.
        """
        self._validate_value_sets(
            onward_value_sets, name="onward_value_sets",
        )
        try:
            state = operator.index(state)
        except TypeError as exc:
            raise ValueError(
                f"state must be an integer in [0, {self.num_states})"
            ) from exc
        if not 0 <= state < self.num_states:
            raise ValueError(
                f"state must be an integer in [0, {self.num_states})"
            )

        value = np.asarray(value, dtype=float)
        if value.shape != (self.reward_dim,):
            raise ValueError(
                f"value must have shape ({self.reward_dim},), "
                f"got {value.shape}"
            )
        if not np.isfinite(value).all():
            raise ValueError("value must contain only finite values")

        states = range(self.num_states)

        vs_trimmed = VertexSets(
            [convex_hull_vertices(onward_value_sets[s]) for s in states],
        )
        with self.lp_factory.open_lp(
            state, vs_trimmed, slack=self._expansion_slack,
        ) as lp:
            if not lp.feasible:
                return None
            return lp.expand_point(value)

    # ------------------------------------------------------------------
    # Convergence metric
    # ------------------------------------------------------------------

    def value_sets_distance(
            self,
            old: VertexSets,
            new: VertexSets,
    ) -> float:
        self._validate_value_sets(old, name="old")
        self._validate_value_sets(new, name="new")
        return max(
            polytope_distance(old[s], new[s])
            for s in range(self.num_states)
        )

    # ------------------------------------------------------------------
    # Validation and parameter management
    # ------------------------------------------------------------------

    def _validate_value_sets(
        self, value_sets: VertexSets, *, name: str,
    ) -> None:
        """Validate value sets at the public engine boundary."""
        if len(value_sets) != self.num_states:
            raise ValueError(
                f"{name} must contain {self.num_states} state sets, "
                f"got {len(value_sets)}"
            )

        for state, points in enumerate(value_sets):
            points = np.asarray(points, dtype=float)
            if points.ndim > 0 and points.shape[0] == 0:
                continue
            if points.ndim != 2 or points.shape[1] != self.reward_dim:
                raise ValueError(
                    f"{name}[{state}] must have shape "
                    f"(n, {self.reward_dim}), got {points.shape}"
                )
            if not np.isfinite(points).all():
                raise ValueError(
                    f"{name}[{state}] must contain only finite values"
                )

    def set_param(self, name: str | None = None, value=None, **kwargs) -> None:
        """Set one or more engine parameters on ``self.params``.

        Accepts a single ``name``/``value`` pair and/or keyword
        parameters. Raises ``AttributeError`` for any unknown parameter
        and ``ValueError`` for invalid algorithm parameters.
        """
        self.params.set_param(name, value, **kwargs)

    def get_param(self, name: str):
        """Return the value of engine parameter *name* from ``self.params``.

        Raises ``AttributeError`` if the parameter does not exist.
        """
        if not hasattr(self.params, name):
            raise AttributeError(f"Unknown parameter: {name!r}")
        return getattr(self.params, name)
