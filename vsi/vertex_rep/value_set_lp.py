"""Per-state optimization sessions for value-set iteration.

A domain implements ``ValueSetLPFactory._build_model`` to build the LP
defining its exact achievable values. The factory owns solver settings and
opens LP or MILP query sessions. Each session owns its model and disposes it
when its context closes.

``LPSession`` adds a companion value variable coupled to the exact value
by a box of half-width ``slack``. Its queries operate on that companion variable;
zero slack gives exact queries. ``MILPSession`` additionally constrains
the companion variable to the grid using integer indices.

Serial queries reuse a quiet environment from ``_gurobi_env.py``.
Parallel execution binds the factory to a separate environment per worker.
These bindings borrow environments; ``_execution.py`` owns their lifetime.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Generic
from typing_extensions import Self

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from vsi.value_set_engine import (
    ExpansionResult, Point, Policy, State, StepOutcomeT,
)
from .grid import Grid
from ._gurobi_env import shared_env
from vsi._validation import as_nonnegative_float


# ============================================================
# Parameters
# ============================================================

@dataclass(frozen=True, kw_only=True)
class LPSolverParams:
    """
    Solver-side parameters for :class:`LPSession`.

    Attributes:
        opt_tol: Gurobi ``OptimalityTol`` applied to the LP.
        fsb_tol: Gurobi ``FeasibilityTol`` applied to the LP.
        trim_distr_tol: Tolerance used by
            :meth:`LPSession.expand_point` to zero out small
            action probabilities before renormalising.
    """
    opt_tol: float = 1e-9
    fsb_tol: float = 1e-9
    trim_distr_tol: float = 1e-6

    def __post_init__(self):
        for param in fields(self):
            object.__setattr__(
                self, param.name,
                as_nonnegative_float(getattr(self, param.name), param.name),
            )


@dataclass(frozen=True, kw_only=True)
class MILPSolverParams:
    """
    Solver-side parameters for :class:`MILPSession`.

    The MILP-level analogue of :class:`LPSolverParams`, grouping the
    Gurobi tolerances that only apply once the value-set LP has been
    promoted to a grid MILP.

    Keep both gaps at zero for integer-hull computation: positive gaps can
    miss vertices and prevent value-set iteration from converging.

    Attributes:
        int_feas_tol: Gurobi ``IntFeasTol`` applied to the MILP.
        mip_gap: Gurobi ``MIPGap`` (relative) applied to the MILP.
        mip_gap_abs: Gurobi ``MIPGapAbs`` (absolute) applied to the MILP.
    """
    int_feas_tol: float = 1e-9
    mip_gap: float = 0.0
    mip_gap_abs: float = 0.0

    def __post_init__(self):
        for param in fields(self):
            object.__setattr__(
                self, param.name,
                as_nonnegative_float(getattr(self, param.name), param.name),
            )


# ============================================================
# ValueSetLPModel (built product)
# ============================================================

@dataclass(frozen=True, kw_only=True)
class ValueSetLPModel(Generic[StepOutcomeT]):
    """
    A built value-set LP model and references to its key variables.

    A :class:`ValueSetLPModel` is a passive container holding one Gurobi
    model encoding the set of values achievable at a single state,
    conditional on the supplied onward value sets.  It is produced by
    :meth:`ValueSetLPFactory._build_model` and consumed by
    :class:`LPSession`, which wraps it and drives it.

    The model defines the *exact* feasible set.  The slacked companion
    variable used by all queries is added separately by
    :class:`LPSession`.

    Attributes:
        solver: Gurobi model encoding the exact feasible set.
        value_var: Exact value variable ``v`` (shape ``(d,)``).
        act_probs_var: Action-distribution variable ``π``.
        onward_weights_var: Onward-value weight variables, keyed by
            step outcome.  The keys are the LP's step outcomes; each
            key carries the ``action`` and ``state`` the branch leads to.
    """
    solver: gp.Model
    value_var: gp.MVar
    act_probs_var: gp.MVar
    onward_weights_var: dict[StepOutcomeT, gp.MVar]


# ============================================================
# ValueSetLPFactory (abstract, domain-facing)
# ============================================================

@dataclass(frozen=True, kw_only=True)
class ValueSetLPFactory(ABC, Generic[StepOutcomeT]):
    """Build value-set LP models and open query sessions for a bound problem.

    Domain subclasses implement ``_build_model``, ``num_states``, and ``reward_dim``.
    The shared opening methods apply the configured parameters and wrap the
    model in the appropriate session. An LP session has no integer grid state;
    an MILP session adds integer grid-index constraints to the LP.
    Factories and their settings are immutable and can be shared by workers.
    """

    lp_params: LPSolverParams = field(default_factory=LPSolverParams)
    milp_params: MILPSolverParams = field(default_factory=MILPSolverParams)

    def open_lp(
        self, s0: State, onward_value_sets, *, slack: float = 0.0,
    ) -> LPSession[StepOutcomeT]:
        """Open an LP session, disposed when its context closes."""
        return self._open_lp(
            s0, onward_value_sets, slack=slack, solver_env=shared_env(),
        )

    def open_milp(
        self, s0: State, onward_value_sets, *, grid: Grid,
    ) -> MILPSession[StepOutcomeT]:
        """Open a session with integer grid-index constraints."""
        return self._open_milp(
            s0, onward_value_sets, grid=grid, solver_env=shared_env(),
        )

    def bind_environment(self, solver_env: gp.Env) -> _BoundLPFactory:
        """Return a worker opener borrowing an externally owned environment."""
        return _BoundLPFactory(self, solver_env)

    def _open_lp(self, s0, onward_value_sets, *, slack, solver_env):
        lp = self._build_model(
            s0, onward_value_sets, params=self.lp_params, solver_env=solver_env,
        )
        try:
            return LPSession(
                lp, onward_value_sets, params=self.lp_params, slack=slack,
            )
        except Exception:
            if lp is not None:
                lp.solver.dispose()
            raise

    def _open_milp(self, s0, onward_value_sets, *, grid, solver_env):
        lp = self._build_model(
            s0, onward_value_sets, params=self.lp_params, solver_env=solver_env,
        )
        try:
            return MILPSession(
                lp, onward_value_sets, params=self.lp_params,
                grid=grid, milp_params=self.milp_params,
            )
        except Exception:
            if lp is not None:
                lp.solver.dispose()
            raise

    @property
    @abstractmethod
    def num_states(self) -> int:
        """Number of problem states the engine iterates over."""
        ...

    @property
    @abstractmethod
    def reward_dim(self) -> int:
        """Value-space (reward) dimension ``d`` the engine iterates in."""
        ...

    @abstractmethod
    def _build_model(
        self,
        s0: State,
        onward_value_sets,
        *,
        params: LPSolverParams,
        solver_env: gp.Env | None = None,
    ) -> ValueSetLPModel[StepOutcomeT] | None:
        """
        Build the Gurobi LP for the *exact* feasible set at *s0*.

        Encodes the domain-specific Bellman and incentive-compatibility
        constraints and returns a built :class:`ValueSetLPModel` carrying the
        raw model and key variables.  Do not add slack here —
        :class:`LPSession` adds the slacked companion variable.

        Args:
            s0: State at which the value set is defined.
            onward_value_sets: Onward value sets from the previous VSI
                iterate, one per state.  The caller is responsible for
                any trimming/canonicalisation (e.g.
                ``convex_hull_vertices``).
            params: Solver-side parameters.
            solver_env: Environment supplied by the opening methods: the
                shared serial environment or an exclusive worker environment.
                Direct calls with ``None`` use Gurobi's default environment.

        Returns:
            A built :class:`ValueSetLPModel`, or ``None`` if the problem is
            trivially infeasible at *s0* (no LP need be constructed).
        """
        ...


# ============================================================
# Worker environment binding
# ============================================================

@dataclass(frozen=True)
class _BoundLPFactory:
    """Forward session creation using one worker's borrowed environment."""

    lp_factory: ValueSetLPFactory
    solver_env: gp.Env

    def open_lp(self, s0, onward_value_sets, *, slack=0.0):
        return self.lp_factory._open_lp(
            s0, onward_value_sets, slack=slack, solver_env=self.solver_env,
        )

    def open_milp(self, s0, onward_value_sets, *, grid):
        return self.lp_factory._open_milp(
            s0, onward_value_sets, grid=grid, solver_env=self.solver_env,
        )


# ============================================================
# LPSession (live operator)
# ============================================================

class LPSession(Generic[StepOutcomeT]):
    """
    Live, per-state operator over a :class:`ValueSetLPModel`.

    A session wraps one built :class:`ValueSetLPModel` (or ``None`` for a
    trivially-infeasible state), adds the slacked companion variable
    coupled to the exact variable with the half-width ``slack`` fixed at
    construction, and exposes the query / expansion / lifecycle API.
    All query methods return or optimise over the slacked variable.

    Open a session through the factory::

        with lp_factory.open_lp(s, onward_value_sets, slack=slack) as lp:
            if not lp.feasible:   # trivially infeasible
                ...
            pt = lp.some_point()
            ...

    The wrapped model is disposed in :meth:`__exit__`.  The environment
    is owned by the caller and must outlive the session.
    """

    def __init__(
        self,
        lp: ValueSetLPModel[StepOutcomeT] | None,
        onward_value_sets,
        *,
        params: LPSolverParams,
        slack: float = 0.0,
    ) -> None:
        """
        Wrap a built :class:`ValueSetLPModel` and fix its slack.

        Adds the slacked companion variable coupled to the exact
        variable with half-width *slack*.  Prefer ``lp_factory.open_lp(...)``, which
        builds the LP and wraps it in one step.

        Args:
            lp: A built :class:`ValueSetLPModel`, or ``None`` if the domain
                hook reported the state to be trivially infeasible.
            onward_value_sets: Onward value sets used to build *lp*;
                retained for back-substitution in :meth:`expand_point`.
            params: Solver-side parameters.
            slack: Non-negative coupling half-width between the exact
                and slacked variables.  ``0`` (default) makes every
                query exact; the grid spacing tolerates rounding to
                the nearest grid cell.
        """
        self._lp = lp
        self._onward_value_sets = list(onward_value_sets)
        self._params = params

        if lp is None:
            # Trivially infeasible at s0: no model was built.
            self._solver = None
            return

        self._solver = lp.solver
        self._reward_dim = lp.value_var.shape[0]

        # --- Slacked value variable and permanent coupling constraints ---
        # _slacked_value_var is coupled with the exact value_var via:
        #   _slacked_value_var[i] - value_var[i] \in [-slack, +slack]
        # slack=0 forces _slacked_value_var = value_var (collapsed by
        # Gurobi's presolve).  All query methods operate on
        # _slacked_value_var; the exact value is recovered via value_var.X.
        d = self._reward_dim
        v = lp.value_var
        solver = self._solver
        v_slacked = solver.addMVar(shape=(d,), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="v_slacked")
        self._slacked_value_var: gp.MVar = v_slacked
        solver.addConstr(v_slacked - v >= np.full(d, -slack), name="coup_lb")
        solver.addConstr(v_slacked - v <= np.full(d,  slack), name="coup_ub")
        solver.update()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def feasible(self) -> bool:
        """
        Whether an LP was built for this state.

        ``False`` means the domain hook reported the state to be
        trivially infeasible (no LP was built) and query methods must
        not be called.  ``True`` means the LP exists, though its
        (slacked) value set may still turn out to be empty when solved.
        """
        return self._lp is not None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        if self._solver is not None:
            self._solver.dispose()
            self._solver = None

    # ------------------------------------------------------------------
    # Set-like queries (operate on _slacked_value_var)
    # ------------------------------------------------------------------

    def _solve_for_point(self, objective, sense: int) -> Point | None:
        """
        Set *objective* with *sense*, solve, and return the slacked
        variable's value, or ``None`` if the (slacked) set is empty.

        Shared core of :meth:`some_point`, :meth:`nearest_point`, and
        :meth:`support_point`.
        """
        solver = self._solver
        solver.setObjective(objective, sense)
        solver.optimize()
        if solver.status == GRB.OPTIMAL:
            return self._slacked_value_var.X.copy()
        if solver.status == GRB.INFEASIBLE:
            return None
        raise RuntimeError(
            f"LPSession: LP returned status {solver.status}"
        )

    def is_nonempty(self) -> bool:
        """Return ``True`` iff the slacked value set is non-empty."""
        return self.some_point() is not None

    def some_point(self) -> Point | None:
        """
        Return an arbitrary feasible point of the slacked value set,
        or ``None`` if empty.
        """
        return self._solve_for_point(0, GRB.MINIMIZE)

    def nearest_point(self, point: Point) -> Point | None:
        """
        Return the point of the slacked feasible set closest to *point*
        in Euclidean distance (its projection), or ``None`` if empty.

        Pass ``np.zeros(d)`` to find the minimum-norm feasible point.
        """
        v_slacked = self._slacked_value_var
        # Minimise (v_slacked - point)^2 = v_slacked @ v_slacked
        #   - 2 * point @ v_slacked + const.
        # The constant does not affect the argmin.
        return self._solve_for_point(
            v_slacked @ v_slacked - 2 * (point @ v_slacked), GRB.MINIMIZE
        )

    def support_point(self, normal: np.ndarray) -> Point | None:
        """
        Return the point of the slacked feasible set maximising
        ``normal @ v_slacked`` (the supporting point in direction
        *normal*), or ``None`` if empty.
        """
        v_slacked = self._slacked_value_var
        return self._solve_for_point(normal @ v_slacked, GRB.MAXIMIZE)

    # ------------------------------------------------------------------
    # Back-substitution
    # ------------------------------------------------------------------

    def expand_point(self, point: Point) -> ExpansionResult | None:
        """
        Decompose *point* into action probabilities and onward policies.

        Pins the slacked variable to *point* (via :meth:`set_target`),
        solves the LP, and extracts the action distribution and onward
        policy values.  Returns ``None`` if no feasible decomposition
        exists (i.e. the exact value set has no element within
        ``slack`` of *point*).

        ``value_achieved`` in the returned :class:`ExpansionResult` is
        the *exact* value ``value_var.X``, which may differ from
        *point* by up to ``slack`` per dimension.

        The tolerance is the ``slack`` fixed at construction.
        """
        lp = self._lp
        self.set_target(point)
        try:
            self._solver.setObjective(0, GRB.MINIMIZE)
            self._solver.optimize()
            if self._solver.status == GRB.INFEASIBLE:
                return None
            if self._solver.status != GRB.OPTIMAL:
                raise RuntimeError(
                    f"expand_point: LP returned status {self._solver.status}"
                )

            value_achieved = lp.value_var.X.copy()
            raw_act_probs = lp.act_probs_var.X.copy()

            # Recover conditional onward values using the original LP
            # probabilities, to which the onward weights are coupled.
            onward_policies: dict = {}
            for stp_out, weights_var in lp.onward_weights_var.items():
                a = stp_out.action
                s = stp_out.state
                if raw_act_probs[a] > self._params.trim_distr_tol:
                    onward_val = (
                        weights_var.X @ self._onward_value_sets[s]
                        / raw_act_probs[a]
                    )
                else:
                    onward_val = np.zeros(self._reward_dim)
                onward_policies[stp_out] = Policy(
                    value=onward_val, init_state=s,
                )

            # Treat numerically insignificant actions as unused, then
            # renormalise the remaining probabilities.
            act_probs = np.where(
                raw_act_probs > self._params.trim_distr_tol,
                raw_act_probs,
                0.0,
            )
            total = act_probs.sum()
            if total == 0:
                raise RuntimeError("All action probabilities were trimmed")
            act_probs /= total

            return ExpansionResult(
                act_probs, onward_policies, value_achieved,
            )
        finally:
            self.clear_target()

    # ------------------------------------------------------------------
    # Target pinning (for re-solve loops)
    # ------------------------------------------------------------------

    def set_target(self, point: Point) -> None:
        """
        Pin the slacked variable to *point* by setting its lower and
        upper bounds to *point*.

        Combined with the coupling constraint (half-width ``slack``),
        this forces the exact variable to lie within ``slack`` of
        *point*.  Use :meth:`clear_target` to release the pin.

        This is a lightweight variable-bound update (no structural model
        changes) and is safe to call in a tight re-solve loop.
        """
        v_slacked = self._slacked_value_var
        pt = np.asarray(point, dtype=float)
        v_slacked.LB = pt
        v_slacked.UB = pt

    def clear_target(self) -> None:
        """
        Release the pin set by :meth:`set_target` by resetting the
        slacked variable's bounds to ``(-inf, +inf)``.
        """
        v_slacked = self._slacked_value_var
        d = self._reward_dim
        v_slacked.LB = np.full(d, -GRB.INFINITY)
        v_slacked.UB = np.full(d,  GRB.INFINITY)


# ============================================================
# MILPSession (MILP promotion)
# ============================================================

class MILPSession(
    LPSession[StepOutcomeT], Generic[StepOutcomeT]
):
    """
    A :class:`LPSession` promoted to a MILP over integer grid
    indices.

    On construction it fixes the coupling slack to ``grid.grid_spacing``
    and pins the slacked variable to the value-space grid with integer
    grid-index variables ``k`` (``v_slacked = grid_spacing·k``), so the
    exact variable is confined to the grid cell surrounding ``k``.  This
    enables facet separation over grid points via
    :meth:`maximize_grid_index`.  Used by the integer-hull path of
    :class:`vsi.VertexRepEngine`.

    Prefer ``lp_factory.open_milp(...)``, which builds the LP, promotes it to a MILP, and
    wraps it in one step.  For a trivially-infeasible state (the factory
    returns ``None``) the session is created with :attr:`feasible`
    ``False`` and no grid indices are attached; callers must check
    :attr:`feasible` before issuing queries.
    """

    def __init__(
        self,
        lp: ValueSetLPModel[StepOutcomeT] | None,
        onward_value_sets,
        *,
        params: LPSolverParams,
        grid: Grid,
        milp_params: MILPSolverParams,
    ) -> None:
        """
        Wrap a built :class:`ValueSetLPModel`, fix the grid-spacing slack, and
        pin the slacked variable to the grid.

        The coupling slack is fixed to ``grid.grid_spacing`` (so the exact
        variable lies within the surrounding grid cell), and integer
        grid-index variables ``k`` are attached with
        ``v_slacked = grid.grid_spacing·k``.  Prefer ``lp_factory.open_milp(...)``, which
        builds the LP and wraps it in one step.

        Args:
            lp: A built :class:`ValueSetLPModel`, or ``None`` if the domain
                hook reported the state to be trivially infeasible.
            onward_value_sets: Onward value sets used to build *lp*.
            params: LP solver-side parameters.
            grid: Value-space :class:`vsi.Grid` supplying the grid
                spacing and optional integer index bounds.
            milp_params: MILP solver-side parameters.
        """
        super().__init__(
            lp, onward_value_sets, params=params, slack=grid.grid_spacing,
        )
        self._grid_idx_var: gp.MVar | None = None
        if lp is not None:
            self._attach_grid_indices(grid=grid, milp_params=milp_params)

    def _attach_grid_indices(
        self, *, grid: Grid, milp_params: MILPSolverParams,
    ) -> None:
        """
        Promote the LP to a MILP by adding integer grid-index variables
        ``k`` and pinning the slacked variable to the grid:

            v_slacked = grid.grid_spacing·k

        Combined with the coupling constraint (slack ``grid.grid_spacing``,
        fixed in :meth:`__init__`), the exact variable is constrained to
        the grid cell:

            v[i] ∈ [grid_spacing·k[i] - grid_spacing,
                    grid_spacing·k[i] + grid_spacing]

        The integer index ``k`` is confined to ``grid.k_min[i] ≤ k[i] ≤
        grid.k_max[i]`` on whichever side(s) the grid is bounded; an
        unbounded side leaves ``k`` free there (the value-set LP keeps
        ``v``, and hence ``k``, bounded in practice).
        """
        solver = self._solver
        v_slacked = self._slacked_value_var
        d = self._reward_dim

        solver.setParam("IntFeasTol", milp_params.int_feas_tol)
        solver.setParam("MIPGap", milp_params.mip_gap)
        solver.setParam("MIPGapAbs", milp_params.mip_gap_abs)

        # Integer grid index, unbounded by default (Gurobi's default
        # integer lb of 0 would wrongly pin the grid to the origin).
        k = solver.addMVar(
            shape=(d,), vtype=GRB.INTEGER,
            lb=-GRB.INFINITY, ub=GRB.INFINITY, name="k",
        )
        if grid.k_min is not None:
            k.LB = np.asarray(grid.k_min, dtype=float)
        if grid.k_max is not None:
            k.UB = np.asarray(grid.k_max, dtype=float)
        # Pin the slacked variable to the grid: v_slacked = grid_spacing·k
        solver.addConstr(v_slacked == grid.grid_spacing * k, name="grid_eq")
        solver.update()
        self._grid_idx_var = k

    def maximize_grid_index(
        self, normal: np.ndarray,
    ) -> tuple[tuple[int, ...], float] | None:
        """
        Maximize ``normal @ k`` over the integer grid indices and
        return ``(grid_index_tuple, objective_value)``.

        Returns ``None`` if the MILP is infeasible.
        """
        if self._grid_idx_var is None:
            raise RuntimeError(
                "maximize_grid_index: session is infeasible (no grid "
                "indices were attached); check `feasible` first."
            )

        k = self._grid_idx_var
        solver = self._solver
        solver.setObjective(normal @ k, GRB.MAXIMIZE)
        solver.optimize()

        if solver.status == GRB.OPTIMAL:
            kt = tuple(int(round(x)) for x in k.X)
            return kt, solver.objVal
        if solver.status == GRB.INFEASIBLE:
            return None
        if solver.status in (GRB.UNBOUNDED, GRB.INF_OR_UNBD):
            raise RuntimeError(
                "maximize_grid_index: MILP is unbounded along the given "
                "direction.  The grid is unbounded and the value-set LP "
                "did not confine the index; supply k_min/k_max in the "
                "Grid."
            )

        raise RuntimeError(
            f"maximize_grid_index: MILP returned status {solver.status}"
        )
