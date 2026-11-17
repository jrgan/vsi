"""Vertex-representation engines and LP helpers for value-set iteration.

This subpackage implements :class:`vsi.ValueSetEngine` for value sets stored as
vertices on a grid. It includes:

* :class:`VertexRepEngine` — iterative MILP facet separation, computing the
  integer hull per state (high-fidelity, the concrete default)
* :class:`SingletonEngine` — single feasible point per state (lightweight)
* :class:`GridEnumEngine` — baseline enumeration of all grid points
* :class:`ValueSetLPFactory` — domain-facing factory that opens per-state LP/MILP sessions
* :class:`ValueSetLPModel` — the built product wrapping a Gurobi model and key variables
* :class:`LPSession` — live per-state operator wrapping a built ``ValueSetLPModel``
* :class:`MILPSession` — ``LPSession`` promoted to a MILP over grid indices
* :class:`Grid` — value-space grid (spacing + optional index bounds)

To add a new domain, define a
``MyDomainValueSetLPFactory(ValueSetLPFactory[MyDomainStepOutcome])``
subclass (implementing ``_build_model``, ``num_states``, and ``reward_dim``) and a
``my_domain_grid`` helper. The same domain setup then works across all
engines in this package.
"""
from .engine import (
    VertexRepEngine, VertexSets, VertexRepEngineParams,
)
from .singleton_engine import SingletonEngine, PointSelection
from .value_set_lp import (
    ValueSetLPModel, ValueSetLPFactory, LPSession, MILPSession,
    LPSolverParams, MILPSolverParams,
)
from .grid import Grid
from .initial_sets import box_corners, box_centre, uniform_value_sets

__all__ = [
    "VertexRepEngine", "VertexSets", "VertexRepEngineParams",
    "SingletonEngine", "PointSelection",
    "uniform_value_sets",
    "ValueSetLPModel", "ValueSetLPFactory",
    "LPSession", "MILPSession",
    "LPSolverParams", "MILPSolverParams",
    "Grid", "box_corners", "box_centre",
]
