"""
Shared visualization utilities, independent of any specific problem domain.

Re-exports all public plot functions so that domain-specific visualize
packages can import from here or re-export selectively.
"""
from experiments.common.visualize._legend import _best_corner_legend
from experiments.common.visualize.point_sets import (
    plot_point_sets_2d,
    plot_point_sets_by_state,
)
from experiments.common.visualize.convergence import (
    plot_convergence,
    plot_iteration_snapshots_2d,
)
from experiments.common.visualize.sweep import (
    plot_bound_sweep,
    plot_scaling,
)
from experiments.common.visualize.trajectory import (
    plot_policy_trajectory,
    plot_singleton_trajectory,
)

__all__ = [
    "_best_corner_legend",
    "plot_point_sets_2d",
    "plot_point_sets_by_state",
    "plot_convergence",
    "plot_iteration_snapshots_2d",
    "plot_bound_sweep",
    "plot_scaling",
    "plot_policy_trajectory",
    "plot_singleton_trajectory",
]

