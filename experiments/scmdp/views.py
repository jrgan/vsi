"""
SCMDP view classes (figure builders).

Each view is a ``BaseView`` subclass whose ``run()`` method returns
``{name: Figure}`` pairs. Call ``view.run_and_save()`` to persist all
figures alongside the runner's artifacts in a single directory.

For re-plotting from previously saved artifacts without re-running VSI,
use :func:`plot_single_run_from_dir`.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from matplotlib.figure import Figure

from experiments.common.view import BaseView, plot_single_run_from_dir as _plot_single_run_from_dir
from experiments.common.io import save_final_value_sets, save_trajectory, save_policy_expansions
from experiments.common.config import VertexSetMode
from experiments.common.visualize import (
    plot_convergence,
    plot_iteration_snapshots_2d,
    plot_policy_trajectory,
    plot_singleton_trajectory,
)
from experiments.common.visualize.sweep import (
    plot_bound_sweep as _plot_bound_sweep,
    plot_scaling as _plot_scaling,
)
from experiments.scmdp.config import ExperimentConfig
from experiments.common.results import value_sets_to_point_lists
from experiments.scmdp.runner import (
    run_experiment,
    run_policy_experiment,
    run_bound_sweep,
    run_scaling_experiment,
    run_singleton_experiment,
)

logger = logging.getLogger(__name__)


class SingleRunView(BaseView):
    """Single SCMDP run with optional policy extraction."""

    config: ExperimentConfig

    def run(self) -> dict[str, Figure]:
        figures: dict[str, Figure] = {}

        run_result, policy_result = run_policy_experiment(self.config)

        logger.info(
            "Runtime: %.3f s, %d iterations",
            run_result.runtime, run_result.num_iterations,
        )

        point_lists = value_sets_to_point_lists(run_result.final_value_sets)

        if run_result.history:
            history_pts = [
                value_sets_to_point_lists(h) for h in run_result.history
            ]
            figures["convergence"] = plot_convergence(
                history_pts, title="Convergence",
            )
            figures["iteration_snapshots"] = plot_iteration_snapshots_2d(history_pts)

        if policy_result.feasible and policy_result.trajectory:
            logger.info(
                "Optimal value: %.4f, vector: %s",
                policy_result.optimum, policy_result.optimal_value,
            )
            figures["policy_trajectory"] = plot_policy_trajectory(
                policy_result.trajectory,
                value_sets=point_lists,
                num_states=self.config.instance.num_states,
                title="Policy Trajectory",
                onward_values=policy_result.expansions["onward_values"],
                target_values=policy_result.expansions["targets"],
            )
            # The runner already persisted config/instance/value sets/history
            # into self.output_dir; only the trajectory needs saving here.
            save_trajectory(policy_result.trajectory, self.output_dir)
            save_policy_expansions(policy_result.expansions, self.output_dir)
        else:
            logger.warning("Policy infeasible; skipping trajectory plot.")

        return figures


class BoundSweepView(BaseView):
    """Sweep constraint bounds and overlay polytopes."""

    config: ExperimentConfig

    def __init__(self, config: ExperimentConfig, bounds: list[float]) -> None:
        super().__init__(config)
        self.bounds = bounds

    def run(self) -> dict[str, Figure]:
        results = run_bound_sweep(self.config, self.bounds)

        point_lists = [
            value_sets_to_point_lists(r.final_value_sets) for r in results
        ]
        labels = [f"B={r.config.instance.bound:.2f}" for r in results]

        return {
            "bound_sweep": _plot_bound_sweep(
                point_lists, labels, title="Bound Sweep",
            ),
        }


# Scaling axis -> (InstanceConfig attribute, axis label).
_SCALING_AXIS_META: dict[str, tuple[str, str]] = {
    "states": ("num_states", "Number of States"),
    "actions": ("num_actions", "Number of Actions"),
    "reward_dim": ("reward_dim", "Reward Dimension"),
}


class ScalingView(BaseView):
    """Runtime and iteration scaling with problem size.

    Sweeps one of ``"states"``, ``"actions"``, or ``"reward_dim"`` while
    holding the others fixed at the values in ``config``. When ``singleton``
    is set, the cheaper singleton inner loop is used instead of the full
    VSI engine.
    """

    config: ExperimentConfig

    def __init__(
        self,
        config: ExperimentConfig,
        sizes: list[int],
        axis: str = "states",
        *,
        singleton: bool = False,
        max_iters: int = 1000,
    ) -> None:
        if axis not in _SCALING_AXIS_META:
            raise ValueError(
                f"axis must be one of {sorted(_SCALING_AXIS_META)}, got {axis!r}"
            )
        super().__init__(config)
        self.sizes = sizes
        self.axis = axis
        self.singleton = singleton
        self.max_iters = max_iters

    def run(self) -> dict[str, Figure]:
        results = run_scaling_experiment(
            self.config, self.axis, self.sizes,
            singleton=self.singleton, max_iters=self.max_iters,
        )
        attr, xlabel = _SCALING_AXIS_META[self.axis]
        sizes = [getattr(r.config.instance, attr) for r in results]
        times = [r.runtime for r in results]
        iters = [r.num_iterations for r in results]

        mode = "Singleton " if self.singleton else ""
        prefix = "singleton_" if self.singleton else ""

        return {
            f"{prefix}scaling_{self.axis}": _plot_scaling(
                sizes, times, iters,
                title=f"{mode}Runtime & Iterations vs. {xlabel}",
                xlabel=xlabel,
            ),
        }


class SingletonView(BaseView):
    """Singleton iteration trajectory."""

    config: ExperimentConfig

    def __init__(
        self,
        config: ExperimentConfig,
        max_iters: int = 1000,
        show_background_hull: bool = False,
    ) -> None:
        super().__init__(replace(
            config, algorithm=replace(
                config.algorithm, mode=VertexSetMode.SINGLE_POINT,
                max_iters=max_iters,
            ),
        ))
        self.max_iters = max_iters
        self.show_background_hull = show_background_hull

    def run(self) -> dict[str, Figure]:
        singleton_result = run_singleton_experiment(
            self.config, max_iters=self.max_iters,
        )

        bg_value_sets = None
        if self.show_background_hull:
            background_config = replace(
                self.config, algorithm=replace(
                    self.config.algorithm, mode=VertexSetMode.GRID_VERTICES,
                ),
            )
            run_result = run_experiment(background_config)
            if run_result.final_value_sets is not None:
                bg_value_sets = value_sets_to_point_lists(
                    run_result.final_value_sets,
                )
                save_final_value_sets(bg_value_sets, self.output_dir / "background")

        return {
            "singleton_trajectory": plot_singleton_trajectory(
                singleton_result.history,
                title="Singleton Iteration Trajectory",
                value_sets=bg_value_sets,
                converged=singleton_result.converged,
                loop_detected=singleton_result.loop_detected,
            ),
        }


# --------------------------------------------------------------------------- #
#  Plot-from-artifacts
# --------------------------------------------------------------------------- #

def plot_single_run_from_dir(
    input_dir: str | Path,
    *,
    num_states: int | None = None,
    title_prefix: str = "",
) -> dict[str, Figure]:
    """Re-render SCMDP single-run figures from saved artifacts.

    See :func:`experiments.common.view.plot_single_run_from_dir`.
    """
    return _plot_single_run_from_dir(
        input_dir, num_states=num_states, title_prefix=title_prefix,
    )
