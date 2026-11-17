"""
Helpers for the runnable examples.

Convert value sets to point arrays and reuse experiment helpers to save
snapshots, trajectories, and animations.
"""
from __future__ import annotations

import logging

import numpy as np
import matplotlib.pyplot as plt

from vsi import RunnerStatus
from experiments.common.io import (clear_run_artifacts, save_final_value_sets, save_value_sets_iteration,
                                   save_trajectory, save_policy_expansions)
from experiments.common.runner import rollout_policy
from experiments.common.visualize import plot_iteration_snapshots_2d, plot_policy_trajectory

logger = logging.getLogger(__name__)


def value_sets_to_point_lists(vs) -> list[np.ndarray]:
    """Convert any ``ValueSets`` representation to a list of point arrays."""
    if vs is None:
        return []
    return [np.array(pts) for pts in vs]


def save_example_run(instance, engine, runner, out_dir, stem: str,
                     make_step_outcome, *, init_state: int = 0,
                     num_steps: int = 12, seed: int = 123) -> None:
    """Save snapshot, animation, and sampled policy artifacts for an example."""
    from experiments.common.animate_value_sets import animate as animate_iterations
    from experiments.common.animate_policy_trajectory import animate as animate_policy

    run_dir = out_dir / stem
    run_dir.mkdir(parents=True, exist_ok=True)
    clear_run_artifacts(run_dir)
    history = [value_sets_to_point_lists(vs) for vs in (runner.sequence or [])]
    for iteration, snapshot in enumerate(history):
        save_value_sets_iteration(snapshot, iteration, run_dir)
    final_sets = value_sets_to_point_lists(runner.result)
    if runner.status == RunnerStatus.CONVERGED and final_sets:
        save_final_value_sets(final_sets, run_dir)
    if history:
        fig = plot_iteration_snapshots_2d(history)
        for fmt in ("png", "pdf"):
            fig.savefig(run_dir / f"iteration_snapshots.{fmt}", dpi=150)
        plt.close(fig)
        animate_iterations(run_dir, run_dir / "iteration_animation.gif")
    if runner.status != RunnerStatus.CONVERGED or not final_sets:
        logger.warning("%s did not converge; policy trajectory was skipped", stem)
        return
    objective = np.zeros(final_sets[0].shape[1])
    objective[0] = 1.0
    policy = rollout_policy(engine, instance, runner.result, objective, init_state,
                            num_steps, seed, make_step_outcome)
    if not policy.feasible:
        logger.warning("No feasible policy for %s", stem)
        return
    save_trajectory(policy.trajectory, run_dir)
    save_policy_expansions(policy.expansions, run_dir)
    fig = plot_policy_trajectory(policy.trajectory, final_sets, len(final_sets),
                                 onward_values=policy.expansions["onward_values"],
                                 target_values=policy.expansions["targets"])
    for fmt in ("png", "pdf"):
        fig.savefig(run_dir / f"policy_trajectory.{fmt}", dpi=150)
    plt.close(fig)
    animate_policy(run_dir, run_dir / "policy_trajectory_animation.gif")
