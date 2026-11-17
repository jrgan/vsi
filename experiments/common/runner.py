"""Generic experiment runner.

``BaseRunner`` holds domain-agnostic logic for single runs, policy
rollout, and singleton iteration. Subclasses provide six hooks:
``build_instance``, ``make_grid``, ``make_engine``, ``save_instance``,
``make_step_outcome``, and ``compute_summary``.

The module also provides :func:`run_parameter_sweep` for scaling/sweep
wrappers.
"""
from __future__ import annotations

import copy
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np

from vsi import (
    PolicyRunner, RunnerStatus, ValueSetEngine, IterationRunner,
    VertexRepEngine, SingletonEngine,
    Grid, box_corners, box_centre, uniform_value_sets,
)
from vsi.vertex_rep.baselines import GridEnumEngine, box_mesh
from vsi.value_set_engine import Policy
from vsi.vertex_rep.polytope import support_function
from experiments.common.config import VertexSetMode
from experiments.common.io import (
    clear_run_artifacts, run_dir, save_config, save_final_value_sets, save_summary,
    save_value_sets_iteration,
)
from experiments.common.results import (
    PolicyResult, RunResult, SingletonResult,
    singleton_vs_to_history, value_sets_to_point_lists,
)

logger = logging.getLogger(__name__)

#: Resolution table mapping the experiment-level :class:`VertexSetMode`
#: tag to a builder that constructs the engine from a value-set LP
#: factory and a grid.  The grid-free :class:`vsi.SingletonEngine`
#: ignores the grid argument (mirrors ``SEED_FN`` below, where the
#: singleton is likewise special-cased).
ENGINE_BY_MODE: dict[VertexSetMode, Callable[..., VertexRepEngine]] = {
    VertexSetMode.GRID_VERTICES: lambda f, g: VertexRepEngine(f, g),
    VertexSetMode.SINGLE_POINT: lambda f, g: SingletonEngine(f),
    VertexSetMode.ALL_GRID_POINTS: lambda f, g: GridEnumEngine(f, g),
}

#: Resolution table mapping the experiment-level :class:`VertexSetMode`
#: tag to the seed helper that produces the initial points each engine
#: expects (corners for the integer hull, centre for the singleton, full
#: mesh for the grid-enumeration baseline).  The box helpers take explicit
#: bounds read off the clipped :class:`vsi.Grid`; the baseline mesh takes
#: the grid directly, since it is operator machinery, not a box seed.
SEED_FN: dict[VertexSetMode, Callable[[Grid], np.ndarray]] = {
    VertexSetMode.GRID_VERTICES: lambda g: box_corners(g.lbs, g.ubs),
    VertexSetMode.SINGLE_POINT: lambda g: box_centre(g.lbs, g.ubs),
    VertexSetMode.ALL_GRID_POINTS: box_mesh,
}

C = TypeVar("C")
R = TypeVar("R")


def rollout_policy(engine, instance, value_sets, objective, init_state: int,
                   num_steps: int, seed: int, make_step_outcome) -> PolicyResult:
    """Roll out a policy and retain each actual target and onward expansion."""
    result = support_function(value_sets[init_state], objective)
    if result is None:
        return PolicyResult(feasible=False)
    optimum, value = result
    policy = Policy(value=value, init_state=init_state)
    policy_runner = PolicyRunner(engine, value_sets, policy, seed=seed)
    traj_rng = np.random.default_rng(seed)
    transitions = instance.transitions
    states, targets, achieved = [], [], []
    onward_steps, onward_states, onward_actions, onward_values = [], [], [], []
    trajectory = []
    prev_state = init_state
    for step in range(num_steps):
        target = np.asarray(policy_runner.next_policy.value, dtype=float).copy()
        expansion = policy_runner.next_step()
        states.append(prev_state)
        targets.append(target)
        achieved_value = np.asarray(expansion.value_achieved, dtype=float).copy()
        achieved.append(achieved_value)
        trajectory.append((prev_state, achieved_value))
        for outcome, onward_policy in expansion.onward_policies.items():
            action, state = int(outcome.action), int(outcome.state)
            if (getattr(outcome, "action_played", action) != action or
                    expansion.action_probs[action] <= 1e-8 or
                    transitions[prev_state, action, state] <= 1e-12):
                continue
            onward_steps.append(step)
            onward_states.append(state)
            onward_actions.append(action)
            onward_values.append(np.asarray(onward_policy.value, dtype=float).copy())
        action_idx = policy_runner.sample_action()
        next_state = int(traj_rng.choice(transitions.shape[2],
                                         p=transitions[prev_state, action_idx]))
        policy_runner.receive_step_outcome(make_step_outcome(action_idx, next_state))
        prev_state = next_state
    dim = len(value)
    expansions = {
        "cache_version": np.array(3),
        "states": np.asarray(states, dtype=int),
        "targets": np.stack(targets),
        "achieved": np.stack(achieved),
        "onward_steps": np.asarray(onward_steps, dtype=int),
        "onward_states": np.asarray(onward_states, dtype=int),
        "onward_actions": np.asarray(onward_actions, dtype=int),
        "onward_values": np.stack(onward_values) if onward_values else np.empty((0, dim)),
    }
    return PolicyResult(feasible=True, optimum=optimum, optimal_value=value,
                        trajectory=trajectory, expansions=expansions)


class BaseRunner(ABC):
    """Template-method experiment runner shared across problem domains."""

    def __init__(self, config):
        self.config = config

    # ------------------------------------------------------------------ #
    #  Domain-specific hooks
    # ------------------------------------------------------------------ #

    @abstractmethod
    def build_instance(self) -> Any:
        """Build the problem instance from ``self.config.instance``."""

    @abstractmethod
    def make_grid(self, instance) -> Grid:
        """Build the value-space ``Grid`` for ``instance``."""

    @abstractmethod
    def make_engine(
        self, instance, grid: Grid, mode: VertexSetMode,
    ) -> ValueSetEngine:
        """Build a configured ``ValueSetEngine`` for ``instance``."""

    @abstractmethod
    def save_instance(self, instance, directory: Path) -> None:
        """Persist the problem instance to ``directory``."""

    @abstractmethod
    def make_step_outcome(self, action_idx: int, next_state: int):
        """Build a domain-appropriate ``StepOutcome``."""

    @abstractmethod
    def compute_summary(self, result: RunResult) -> dict:
        """Domain-specific summary dictionary for ``result``."""

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def run(self) -> RunResult:
        """Build instance → run VSI → save artifacts → return ``RunResult``."""
        directory = run_dir(self.config.output.output_dir, self.config.run_label())
        clear_run_artifacts(directory)
        logger.info("Starting run: %s", directory.name)

        if self.config.output.save_config:
            save_config(self.config, directory)

        instance = self.build_instance()
        result = self.run_algorithm(instance)

        self.save_results(result, directory)
        save_summary(self.compute_summary(result), directory)

        logger.info("Run complete: %s (%.2fs)", directory.name, result.runtime)
        return result

    def run_algorithm(self, instance) -> RunResult:
        """Run VSI on ``instance`` and return a timed ``RunResult``."""
        ac = self.config.algorithm
        mode = ac.mode
        grid = self.make_grid(instance)
        engine = self.make_engine(instance, grid, mode)
        runner = IterationRunner(engine)
        init_value_sets = uniform_value_sets(
            SEED_FN[mode](grid), instance.num_states,
        )

        t_start = time.perf_counter()
        runner.run(
            init_value_sets=init_value_sets,
            max_iters=ac.max_iters,
            keep_history=self.config.output.save_history,
        )
        runtime = time.perf_counter() - t_start

        converged = runner.status == RunnerStatus.CONVERGED
        history = list(runner.sequence) if runner.sequence else []

        logger.info(
            "Experiment completed: %d iterations in %.3f seconds (status: %s)",
            runner.num_iterations, runtime, runner.status.name,
        )

        return RunResult(
            config=self.config,
            instance=instance,
            final_value_sets=runner.result if converged else None,
            history=history,
            runtime=runtime,
            num_iterations=runner.num_iterations,
            converged=converged,
            status=runner.status,
        )

    def save_results(self, result: RunResult, directory: Path) -> None:
        """Persist instance / history / final value sets per the output config."""
        out = self.config.output
        if out.save_instance:
            self.save_instance(result.instance, directory)
        if out.save_history:
            for i, vs in enumerate(result.history):
                save_value_sets_iteration(
                    value_sets_to_point_lists(vs), i, directory,
                )
        if out.save_final and result.final_value_sets is not None:
            save_final_value_sets(
                value_sets_to_point_lists(result.final_value_sets), directory,
            )

    def run_policy(self, instance, value_sets=None) -> PolicyResult:
        """Extract the optimal policy and roll it out.

        Policy parameters are read from ``self.config.policy``.
        Converged ``value_sets`` may be passed in to skip the expensive
        value-set iteration; the caller is responsible for their convergence.
        """
        pc = self.config.policy
        mode = self.config.algorithm.mode
        grid = self.make_grid(instance)
        obj = pc.objective_array(grid.dim)
        engine = self.make_engine(instance, grid, mode)

        if value_sets is not None:
            avs = copy.deepcopy(value_sets)
            logger.info("Reusing pre-computed achievable value sets.")
        else:
            vsi_runner = IterationRunner(engine)
            vsi_runner.run(
                init_value_sets=uniform_value_sets(
                    SEED_FN[mode](grid), instance.num_states,
                ),
                max_iters=self.config.algorithm.max_iters,
            )
            if vsi_runner.status != RunnerStatus.CONVERGED:
                logger.warning("VSI did not converge; cannot extract policy.")
                return PolicyResult(feasible=False)
            avs = vsi_runner.result

        return rollout_policy(engine, instance, avs, obj, pc.init_state,
                              pc.num_steps, pc.seed, self.make_step_outcome)

    def run_singleton(self, max_iters: int = 100) -> SingletonResult:
        """Run singleton (single-point) VSI and record the trajectory."""
        self.config = replace(
            self.config,
            algorithm=replace(
                self.config.algorithm,
                mode=VertexSetMode.SINGLE_POINT, max_iters=max_iters,
            ),
        )
        directory = run_dir(self.config.output.output_dir, self.config.run_label())
        clear_run_artifacts(directory)
        if self.config.output.save_config:
            save_config(self.config, directory)
        instance = self.build_instance()
        mode = VertexSetMode.SINGLE_POINT
        grid = self.make_grid(instance)
        engine = self.make_engine(instance, grid, mode)
        runner = IterationRunner(engine)
        init_value_sets = uniform_value_sets(
            SEED_FN[mode](grid), instance.num_states,
        )

        t_start = time.perf_counter()
        runner.run(
            init_value_sets=init_value_sets,
            max_iters=max_iters,
            detect_loops=True,
        )
        runtime = time.perf_counter() - t_start

        sequence = runner.sequence or []
        saved_result = RunResult(
            config=self.config,
            instance=instance,
            final_value_sets=(
                runner.result if runner.status == RunnerStatus.CONVERGED else None
            ),
            history=sequence if self.config.output.save_history else [],
            runtime=runtime,
            num_iterations=runner.num_iterations,
            converged=(runner.status == RunnerStatus.CONVERGED),
            status=runner.status,
        )
        self.save_results(saved_result, directory)
        summary = self.compute_summary(saved_result)
        summary["loop_detected"] = runner.status == RunnerStatus.LOOP_DETECTED
        summary["loop_iteration"] = runner.loop_back_iter
        save_summary(summary, directory)

        history = singleton_vs_to_history(
            sequence, grid.dim,
        )
        logger.info(
            "Singleton experiment: %d snapshots in %.3f seconds",
            len(history), runtime,
        )

        return SingletonResult(
            config=self.config,
            instance=instance,
            history=history,
            num_iterations=runner.num_iterations,
            runtime=runtime,
            converged=(runner.status == RunnerStatus.CONVERGED),
            loop_detected=(runner.status == RunnerStatus.LOOP_DETECTED),
            loop_iteration=runner.loop_back_iter,
        )


# --------------------------------------------------------------------------- #
#  Parameter sweep helper
# --------------------------------------------------------------------------- #

def run_parameter_sweep(
    base_config: C,
    param_name: str,
    param_values: list,
    run_fn: Callable[[C], R],
    set_param: Callable[[C, Any], C],
) -> list[R]:
    """Run ``run_fn`` once per parameter value.

    Each call receives a deep-copied ``base_config`` with the swept
    parameter applied by ``set_param``.
    """
    results: list[R] = []
    for val in param_values:
        cfg = set_param(copy.deepcopy(base_config), val)
        logger.info("=== %s = %s ===", param_name, val)
        results.append(run_fn(cfg))
    return results
