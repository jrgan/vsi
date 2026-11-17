"""Markov game experiment runner.

Adds Markov-game-specific hooks to
:class:`experiments.common.runner.BaseRunner` and exposes helper
functions used by views and demos.
"""
from __future__ import annotations

import logging
from pathlib import Path

from markovgame import (
    MarkovGame, MarkovGameLPFactory, StepOutcome, markovgame_grid,
)
from vsi import ValueSetEngine, Grid

from experiments.common.config import VertexSetMode
from experiments.common.results import PolicyResult, RunResult, SingletonResult
from experiments.common.runner import BaseRunner, ENGINE_BY_MODE, run_parameter_sweep
from experiments.markovgame.config import ExperimentConfig
from experiments.markovgame.instance import generate_instance, save_instance
from experiments.markovgame.summary import compute_summary

logger = logging.getLogger(__name__)


class MGExperimentRunner(BaseRunner):
    """Experiment runner for Markov game problems."""

    config: ExperimentConfig

    def build_instance(self) -> MarkovGame:
        ic = self.config.instance
        logger.info(
            "Building MarkovGame (method=%s): S=%d, N=%d, A_per_agent=%d, "
            "gamma=%.2f, seed=%s",
            ic.method, ic.num_states, ic.num_agents,
            ic.num_actions_per_agent, ic.gamma, ic.seed,
        )
        return generate_instance(ic)

    def make_grid(self, instance: MarkovGame) -> Grid:
        ac = self.config.algorithm
        return markovgame_grid(instance, eps=ac.eps, delta=ac.delta)

    def make_engine(
        self, instance: MarkovGame, grid: Grid, mode: VertexSetMode,
    ) -> ValueSetEngine:
        ac = self.config.algorithm
        engine = ENGINE_BY_MODE[mode](
            MarkovGameLPFactory(instance), grid,
        )
        engine.set_param(vsi_operator_parallel=ac.parallel_by_state)
        return engine

    def save_instance(self, instance: MarkovGame, directory: Path) -> None:
        save_instance(instance, directory)

    def make_step_outcome(self, action_idx: int, next_state: int) -> StepOutcome:
        return StepOutcome(
            action=action_idx, state=next_state, action_played=action_idx,
        )

    def compute_summary(self, result: RunResult) -> dict:
        return compute_summary(result)


# --------------------------------------------------------------------------- #
#  Convenience functional API
# --------------------------------------------------------------------------- #

def run_experiment(config: ExperimentConfig) -> RunResult:
    """Execute a single Markov game experiment run."""
    return MGExperimentRunner(config).run()


def run_policy_experiment(
    config: ExperimentConfig,
    reuse_value_sets: bool = True,
) -> tuple[RunResult, PolicyResult]:
    """Run a Markov game experiment then extract and execute the optimal policy."""
    runner = MGExperimentRunner(config)
    run_result = runner.run()
    if not run_result.converged:
        logger.warning("VSI did not converge; skipping policy extraction.")
        return run_result, PolicyResult(feasible=False)
    policy_result = runner.run_policy(
        run_result.instance,
        value_sets=run_result.final_value_sets if reuse_value_sets else None,
    )
    return run_result, policy_result


# Scaling axes: name -> (label prefix used in run names, InstanceConfig attribute).
_SCALING_AXES: dict[str, tuple[str, str]] = {
    "states": ("S", "num_states"),
    "actions": ("A", "num_actions_per_agent"),
    "num_agents": ("N", "num_agents"),
}


def run_scaling_experiment(
    base_config: ExperimentConfig,
    axis: str,
    sizes: list[int],
    *,
    singleton: bool = False,
    max_iters: int = 1000,
) -> list[RunResult] | list[SingletonResult]:
    """Sweep a problem-size axis and collect runtime / iteration counts.

    Args:
        base_config: Template config; the swept attribute is overwritten per size.
        axis: One of ``"states"``, ``"actions"`` (actions per agent), or
            ``"num_agents"``.
        sizes: Values to sweep along ``axis``.
        singleton: If ``True``, run the singleton inner loop (cheaper) and return
            :class:`SingletonResult` objects; otherwise run the full VSI engine
            and return :class:`RunResult` objects.
        max_iters: Iteration cap for the singleton runs (ignored when
            ``singleton`` is ``False``).

    Returns:
        One result per size, in the order of ``sizes``.
    """
    try:
        prefix, attr = _SCALING_AXES[axis]
    except KeyError:
        raise ValueError(
            f"Unknown scaling axis {axis!r}; expected one of {sorted(_SCALING_AXES)}."
        ) from None

    def _set(cfg: ExperimentConfig, n: int) -> ExperimentConfig:
        setattr(cfg.instance, attr, n)
        cfg.name = f"{base_config.name}_{prefix}{n}"
        return cfg

    if singleton:
        def run_fn(cfg: ExperimentConfig) -> SingletonResult:
            return MGExperimentRunner(cfg).run_singleton(max_iters=max_iters)
    else:
        run_fn = run_experiment

    return run_parameter_sweep(base_config, axis, sizes, run_fn, _set)


def run_singleton_experiment(
    config: ExperimentConfig,
    max_iters: int = 100,
) -> SingletonResult:
    """Run singleton value-set iteration and record the trajectory."""
    return MGExperimentRunner(config).run_singleton(max_iters=max_iters)
