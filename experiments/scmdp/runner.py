"""SCMDP experiment runner.

Adds SCMDP-specific hooks to
:class:`experiments.common.runner.BaseRunner` and exposes helper
functions used by views and demos.
"""
from __future__ import annotations

import logging
from pathlib import Path

from scmdp import SCMDP, SCMDPLPFactory, scmdp_grid
from vsi import ValueSetEngine, Grid
from vsi.value_set_engine import StepOutcome

from experiments.common.config import VertexSetMode
from experiments.common.results import PolicyResult, RunResult, SingletonResult
from experiments.common.runner import BaseRunner, ENGINE_BY_MODE, run_parameter_sweep
from experiments.scmdp.config import ExperimentConfig
from experiments.scmdp.instance import generate_instance, save_instance
from experiments.scmdp.summary import compute_summary

logger = logging.getLogger(__name__)


class SCMDPExperimentRunner(BaseRunner):
    """Experiment runner for SCMDP problems."""

    config: ExperimentConfig

    def build_instance(self) -> SCMDP:
        ic = self.config.instance
        logger.info(
            "Building SCMDP (method=%s): S=%d, A=%d, d=%d, gamma=%.2f, "
            "seed=%s, reward_min=%.2f, reward_max=%.2f, bound=%.2f",
            ic.method, ic.num_states, ic.num_actions, ic.reward_dim,
            ic.gamma, ic.seed, ic.reward_min, ic.reward_max, ic.bound,
        )
        return generate_instance(ic)

    def make_grid(self, instance: SCMDP) -> Grid:
        ac = self.config.algorithm
        return scmdp_grid(instance, eps=ac.eps, delta=ac.delta)

    def make_engine(
        self, instance: SCMDP, grid: Grid, mode: VertexSetMode,
    ) -> ValueSetEngine:
        ac = self.config.algorithm
        engine = ENGINE_BY_MODE[mode](
            SCMDPLPFactory(instance), grid,
        )
        engine.set_param(vsi_operator_parallel=ac.parallel_by_state)
        return engine

    def save_instance(self, instance: SCMDP, directory: Path) -> None:
        save_instance(instance, directory)

    def make_step_outcome(self, action_idx: int, next_state: int) -> StepOutcome:
        return StepOutcome(action_idx, next_state)

    def compute_summary(self, result: RunResult) -> dict:
        return compute_summary(result)


# --------------------------------------------------------------------------- #
#  Convenience functional API
# --------------------------------------------------------------------------- #

def run_experiment(config: ExperimentConfig) -> RunResult:
    """Execute a single SCMDP experiment run."""
    return SCMDPExperimentRunner(config).run()


def run_policy_experiment(
    config: ExperimentConfig,
    reuse_value_sets: bool = True,
) -> tuple[RunResult, PolicyResult]:
    """Run an SCMDP experiment then extract and execute the optimal policy."""
    runner = SCMDPExperimentRunner(config)
    run_result = runner.run()
    if not run_result.converged:
        logger.warning("VSI did not converge; skipping policy extraction.")
        return run_result, PolicyResult(feasible=False)
    policy_result = runner.run_policy(
        run_result.instance,
        value_sets=run_result.final_value_sets if reuse_value_sets else None,
    )
    return run_result, policy_result


def run_bound_sweep(
    base_config: ExperimentConfig,
    bounds: list[float],
) -> list[RunResult]:
    """Run experiments with a sweep of constraint bounds."""
    def _set_bound(cfg: ExperimentConfig, b: float) -> ExperimentConfig:
        cfg.instance.bound = b
        cfg.name = f"{base_config.name}_B{b}"
        return cfg
    return run_parameter_sweep(base_config, "Bound", bounds, run_experiment, _set_bound)


# Scaling axes: name -> (label prefix used in run names, InstanceConfig attribute).
_SCALING_AXES: dict[str, tuple[str, str]] = {
    "states": ("S", "num_states"),
    "actions": ("A", "num_actions"),
    "reward_dim": ("d", "reward_dim"),
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
        axis: One of ``"states"``, ``"actions"``, or ``"reward_dim"``.
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
            return SCMDPExperimentRunner(cfg).run_singleton(max_iters=max_iters)
    else:
        run_fn = run_experiment

    return run_parameter_sweep(base_config, axis, sizes, run_fn, _set)


def run_singleton_experiment(
    config: ExperimentConfig,
    max_iters: int = 100,
) -> SingletonResult:
    """Run singleton value-set iteration and record the trajectory."""
    return SCMDPExperimentRunner(config).run_singleton(max_iters=max_iters)
