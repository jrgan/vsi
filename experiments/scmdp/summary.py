"""SCMDP experiment summary computation."""
from __future__ import annotations

from experiments.common.summary import compute_summary as _compute_summary
from experiments.common.results import RunResult
from experiments.scmdp.config import AlgorithmConfig, InstanceConfig


def compute_summary(result: RunResult) -> dict:
    """Compute summary statistics from an SCMDP run result."""
    return _compute_summary(result, _instance_summary, _algorithm_summary)


def _instance_summary(ic: InstanceConfig) -> dict:
    return {
        "num_states": ic.num_states,
        "num_actions": ic.num_actions,
        "reward_dim": ic.reward_dim,
        "gamma": ic.gamma,
        "constraint_bound": ic.bound,
        "seed": ic.seed,
    }


def _algorithm_summary(ac: AlgorithmConfig) -> dict:
    return {
        "eps": ac.eps,
        "delta": ac.delta,
        "max_iters": ac.max_iters,
        "parallel_by_state": ac.parallel_by_state,
    }
