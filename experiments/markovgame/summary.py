"""Markov game experiment summary computation."""
from __future__ import annotations

from experiments.common.summary import compute_summary as _compute_summary
from experiments.common.results import RunResult
from experiments.markovgame.config import AlgorithmConfig, InstanceConfig


def compute_summary(result: RunResult) -> dict:
    """Compute summary statistics from a Markov game run result."""
    return _compute_summary(result, _instance_summary, _algorithm_summary)


def _instance_summary(ic: InstanceConfig) -> dict:
    return {
        "num_states": ic.num_states,
        "num_agents": ic.num_agents,
        "num_actions_per_agent": ic.num_actions_per_agent,
        "gamma": ic.gamma,
        "seed": ic.seed,
    }


def _algorithm_summary(ac: AlgorithmConfig) -> dict:
    return {
        "eps": ac.eps,
        "delta": ac.delta,
        "max_iters": ac.max_iters,
        "parallel_by_state": ac.parallel_by_state,
    }

