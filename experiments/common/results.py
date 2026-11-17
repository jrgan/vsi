"""
Shared result dataclasses for experiments.

A single ``RunResult`` / ``SingletonResult`` / ``PolicyResult`` is used
across all problem domains; the problem instance is stored in the
``instance`` field as an opaque object.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from vsi import RunnerStatus

if TYPE_CHECKING:
    from experiments.common.config import BaseExperimentConfig


@dataclass
class _BaseResult:
    """Fields common to every experiment result."""
    config: Optional["BaseExperimentConfig"] = None
    instance: object = None
    runtime: float = 0.0
    num_iterations: int = 0
    converged: bool = False


@dataclass
class RunResult(_BaseResult):
    """Result of a full value-iteration run."""
    final_value_sets: object = None
    history: list = field(default_factory=list)
    status: RunnerStatus = RunnerStatus.INITIATED


@dataclass
class SingletonResult(_BaseResult):
    """Result of a singleton (single-point) iteration.

    ``loop_iteration`` is the iteration index the run looped back to
    (i.e. where the detected cycle starts), or ``None`` if no loop was
    detected.
    """
    history: list[list[np.ndarray]] = field(default_factory=list)
    loop_detected: bool = False
    loop_iteration: int | None = None


@dataclass
class PolicyResult:
    """Result of policy extraction and execution."""
    feasible: bool = False
    optimum: float | None = None
    optimal_value: np.ndarray | None = None
    trajectory: list = field(default_factory=list)
    expansions: dict[str, np.ndarray] | None = None


def value_sets_to_point_lists(vs) -> list[np.ndarray]:
    """Convert any ValueSets representation to a list of point arrays."""
    if vs is None:
        return []
    return [np.array(pts) for pts in vs]


def singleton_vs_to_history(sequence, reward_dim: int) -> list[list[np.ndarray]]:
    """Convert ``IterationRunner.sequence`` to per-state singleton history.

    Each entry becomes a list of 1-D arrays, one per state.  Empty value
    sets are replaced by NaN vectors.
    """
    history: list[list[np.ndarray]] = []
    for vs in sequence:
        snapshot = []
        for s in range(len(vs)):
            pts = np.array(vs[s])
            if pts.size == 0:
                snapshot.append(np.full(reward_dim, np.nan))
            else:
                snapshot.append(pts.reshape(-1))
        history.append(snapshot)
    return history

