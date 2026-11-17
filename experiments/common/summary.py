"""
Problem-agnostic summary computation helpers.

Provides reusable functions for summarizing value sets, convergence
histories, and runtime performance, plus :func:`compute_summary` which
composes them with domain-specific instance / algorithm summaries.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from experiments.common.results import RunResult, value_sets_to_point_lists


# =========================================================================== #
#  Top-level summary composer
# =========================================================================== #

def compute_summary(
    result: RunResult,
    instance_summary: Callable[[object], dict],
    algorithm_summary: Callable[[object], dict],
) -> dict:
    """Build a standard summary dictionary for any ``RunResult``.

    ``instance_summary`` and ``algorithm_summary`` are passed the
    corresponding sub-configs (``result.config.instance`` and
    ``result.config.algorithm``) and return domain-specific dicts.
    """
    final_pts = value_sets_to_point_lists(result.final_value_sets)
    summary = {
        "name": result.config.name,
        "uid": result.config.uid(),
        "instance": instance_summary(result.config.instance),
        "algorithm": algorithm_summary(result.config.algorithm),
        "performance": performance_summary(result),
        "final_value_sets": value_sets_summary(final_pts),
    }
    if result.history:
        history_pts = [value_sets_to_point_lists(h) for h in result.history]
        summary["convergence"] = convergence_summary(history_pts)
    return summary


# =========================================================================== #
#  Performance summary
# =========================================================================== #

def performance_summary(result) -> dict:
    """Summarize runtime performance from any result with the standard fields.

    Expects *result* to have ``runtime``, ``num_iterations``, ``converged``,
    and ``status`` attributes (as provided by ``BaseRunResult``).
    """
    return {
        "runtime_seconds": result.runtime,
        "num_iterations": result.num_iterations,
        "converged": result.converged,
        "status": result.status.name if hasattr(result.status, "name") else str(result.status),
        "seconds_per_iteration": (
            result.runtime / result.num_iterations
            if result.num_iterations > 0 else 0.0
        ),
    }


# =========================================================================== #
#  Value-set summary
# =========================================================================== #

def value_sets_summary(value_sets: list[np.ndarray]) -> dict:
    """Summarize a collection of value sets (list of point arrays).

    Returns per-state statistics (point count, min, max, mean, area for 2-D).
    """
    point_counts = [len(vs) for vs in value_sets]
    states: dict[str, dict] = {}
    for i, vs in enumerate(value_sets):
        info: dict = {"num_points": len(vs)}
        if len(vs) > 0:
            info["min"] = vs.min(axis=0).tolist()
            info["max"] = vs.max(axis=0).tolist()
            info["mean"] = vs.mean(axis=0).tolist()
            if vs.shape[1] == 2:
                info["approx_area"] = convex_hull_area_2d(vs)
        states[f"state_{i}"] = info

    return {
        "num_states": len(value_sets),
        "total_points": sum(point_counts),
        "points_per_state": point_counts,
        "states": states,
    }


# =========================================================================== #
#  Convergence summary
# =========================================================================== #

def convergence_summary(history: list[list[np.ndarray]]) -> dict:
    """Summarize convergence across iterations.

    Tracks how point counts and value set volumes change per iteration.
    """
    iterations = []
    for i, value_sets in enumerate(history):
        counts = [len(vs) for vs in value_sets]
        entry: dict = {
            "iteration": i,
            "total_points": sum(counts),
            "points_per_state": counts,
        }
        if value_sets and value_sets[0].ndim == 2 and value_sets[0].shape[1] == 2:
            areas = [convex_hull_area_2d(vs) for vs in value_sets]
            entry["areas_per_state"] = areas
        iterations.append(entry)
    return {"iterations": iterations}


# =========================================================================== #
#  Grid helpers
# =========================================================================== #

def convex_hull_area_2d(points: np.ndarray) -> float:
    """Compute the area of the convex hull of 2D points.

    Returns 0.0 if fewer than 3 points or on error.
    """
    if len(points) < 3:
        return 0.0
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        return float(hull.volume)  # In 2D, volume = area
    except Exception:
        return 0.0

