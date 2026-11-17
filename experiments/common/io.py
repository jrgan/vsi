"""
Problem-agnostic experiment I/O utilities.

All I/O functions operate on a single ``directory`` argument (a
:class:`pathlib.Path` or string). Helper :func:`run_dir` composes a
canonical ``<output_dir>/<run_label>/`` path from a config and creates
it on disk.

Canonical layout for a run:

    <output_dir>/<run_label>/
        config.json
        summary.json
        instance.npz                  (problem-specific I/O)
        final_value_sets.npz           (only after convergence)
        trajectory.npz                (only if a policy was rolled out)
        history/iter_NNN.npz          (only if save_history=True)
        background/final_value_sets.npz  (optional singleton plot background)
        <figure>.{png,pdf}            (default formats, written by view classes)
"""
from __future__ import annotations

import json
import logging
from enum import Enum
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# =========================================================================== #
#  Path composition
# =========================================================================== #

def run_dir(output_dir, run_label: str) -> Path:
    """Return ``Path(output_dir) / run_label``, creating it if needed."""
    p = Path(output_dir) / run_label
    p.mkdir(parents=True, exist_ok=True)
    return p


def clear_run_artifacts(directory) -> None:
    """Remove known outputs before a rerun; leave unrelated files untouched."""
    directory = Path(directory)
    paths = [directory / name for name in (
        "config.json", "summary.json", "instance.npz",
        "final_value_sets.npz", "trajectory.npz", "policy_expansions.npz",
        "background/final_value_sets.npz",
    )]
    paths.extend((directory / "history").glob("iter_*.npz"))
    for name in ("convergence", "snapshots", "trajectory", "iteration_snapshots",
                 "policy_trajectory", "singleton_trajectory"):
        paths.extend(directory / f"{name}.{fmt}" for fmt in ("png", "pdf", "svg"))
    paths.extend(directory / name for name in
                 ("value_set_iteration.gif", "policy_trajectory.gif",
                  "iteration_animation.gif", "policy_trajectory_animation.gif"))
    for path in paths:
        if path.is_file():
            path.unlink()


# =========================================================================== #
#  Config / summary
# =========================================================================== #

def save_config(config, directory) -> Path:
    """Save experiment configuration as JSON into ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.json"
    path.write_text(config.to_json())
    logger.info("Config saved to %s", path)
    return path


def load_config_dict(directory) -> dict:
    """Load experiment configuration as a raw dictionary."""
    return json.loads((Path(directory) / "config.json").read_text())


def load_config(directory, config_cls=None):
    """Load an experiment configuration as ``config_cls`` (or a dict)."""
    d = load_config_dict(directory)
    if config_cls is None:
        return d
    return config_cls.from_dict(d)


def save_summary(summary: dict, directory) -> Path:
    """Save summary statistics as JSON into ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "summary.json"
    path.write_text(json.dumps(summary, indent=2, default=_json_default))
    logger.info("Summary saved to %s", path)
    return path


def load_summary(directory) -> dict:
    """Load summary statistics from JSON."""
    return json.loads((Path(directory) / "summary.json").read_text())


# =========================================================================== #
#  Value-set I/O
# =========================================================================== #

def _value_sets_to_arrays(value_sets: list[np.ndarray]) -> dict:
    arrays = {f"state_{i}": np.asarray(vs) for i, vs in enumerate(value_sets)}
    arrays["num_states"] = np.array([len(value_sets)])
    return arrays


def _arrays_to_value_sets(data) -> list[np.ndarray]:
    n = int(data["num_states"][0])
    return [data[f"state_{i}"] for i in range(n)]


def save_final_value_sets(value_sets: list[np.ndarray], directory) -> Path:
    """Save final achievable value sets into ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "final_value_sets.npz"
    np.savez(path, **_value_sets_to_arrays(value_sets))
    logger.info("Final value sets saved to %s", path)
    return path


def load_final_value_sets(directory) -> list[np.ndarray]:
    """Load final achievable value sets."""
    data = np.load(Path(directory) / "final_value_sets.npz")
    return _arrays_to_value_sets(data)


def save_value_sets_iteration(
    value_sets: list[np.ndarray], iteration: int, directory,
) -> Path:
    """Save value sets from one iteration under ``directory/history/``."""
    hist_dir = Path(directory) / "history"
    hist_dir.mkdir(parents=True, exist_ok=True)
    path = hist_dir / f"iter_{iteration:03d}.npz"
    np.savez(path, **_value_sets_to_arrays(value_sets))
    return path


def load_value_sets_iteration(iteration: int, directory) -> list[np.ndarray]:
    """Load value sets for a single iteration."""
    path = Path(directory) / "history" / f"iter_{iteration:03d}.npz"
    return _arrays_to_value_sets(np.load(path))


def list_history_iterations(directory) -> list[int]:
    """List available iteration indices in ``directory/history/``."""
    hist_dir = Path(directory) / "history"
    if not hist_dir.exists():
        return []
    iters = []
    for f in sorted(hist_dir.glob("iter_*.npz")):
        try:
            iters.append(int(f.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(iters)


def load_history(directory) -> list[list[np.ndarray]]:
    """Load every saved iteration's value sets in order."""
    return [load_value_sets_iteration(i, directory)
            for i in list_history_iterations(directory)]


# =========================================================================== #
#  Trajectory I/O
# =========================================================================== #

def save_policy_expansions(expansions: dict[str, np.ndarray], directory) -> Path:
    """Save the exact target and onward values from a policy rollout."""
    path = Path(directory) / "policy_expansions.npz"
    np.savez(path, **expansions)
    return path


def save_trajectory(
    trajectory: list[tuple[int, np.ndarray]], directory,
) -> Path:
    """Save a policy rollout (list of ``(state, value_vector)`` pairs)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "trajectory.npz"
    if not trajectory:
        np.savez(path, states=np.empty((0,), dtype=int),
                 values=np.empty((0, 0), dtype=float))
        return path
    states = np.array([int(s) for s, _ in trajectory], dtype=int)
    values = np.stack([np.asarray(v, dtype=float) for _, v in trajectory], axis=0)
    np.savez(path, states=states, values=values)
    logger.info("Trajectory saved to %s", path)
    return path


def load_trajectory(directory) -> list[tuple[int, np.ndarray]]:
    """Load a trajectory saved by :func:`save_trajectory`."""
    data = np.load(Path(directory) / "trajectory.npz")
    states, values = data["states"], data["values"]
    return [(int(states[i]), values[i]) for i in range(len(states))]


# =========================================================================== #
#  JSON helpers
# =========================================================================== #

def _json_default(obj):
    """JSON serializer for numpy types and enums."""
    if isinstance(obj, Enum):
        return obj.name
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
