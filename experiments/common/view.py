"""Base view class and plot-from-artifacts helper for experiments.

``BaseView`` separates computation (``run``) from persistence
(``run_and_save``). Subclasses return ``{name: Figure}``; the base class
saves each figure to the run directory in configured formats.

The plot-from-directory helpers re-render regular and singleton runs from saved
artifacts.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np

from matplotlib.figure import Figure

from experiments.common.io import (
    list_history_iterations,
    load_config_dict,
    load_final_value_sets,
    load_history,
    load_summary,
    load_trajectory,
    load_value_sets_iteration,
    run_dir,
)
from experiments.common.visualize import (
    plot_convergence,
    plot_iteration_snapshots_2d,
    plot_policy_trajectory,
    plot_singleton_trajectory,
)
from experiments.common.results import singleton_vs_to_history
from experiments.common.visualize.style import style_figure

logger = logging.getLogger(__name__)


class BaseView(ABC):
    """Base class for all view (figure-producing) experiments.

    Attributes:
        config: The experiment configuration (problem-specific subclass).
    """

    def __init__(self, config) -> None:
        self.config = config
        self._output_dir: Path | None = None

    @property
    def output_dir(self) -> Path:
        """Return the canonical run directory (shared with the runner)."""
        if self._output_dir is None:
            self._output_dir = run_dir(
                self.config.output.output_dir,
                self.config.run_label(),
            )
        return self._output_dir

    @abstractmethod
    def run(self) -> dict[str, Figure]:
        """Execute the view.

        Returns:
            Mapping from figure name (without extension) to matplotlib
            Figure objects.
        """
        ...

    def run_and_save(self, *, generate_animations: bool = True) -> Path:
        """Run the view and save all figures to the output directory."""
        figures = self.run()
        out = self.config.output
        for name, fig in figures.items():
            style_figure(fig, fixed_layout=name in {"iteration_snapshots", "policy_trajectory"})
            for fmt in out.image_formats:
                path = self.output_dir / f"{name}.{fmt}"
                fig.savefig(str(path), dpi=out.image_dpi, format=fmt)
                logger.info("Saved %s", path)
        if (generate_animations and "iteration_snapshots" in figures and
                list_history_iterations(self.output_dir)):
            from experiments.common.animate_value_sets import animate as animate_iterations
            animate_iterations(self.output_dir, self.output_dir / "iteration_animation.gif")
        if (generate_animations and "policy_trajectory" in figures and
                (self.output_dir / "final_value_sets.npz").exists()):
            from experiments.common.animate_policy_trajectory import animate as animate_policy
            animate_policy(self.output_dir, self.output_dir / "policy_trajectory_animation.gif")
        logger.info("All outputs saved to %s", self.output_dir)
        return self.output_dir


# --------------------------------------------------------------------------- #
#  Plot-from-artifacts (domain-agnostic)
# --------------------------------------------------------------------------- #

def plot_single_run_from_dir(
    input_dir: str | Path,
    *,
    num_states: int | None = None,
    title_prefix: str = "",
) -> dict[str, Figure]:
    """Re-render single-run figures from previously saved artifacts.

    Reads ``final_value_sets.npz``, ``history/iter_*.npz``,
    ``trajectory.npz``, and saved policy targets when available. Returns the
    same ``{name: Figure}`` mapping for whichever artifacts are present.

    Args:
        input_dir: Directory containing the saved artifacts.
        num_states: Number of states; inferred from final sets, saved config,
            or the trajectory when omitted.
        title_prefix: Optional string prepended to figure titles
            (e.g. ``"MG "`` for Markov game artifacts).
    """
    input_dir = Path(input_dir)
    figures: dict[str, Figure] = {}

    point_lists = None
    if (input_dir / "final_value_sets.npz").exists():
        point_lists = load_final_value_sets(input_dir)

    iterations = list_history_iterations(input_dir)
    if iterations:
        history = [load_value_sets_iteration(i, input_dir) for i in iterations]
        figures["convergence"] = plot_convergence(
            history, title=f"{title_prefix}Convergence".strip(),
        )
        figures["iteration_snapshots"] = plot_iteration_snapshots_2d(history)

    if (input_dir / "trajectory.npz").exists():
        trajectory = load_trajectory(input_dir)
        if trajectory:
            onward_values = target_values = None
            expansions_path = input_dir / "policy_expansions.npz"
            if expansions_path.exists():
                with np.load(expansions_path) as expansions:
                    onward_values = expansions["onward_values"]
                    if "targets" in expansions:
                        target_values = expansions["targets"]
            ns = num_states
            if ns is None and point_lists is None and (input_dir / "config.json").exists():
                ns = load_config_dict(input_dir)["instance"]["num_states"]
            figures["policy_trajectory"] = plot_policy_trajectory(
                trajectory,
                value_sets=point_lists,
                num_states=ns,
                title=f"{title_prefix}Policy Trajectory".strip(),
                onward_values=onward_values,
                target_values=target_values,
            )

    for name, fig in figures.items():
        style_figure(fig, fixed_layout=name in {"iteration_snapshots", "policy_trajectory"})
    return figures


def plot_singleton_from_dir(
    input_dir: str | Path,
    *,
    title: str = "Singleton Iteration Trajectory",
) -> dict[str, Figure]:
    """Replot a singleton run from saved history and summary, without solving."""
    input_dir = Path(input_dir)
    history = load_history(input_dir)
    if not history:
        raise ValueError("No singleton history found; run with save_history=True")
    summary = load_summary(input_dir)
    background_dir = input_dir / "background"
    background = (
        load_final_value_sets(background_dir)
        if (background_dir / "final_value_sets.npz").exists() else None
    )
    reward_dim = history[0][0].shape[1]
    figures = {
        "singleton_trajectory": plot_singleton_trajectory(
            singleton_vs_to_history(history, reward_dim),
            title=title,
            value_sets=background,
            converged=summary["performance"]["converged"],
            loop_detected=summary["loop_detected"],
        ),
    }
    style_figure(figures["singleton_trajectory"])
    return figures
