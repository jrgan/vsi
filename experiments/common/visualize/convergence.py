"""
Convergence and iteration-snapshot plots.  Problem-agnostic.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.spatial import ConvexHull
from experiments.common.visualize._legend import _best_corner_legend
from experiments.common.visualize.style import animation_axes, shared_bounds, style_state_axis


def _outline(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        return np.empty((0, 2))
    points = points[:, :2]
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) >= 3:
        try:
            return points[ConvexHull(points).vertices]
        except Exception:
            pass
    return points


def plot_convergence(
    history: list[list[np.ndarray]],
    title: str = "Convergence",
) -> Figure:
    """
    Plot how value sets shrink across iterations.

    Shows total point count and per-state point counts over iterations.

    Args:
        history: List of value-set snapshots, one per iteration.
        title: Plot title.

    Returns:
        The matplotlib Figure.
    """
    n_iters = len(history)
    if n_iters == 0:
        fig, ax = plt.subplots()
        ax.set_title(title + " (no data)")
        return fig

    n_states = len(history[0])
    iters = list(range(n_iters))
    totals = [sum(len(vs) for vs in snap) for snap in history]
    per_state = [[len(snap[s]) for snap in history] for s in range(n_states)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(iters, totals, "o-", linewidth=2, markersize=4)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Total Points")
    ax1.set_title("Total Points Across Iterations")
    ax1.grid(True, alpha=0.3)

    for s in range(n_states):
        ax2.plot(iters, per_state[s], "o-", markersize=3, label=f"State {s}")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Points")
    ax2.set_title("Points per State")
    _best_corner_legend(ax2, fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_iteration_snapshots_2d(
    history: list[list[np.ndarray]],
    iterations: Optional[list[int]] = None,
    title: Optional[str] = None,
) -> Figure:
    """
    Plot 2D value-set evolution with shared axis limits across state subplots.

    Each iteration's convex hull is overlaid on the same axes, coloured
    from light (early) to dark (late) so that the shrinkage across
    iterations is visible at a glance.

    Args:
        history: ``history[t][s]`` is the point array for state *s* at
            iteration *t*.
        iterations: Which iterations to show.  Defaults to all.
        title: Overall figure title.

    Returns:
        The matplotlib Figure.
    """
    if iterations is None:
        iterations = list(range(len(history)))

    n_iters = len(iterations)
    n_states = len(history[0])

    fig, axes = animation_axes(n_states)
    outlines = [[_outline(history[it][s]) for s in range(n_states)] for it in iterations]
    xlim, ylim = shared_bounds(p for snapshot in outlines for p in snapshot)
    cmap = plt.cm.Blues
    norm = plt.Normalize(vmin=-max(n_iters - 1, 1),
                         vmax=2 * max(n_iters - 1, 1))
    outer_fill_alpha = 0.15
    inner_layer_alpha = 1 - (1 - 0.4) ** (1 / max(n_iters - 1, 1))

    for s in range(n_states):
        ax = axes[s]
        style_state_axis(ax, s, xlim, ylim)

        for idx, it in enumerate(iterations):
            vs = outlines[idx][s]
            if not len(vs):
                continue
            color = cmap(min(norm(idx) + 0.15, 1.0))
            boundary_color = "blue" if idx == n_iters - 1 else color
            if len(vs) >= 3:
                ax.fill(vs[:, 0], vs[:, 1], color=color,
                        alpha=outer_fill_alpha if idx == 0 else inner_layer_alpha,
                        zorder=2 + idx)
                closed = np.vstack((vs, vs[0]))
                ax.plot(closed[:, 0], closed[:, 1], color=boundary_color,
                        lw=1.3 if idx == n_iters - 1 else 0.65, zorder=3 + idx)
            elif len(vs) == 2:
                ax.plot(vs[:, 0], vs[:, 1], color=boundary_color,
                        lw=1.3 if idx == n_iters - 1 else 0.65, zorder=3 + idx)
            else:
                ax.scatter(vs[:, 0], vs[:, 1], color=boundary_color,
                           s=15 if idx == n_iters - 1 else 5, zorder=3 + idx)

    fig.suptitle(
        title or "Value Set Evolution Across Iterations",
        fontsize=18,
    )
    return fig
