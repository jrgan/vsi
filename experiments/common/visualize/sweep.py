"""
Bound-sweep and scaling plots.  Problem-agnostic.

Functions accept plain lists/arrays rather than domain-specific result
objects so that any experiment runner can use them.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.spatial import ConvexHull, QhullError
from vsi.vertex_rep.polytope import convex_hull_vertices

from experiments.common.visualize._legend import _best_corner_legend


def plot_bound_sweep(
    sweep_point_lists: list[list[np.ndarray]],
    sweep_labels: list[str],
    title: str = "Effect of Constraint Bound",
) -> Figure:
    """
    Overlay value-set polytopes from a parameter sweep, one subplot per state.

    Each sweep entry is drawn in a different colour, without vertex markers;
    state panels share limits.

    Args:
        sweep_point_lists: ``sweep_point_lists[b][s]`` is the point array
            for state *s* at sweep index *b*.
        sweep_labels: Human-readable label for each sweep entry
            (e.g. ``"B=5.00 (12 vtx)"``).
        title: Overall figure title.

    Returns:
        The matplotlib Figure.
    """
    if not sweep_point_lists:
        fig, ax = plt.subplots()
        ax.set_title(title + " (no data)")
        return fig

    num_states = max(len(pl) for pl in sweep_point_lists)
    if num_states == 0:
        fig, ax = plt.subplots()
        ax.set_title(title + " (no data)")
        return fig

    n_entries = len(sweep_point_lists)

    cols = min(num_states, 4)
    rows = (num_states + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False,
        sharex=True, sharey=True,
    )

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=max(n_entries - 1, 1))

    for s in range(num_states):
        ax = axes[s // cols][s % cols]
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.set_aspect("equal", adjustable="box")

        for b_idx in range(n_entries):
            pl = sweep_point_lists[b_idx]
            vs = pl[s] if s < len(pl) else np.empty((0, 2))

            if len(vs) == 0 or vs.ndim < 2:
                continue
            if vs.shape[1] > 2:
                vs = vs[:, :2]
            elif vs.shape[1] < 2:
                continue

            vs = convex_hull_vertices(vs)
            if len(vs) == 0:
                continue

            color = cmap(norm(b_idx))
            fill_alpha = 0.15 + 0.15 * (b_idx / max(n_entries - 1, 1))
            label = sweep_labels[b_idx] if b_idx < len(sweep_labels) else f"#{b_idx}"

            if len(vs) >= 3:
                try:
                    hull = ConvexHull(vs)
                    hull_idx = np.append(hull.vertices, hull.vertices[0])
                    ax.fill(
                        vs[hull_idx, 0], vs[hull_idx, 1],
                        alpha=fill_alpha, color=color,
                    )
                    ax.plot(
                        vs[hull_idx, 0], vs[hull_idx, 1],
                        color=color, linewidth=1.3, label=label,
                    )
                except QhullError:
                    ax.plot(vs[:, 0], vs[:, 1], color=color,
                            linewidth=1.3, label=label)
            elif len(vs) == 2:
                ax.plot(vs[:, 0], vs[:, 1], color=color, linewidth=1.3, label=label)
            else:
                ax.scatter(vs[:, 0], vs[:, 1], c=[color], s=10, label=label, zorder=4)

        ax.set_title(f"State {s}")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        _best_corner_legend(ax, fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(num_states, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_scaling(
    sizes: list[int],
    times: list[float],
    iterations: list[int],
    title: str = "Runtime & Iterations vs. Problem Size",
    xlabel: str = "Number of States",
) -> Figure:
    """
    Plot runtime and iteration count vs. problem size.

    Args:
        sizes: Problem sizes (e.g. number of states or actions).
        times: Runtime in seconds for each size.
        iterations: Number of iterations for each size.
        title: Plot title.
        xlabel: Label for the x-axis.

    Returns:
        The matplotlib Figure.
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

    color1 = "tab:blue"
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Runtime (seconds)", color=color1)
    ax1.plot(sizes, times, "o-", color=color1, linewidth=2, markersize=6, label="Runtime")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("Iterations", color=color2)
    ax2.plot(sizes, iterations, "s--", color=color2, linewidth=2, markersize=6, label="Iterations")
    ax2.tick_params(axis="y", labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    _best_corner_legend(ax1, handles=lines1 + lines2, labels=labels1 + labels2, fontsize=8)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig
