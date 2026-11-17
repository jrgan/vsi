"""
Value-set point-set plots (2D).  Problem-agnostic.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.spatial import ConvexHull
from vsi.vertex_rep.polytope import convex_hull_vertices

from experiments.common.visualize._legend import _best_corner_legend

logger = logging.getLogger(__name__)


def plot_point_sets_2d(
    point_sets: list[np.ndarray],
    title: str = "Value Sets",
    state_labels: Optional[list[str]] = None,
    ax: Optional[plt.Axes] = None,
    show_hull: bool = True,
    hide_interior_points: bool = True,
    alpha: float = 0.3,
) -> Figure:
    """
    Plot 2D value sets for each state.

    If the point arrays have more than two dimensions, the first two
    dimensions are used for plotting.

    Args:
        point_sets: List of arrays, each of shape (n_i, d) with d >= 2.
        title: Plot title.
        state_labels: Optional labels for each state.
        ax: Optional matplotlib Axes to plot on.
        show_hull: Whether to draw the convex hull boundary.
        alpha: Fill transparency for convex hulls.
        hide_interior_points: Whether to only show hull vertices (for clarity).

    Returns:
        The matplotlib Figure.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = ax.get_figure()

    colors = plt.cm.tab10.colors

    if hide_interior_points:
        point_sets = [convex_hull_vertices(points) for points in point_sets]

    for i, points in enumerate(point_sets):
        if len(points) == 0 or points.ndim < 2:
            continue

        if points.shape[1] > 2:
            points = points[:, :2]
        elif points.shape[1] < 2:
            logger.warning("State %d has %d dimensions; skipping.", i, points.shape[1])
            continue

        color = colors[i % len(colors)]
        label = state_labels[i] if state_labels else f"State {i}"
        ax.scatter(points[:, 0], points[:, 1], c=[color], s=12, label=label, zorder=3)

        if show_hull and len(points) >= 3:
            try:
                hull = ConvexHull(points)
                hull_pts = np.append(hull.vertices, hull.vertices[0])
                ax.fill(
                    points[hull_pts, 0], points[hull_pts, 1],
                    alpha=alpha, color=color,
                )
                ax.plot(
                    points[hull_pts, 0], points[hull_pts, 1],
                    color=color, linewidth=1.0,
                )
            except Exception:
                pass

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title(title)
    _best_corner_legend(ax, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_point_sets_by_state(
    point_sets: list[np.ndarray],
    title: str = "Value Sets by State",
) -> Figure:
    """
    Plot each state's 2D value set in a separate subplot with shared limits.

    Args:
        point_sets: List of arrays, each of shape (n_i, 2).
        title: Overall figure title.

    Returns:
        The matplotlib Figure.
    """
    n = len(point_sets)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(5 * cols, 4 * rows), sharex=True, sharey=True,
    )
    axes = np.atleast_1d(axes).flatten()

    for i, vs in enumerate(point_sets):
        axes[i].tick_params(labelbottom=True, labelleft=True)
        plot_point_sets_2d(
            [vs], title=f"State {i} ({len(vs)} pts)",
            state_labels=[f"State {i}"], ax=axes[i],
        )

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig
