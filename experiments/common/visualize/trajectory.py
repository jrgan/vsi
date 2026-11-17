"""
Policy trajectory and singleton iteration trajectory plots.  Problem-agnostic.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.spatial import ConvexHull
from vsi.vertex_rep.polytope import convex_hull_vertices

from experiments.common.visualize._legend import _best_corner_legend
from experiments.common.visualize.style import animation_axes, shared_bounds, style_state_axis
from experiments.common.visualize.convergence import _outline


def plot_policy_trajectory(
    trajectory: list[tuple[int, np.ndarray]],
    value_sets: list[np.ndarray] | None = None,
    num_states: int | None = None,
    title: str = "Policy Trajectory",
    onward_values: np.ndarray | None = None,
    target_values: np.ndarray | None = None,
) -> Figure:
    """
    Plot policy target values on per-state subplots.

    Each target is plotted in the state recorded by ``trajectory``. If targets
    are unavailable (as in older artifacts), use the achieved values instead.
    State panels share axis limits; target dots are red. The first, every
    tenth, and the final iteration are labelled.

    Args:
        trajectory: List of ``(state, v_achieved)`` tuples.
        value_sets: Optional achievable value sets shown as convex hulls.
        num_states: Number of states.  Inferred from *value_sets* if not given.
        title: Figure title.
        onward_values: Optional onward values included in axis limits.
        target_values: Optional policy targets, one per trajectory step.

    Returns:
        The matplotlib Figure.
    """
    if num_states is None:
        if value_sets is not None:
            num_states = len(value_sets)
        else:
            num_states = max(s for s, _ in trajectory) + 1

    num_steps = len(trajectory)
    plotted_values = (target_values if target_values is not None else
                      np.asarray([value for _, value in trajectory]))
    fig, axes = animation_axes(num_states)
    outlines = [_outline(points) for points in value_sets] if value_sets is not None else []
    groups = outlines + [plotted_values]
    if onward_values is not None:
        groups.append(onward_values)
    xlim, ylim = shared_bounds(groups)

    for s in range(num_states):
        ax = axes[s]
        style_state_axis(ax, s, xlim, ylim)
        if outlines and len(outlines[s]):
            points = outlines[s]
            color = plt.cm.tab10.colors[0]
            if len(points) >= 3:
                ax.fill(points[:, 0], points[:, 1], color=color, alpha=0.15)
                closed = np.vstack((points, points[0]))
                ax.plot(closed[:, 0], closed[:, 1], color=color, lw=1.0)
            elif len(points) == 2:
                ax.plot(points[:, 0], points[:, 1], color=color, lw=1.0)

    for step, (state, _) in enumerate(trajectory):
        ax = axes[state]
        target = plotted_values[step]
        ax.scatter(
            target[0], target[1],
            color="red", s=40,
            edgecolors="darkred", linewidths=0.8,
            zorder=4,
        )
        if step % 10 == 0 or step == num_steps - 1:
            start = step == 0
            ax.annotate(
                f"iter {step}",
                (target[0], target[1]),
                xytext=(0, -6 if start else 6), textcoords="offset points",
                fontsize=9, ha="center", va="top" if start else "bottom",
            )
    fig.suptitle(title, fontsize=18)
    return fig


def plot_singleton_trajectory(
    history: list[list[np.ndarray]],
    title: str = "Singleton Iteration Trajectory",
    value_sets: Optional[list[np.ndarray]] = None,
    converged: bool = True,
    loop_detected: bool = False,
) -> Figure:
    """
    Plot singleton trajectories in per-state subplots with shared axis limits.

    Args:
        history: ``history[t][s]`` is a 1-D array of shape ``(d,)`` —
            the singleton value for state *s* at iteration *t*.
        title: Overall figure title.
        value_sets: Optional list of point arrays (one per state) to
            render as background convex hulls.
        converged: Whether the singleton iteration converged to a fixed
            point.  Affects the end-point marker style.
        loop_detected: Whether a loop was detected during singleton
            iteration.  Shown as a blue cross end-point marker.

    Returns:
        The matplotlib Figure.
    """
    n_iters = len(history)
    if n_iters == 0:
        fig, ax = plt.subplots()
        ax.set_title(title + " (no data)")
        return fig

    n_states = len(history[0])
    cols = min(n_states, 4)
    rows = (n_states + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False,
        sharex=True, sharey=True,
    )

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=max(n_iters - 1, 1))

    for s in range(n_states):
        ax = axes[s // cols][s % cols]
        ax.tick_params(labelbottom=True, labelleft=True)

        if value_sets is not None and len(value_sets[s]) >= 3:
            pts = convex_hull_vertices(value_sets[s])
            try:
                hull = ConvexHull(pts)
                hull_idx = np.append(hull.vertices, hull.vertices[0])
                ax.fill(pts[hull_idx, 0], pts[hull_idx, 1], alpha=0.12, color="grey")
                ax.plot(pts[hull_idx, 0], pts[hull_idx, 1], color="grey", linewidth=0.6, alpha=0.5)
            except Exception:
                pass

        traj = np.array([history[t][s] for t in range(n_iters)])
        valid = ~np.isnan(traj).any(axis=1)

        if not valid.any():
            ax.set_title(f"State {s} (infeasible)")
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            ax.grid(True, alpha=0.3)
            continue

        lw = max(0.4, min(1.0, 60 / n_iters))
        dot_size = max(2, min(10, 300 / n_iters))
        line_alpha = max(0.4, min(0.8, 50 / n_iters))

        for t in range(n_iters - 1):
            if valid[t] and valid[t + 1]:
                color = cmap(norm(t))
                ax.plot(traj[t:t + 2, 0], traj[t:t + 2, 1], color=color, linewidth=lw, alpha=line_alpha)

        valid_idx = np.where(valid)[0]
        colors = [cmap(norm(t)) for t in valid_idx]
        ax.scatter(traj[valid, 0], traj[valid, 1], c=colors, s=dot_size, zorder=4, edgecolors="none")

        first, last = valid_idx[0], valid_idx[-1]
        ax.scatter(traj[first, 0], traj[first, 1], marker="o", s=40, facecolors="none",
                   edgecolors="green", linewidths=1.2, zorder=5, label="start")

        if last < n_iters - 1:
            ax.scatter(traj[last, 0], traj[last, 1], marker="o", s=50, facecolors="none",
                       edgecolors="red", linewidths=1.0, linestyle="dotted",
                       zorder=5, label=f"iter {last} (infeasible)")
        elif loop_detected:
            ax.scatter(traj[last, 0], traj[last, 1], marker="x", s=40, color="blue",
                       linewidths=1, zorder=5, label=f"iter {last} (loop)")
        elif not converged:
            ax.scatter(traj[last, 0], traj[last, 1], marker="o", s=50, facecolors="none",
                       edgecolors="red", linewidths=1.2, zorder=5, label=f"iter {last} (not converged)")
        else:
            ax.scatter(traj[last, 0], traj[last, 1], marker="x", s=40, color="red",
                       linewidths=1, zorder=5, label=f"iter {last} (converged)")

        ax.set_title(f"State {s}")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        _best_corner_legend(ax, fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(n_states, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig
