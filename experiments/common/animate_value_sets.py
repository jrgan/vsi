"""Render saved 2D value-set iterations as a GIF."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull, QhullError

from experiments.common.io import list_history_iterations, load_value_sets_iteration


def _outline(points: np.ndarray) -> np.ndarray:
    """Return ordered hull vertices, or the available points if degenerate."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        return np.empty((0, 2))
    points = points[:, :2]
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        return points
    try:
        return points[ConvexHull(points).vertices]
    except QhullError:
        return points


def animate(run_dir: Path, output: Path, duration_ms: int = 330) -> None:
    iterations = list_history_iterations(run_dir)
    if not iterations or iterations[0] != 0:
        raise ValueError(f"Expected a saved history starting at iteration 0: {run_dir}")
    if iterations != list(range(iterations[-1] + 1)):
        raise ValueError(f"History contains missing iterations: {run_dir}")

    history = [load_value_sets_iteration(i, run_dir) for i in iterations]
    num_states = len(history[0])
    if num_states == 0 or any(len(snapshot) != num_states for snapshot in history):
        raise ValueError("History must contain the same nonzero number of states throughout")
    outlines = [[_outline(points) for points in snapshot] for snapshot in history]
    all_points = [points for snapshot in outlines for points in snapshot if len(points)]
    if not all_points:
        raise ValueError("No 2D value sets found in the saved history")

    bounds = np.concatenate(all_points)
    low = bounds.min(axis=0)
    high = bounds.max(axis=0)
    padding = np.maximum((high - low) * 0.055, 0.05)
    xlim = (low[0] - padding[0], high[0] + padding[0])
    ylim = (low[1] - padding[1], high[1] + padding[1])

    fig, axes = plt.subplots(1, num_states, figsize=(4 * num_states, 4.6), dpi=150,
                             squeeze=False, sharex=True, sharey=True)
    # Layout is set once. Recomputing it for each frame moves the panels as
    # tick labels and the iteration title change.
    fig.subplots_adjust(left=0.045, right=0.995, bottom=0.19, top=0.81, wspace=0.30)
    title = fig.suptitle("", fontsize=18)
    cmap = plt.cm.Blues
    norm = plt.Normalize(vmin=-max(len(iterations) - 1, 1),
                         vmax=2 * max(len(iterations) - 1, 1))
    outer_fill_alpha = 0.15
    inner_extra_alpha = 0.4  # Additional opacity at the centre by the final frame.
    inner_layer_alpha = 1 - (1 - inner_extra_alpha) ** (1 / max(len(iterations) - 1, 1))
    frames: list[Image.Image] = []
    try:
        for frame_index, iteration in enumerate(iterations):
            for state, ax in enumerate(axes.flat[:num_states]):
                ax.clear()
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
                ax.set_aspect("equal", adjustable="box")
                ax.set_anchor("C")
                ax.set_title(f"State {state}", fontsize=14.4)
                ax.set_xlabel("Dimension 1", fontsize=12)
                ax.set_ylabel("Dimension 2", fontsize=12)
                ax.grid(alpha=0.2)
                ax.tick_params(labelbottom=True, labelleft=True, labelsize=12)

                for past in range(frame_index + 1):
                    points = outlines[past][state]
                    if not len(points):
                        continue
                    current = past == frame_index
                    color=cmap(min(norm(past) + 0.15, 1.0))
                    boundary_color = "blue" if current else color
                    if len(points) >= 3:
                        ax.fill(points[:, 0], points[:, 1], color=color,
                                alpha=outer_fill_alpha if past == 0 else inner_layer_alpha,
                                zorder=2 + past)
                        closed = np.vstack((points, points[0]))
                        ax.plot(closed[:, 0], closed[:, 1], color=boundary_color,
                                lw=1.3 if current else 0.65,
                                zorder=3 + past)
                    elif len(points) == 2:
                        ax.plot(points[:, 0], points[:, 1], color=boundary_color,
                                lw=1.3 if current else 0.65, zorder=3 + past)
                    else:
                        ax.scatter(points[:, 0], points[:, 1], color=boundary_color,
                                   s=15 if current else 5, zorder=3 + past)

            label = "last saved iterate" if frame_index == len(iterations) - 1 else "building value sets"
            title.set_text(f"Value-set iteration  {iteration:0{len(str(iterations[-1]))}d}"
                           f"/{iterations[-1]}  •  {label}")
            fig.canvas.draw()
            frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
            frames.append(frame.quantize(colors=128))
    finally:
        plt.close(fig)

    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:],
                   duration=[975] + [duration_ms] * (len(frames) - 2) + [2700],
                   loop=0, optimize=True, disposal=2)
    print(f"Saved {len(frames)} frames to {output}")
