"""Typography and fixed panel geometry shared by experiment figures."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def style_figure(fig, *, fixed_layout: bool = False) -> None:
    """Use the animation's type scale on every experiment figure."""
    for title in fig.texts:
        title.set_fontsize(18)
    for ax in fig.axes:
        ax.title.set_fontsize(14.4)
        ax.xaxis.label.set_fontsize(12)
        ax.yaxis.label.set_fontsize(12)
        ax.tick_params(labelsize=12)
        legend = ax.get_legend()
        if legend is not None:
            for label in legend.get_texts():
                label.set_fontsize(10)
    if fixed_layout:
        fig.subplots_adjust(left=0.045, right=0.995, bottom=0.19,
                            top=0.81, wspace=0.30)
    else:
        fig.tight_layout()


def animation_axes(num_states: int):
    """Create the same state-panel grid used by both animations."""
    fig, axes = plt.subplots(1, num_states, figsize=(4 * num_states, 4.6),
                             squeeze=False, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.045, right=0.995, bottom=0.19,
                        top=0.81, wspace=0.30)
    return fig, axes.flat


def shared_bounds(groups):
    """Fixed padded limits across all states and all plotted iterations."""
    points = [np.asarray(p)[:, :2] for p in groups if len(p)]
    bounds = np.concatenate(points)
    low, high = bounds.min(axis=0), bounds.max(axis=0)
    padding = np.maximum((high - low) * 0.055, 0.05)
    return (low[0] - padding[0], high[0] + padding[0]), (low[1] - padding[1], high[1] + padding[1])


def style_state_axis(ax, state: int, xlim, ylim) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("C")
    ax.set_title(f"State {state}", fontsize=14.4)
    ax.set_xlabel("Dimension 1", fontsize=12)
    ax.set_ylabel("Dimension 2", fontsize=12)
    ax.grid(alpha=0.2)
    ax.tick_params(labelbottom=True, labelleft=True, labelsize=12)
