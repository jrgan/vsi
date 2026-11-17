"""
Legend placement utility.

Provides ``_best_corner_legend`` which places a matplotlib legend at the
corner of the axes that overlaps plotted data the least.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

# Four-corner legend location codes (matplotlib names)
_CORNER_LOCS = ["upper left", "upper right", "lower left", "lower right"]


def _best_corner_legend(
    ax: plt.Axes,
    handles=None,
    labels=None,
    **kwargs,
) -> None:
    """Place a legend at whichever corner overlaps data the least.

    Tries each of the four corners, measures the overlap between the
    legend bounding box and the data bounding box, and keeps the one
    with the smallest overlap area.  Falls back to ``"upper right"``
    if measurement fails.

    Args:
        ax: The axes to add the legend to.
        handles: Legend handles.  Fetched from *ax* when ``None``.
        labels: Legend labels.  Fetched from *ax* when ``None``.
        **kwargs: Forwarded to ``ax.legend()``.
    """
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    fig = ax.get_figure()
    fig.canvas.draw()  # ensure renderer is available

    renderer = fig.canvas.get_renderer()

    best_loc = "upper right"
    best_overlap = float("inf")

    # Data bbox in display coords
    data_bb = ax.get_window_extent(renderer)

    for loc in _CORNER_LOCS:
        leg = ax.legend(handles, labels, loc=loc, **kwargs)
        fig.canvas.draw()
        leg_bb = leg.get_window_extent(renderer)

        # Geometric overlap area
        x0 = max(leg_bb.x0, data_bb.x0)
        y0 = max(leg_bb.y0, data_bb.y0)
        x1 = min(leg_bb.x1, data_bb.x1)
        y1 = min(leg_bb.y1, data_bb.y1)
        overlap = max(0, x1 - x0) * max(0, y1 - y0)

        # Penalise overlap with actual plotted points / lines
        inv = ax.transData.inverted()
        leg_data = inv.transform([[leg_bb.x0, leg_bb.y0],
                                  [leg_bb.x1, leg_bb.y1]])
        lx0, ly0 = leg_data[0]
        lx1, ly1 = leg_data[1]
        n_inside = 0
        for line in ax.lines:
            xd, yd = line.get_xdata(), line.get_ydata()
            n_inside += int(np.sum(
                (xd >= min(lx0, lx1)) & (xd <= max(lx0, lx1)) &
                (yd >= min(ly0, ly1)) & (yd <= max(ly0, ly1))
            ))
        for coll in ax.collections:
            offsets = coll.get_offsets()
            if len(offsets):
                n_inside += int(np.sum(
                    (offsets[:, 0] >= min(lx0, lx1)) & (offsets[:, 0] <= max(lx0, lx1)) &
                    (offsets[:, 1] >= min(ly0, ly1)) & (offsets[:, 1] <= max(ly0, ly1))
                ))
        # Combine: geometric overlap + data-point penalty
        score = overlap + n_inside * 50

        if score < best_overlap:
            best_overlap = score
            best_loc = loc

    ax.legend(handles, labels, loc=best_loc, **kwargs)

