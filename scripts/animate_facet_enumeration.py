"""Animate a schematic 2D MILP facet search.

The toy grid points stand in for feasible MILP solutions. They are enumerated
only to draw the schematic; VertexRepEngine instead finds an extreme point by
solving a MILP for each facet direction.

Usage: python -m scripts.animate_facet_enumeration [--output FILE.gif]
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull


@dataclass
class Frame:
    round_num: int
    title: str
    caption: str
    hull_points: np.ndarray
    discovered: set[tuple[int, int]]
    facet: np.ndarray | None = None
    normal: np.ndarray | None = None
    result: tuple[int, int] | None = None


def search_frames() -> tuple[list[Frame], np.ndarray]:
    """Follow the engine's facet loop, using a finite toy set as the oracle."""
    # These integer points represent the indices admitted by a toy MILP.
    polygon = np.array([
        [-0.6, -0.4], [3.4, -0.5], [4.7, 0.8],
        [4.3, 3.6], [2.4, 4.8], [-0.3, 3.9],
    ])
    equations = ConvexHull(polygon).equations
    lattice = np.array(list(itertools.product(range(-1, 6), repeat=2)))
    candidates = lattice[np.all(lattice @ equations[:, :2].T + equations[:, 2] <= 0, axis=1)]

    # The engine begins with corners of the grid cell around an LP seed.
    discovered = set(itertools.product((0, 1), repeat=2))
    frames = [Frame(
        0, "Seed grid cell", "Start with the four grid-cell corners around an LP-feasible point.",
        np.array(sorted(discovered)), discovered.copy(),
    )]

    for round_num in itertools.count(1):
        points = np.array(sorted(discovered))
        hull = ConvexHull(points)
        new_this_round = 0
        for facet_num, (equation, simplex) in enumerate(
            zip(hull.equations, hull.simplices), start=1
        ):
            normal, offset = equation[:2], equation[2]
            facet = points[simplex]
            frames.append(Frame(
                round_num, f"Facet {facet_num} of {len(hull.simplices)}",
                "Query the MILP: maximize the outward normal · grid index.",
                points, discovered.copy(), facet, normal,
            ))

            # This argmax illustrates the MILP query, but does not run a solver.
            result = tuple(map(int, candidates[np.argmax(candidates @ normal)]))
            beyond = normal @ result + offset > 1e-6
            is_new = beyond and result not in discovered
            if is_new:
                discovered.add(result)
                new_this_round += 1
            caption = (
                "The extreme grid point is beyond this facet: add it."
                if is_new else "No new grid point lies beyond this facet."
            )
            frames.append(Frame(
                round_num, f"Facet {facet_num} of {len(hull.simplices)}",
                caption, points, discovered.copy(), facet, normal, result,
            ))

        if new_this_round == 0:
            break
        frames.append(Frame(
            round_num, "Rebuild the hull",
            f"Found {new_this_round} new grid points; form their hull and check its facets.",
            np.array(sorted(discovered)), discovered.copy(),
        ))

    frames.append(Frame(
        round_num, "Search complete",
        "Every facet has been checked without finding a new grid point.",
        np.array(sorted(discovered)), discovered.copy(),
    ))
    return frames, candidates


def animate(output: Path, duration_ms: int = 1000) -> None:
    frames, candidates = search_frames()
    fig, ax = plt.subplots(figsize=(9.4, 6.4), dpi=120)
    fig.subplots_adjust(left=0.08, right=0.72, bottom=0.20, top=0.84)
    fig.suptitle("MILP facet search (schematic)", fontsize=17)
    caption = fig.text(0.5, 0.115, "", ha="center", fontsize=11)
    fig.text(
        0.5, 0.055,
        "Faint points are shown for illustration; the engine finds them by MILP queries.",
        ha="center", fontsize=9, color="0.4",
    )

    images: list[Image.Image] = []
    try:
        for frame in frames:
            ax.clear()
            ax.set_xlim(-1, 5.5)
            ax.set_ylim(-1, 5.5)
            ax.set_aspect("equal")
            ax.set_xticks(range(0, 6))
            ax.set_yticks(range(0, 6))
            ax.grid(color="0.88", linewidth=0.8)
            ax.set_xlabel("Grid index $k_1$")
            ax.set_ylabel("Grid index $k_2$")
            ax.set_title(frame.title if frame.round_num == 0 else
                         f"Round {frame.round_num} · {frame.title}", fontsize=13)

            ax.scatter(candidates[:, 0], candidates[:, 1], s=25, color="0.78",
                       label="MILP-feasible points", zorder=2)
            hull = ConvexHull(frame.hull_points)
            boundary = np.append(hull.vertices, hull.vertices[0])
            outline = frame.hull_points[boundary]
            ax.fill(outline[:, 0], outline[:, 1], color="tab:blue", alpha=0.12)
            ax.plot(outline[:, 0], outline[:, 1], color="tab:blue", lw=2,
                    label="Current hull", zorder=3)
            known = np.array(sorted(frame.discovered))
            ax.scatter(known[:, 0], known[:, 1], s=65, color="tab:blue",
                       edgecolor="white", linewidth=0.8,
                       label="Discovered points", zorder=4)

            if frame.round_num == 0:
                ax.scatter([0.4], [0.5], s=85, marker="x", color="purple",
                           linewidth=2, label="LP seed", zorder=5)
            if frame.facet is not None:
                ax.plot(frame.facet[:, 0], frame.facet[:, 1], color="darkorange",
                        lw=4, solid_capstyle="round", label="Facet being checked", zorder=5)
                midpoint = frame.facet.mean(axis=0)
                ax.annotate("", xy=midpoint + 0.7 * frame.normal, xytext=midpoint,
                            arrowprops={"arrowstyle": "->", "color": "darkorange", "lw": 2})
            if frame.result is not None:
                ax.scatter(*frame.result, s=115, color="crimson", marker="*",
                           edgecolor="white", linewidth=0.5,
                           label="MILP extreme point", zorder=6)

            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1),
                      fontsize=9, framealpha=0.95)
            caption.set_text(frame.caption)
            fig.canvas.draw()
            # The buffer can be larger than get_width_height() on Retina displays.
            image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
            images.append(image.quantize(colors=128))
    finally:
        plt.close(fig)

    output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output, save_all=True, append_images=images[1:],
        duration=[1200] + [duration_ms] * (len(images) - 2) + [2500],
        loop=0, optimize=True, disposal=2,
    )
    print(f"Saved {len(images)} schematic frames to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/facet_enumeration.gif"),
                        help="Output GIF (default: results/facet_enumeration.gif)")
    parser.add_argument("--duration-ms", type=int, default=1000,
                        help="Milliseconds per intermediate frame")
    args = parser.parse_args()
    if args.duration_ms <= 0:
        parser.error("--duration-ms must be positive")
    animate(args.output, args.duration_ms)


if __name__ == "__main__":
    main()
