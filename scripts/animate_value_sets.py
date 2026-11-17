"""Regenerate an iteration GIF and static plots from saved data.

Usage: python -m scripts.animate_value_sets RUN_DIR [--output FILE.gif]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.common.animate_value_sets import animate
from experiments.common.io import load_config_dict
from experiments.common.view import plot_single_run_from_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run directory containing history/iter_*.npz")
    parser.add_argument("--output", type=Path, help="Output GIF (default: RUN_DIR/iteration_animation.gif)")
    parser.add_argument("--duration-ms", type=int, default=330, help="Milliseconds per intermediate frame")
    args = parser.parse_args()
    animate(args.run_dir, args.output or args.run_dir / "iteration_animation.gif", args.duration_ms)
    config = load_config_dict(args.run_dir) if (args.run_dir / "config.json").exists() else {}
    dpi = config.get("output", {}).get("image_dpi", 150)
    figures = plot_single_run_from_dir(args.run_dir)
    try:
        for fmt in ("png", "pdf"):
            path = args.run_dir / f"iteration_snapshots.{fmt}"
            figures["iteration_snapshots"].savefig(path, dpi=dpi)
            print(f"Saved {path}")
    finally:
        for figure in figures.values():
            plt.close(figure)


if __name__ == "__main__":
    main()
