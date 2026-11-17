"""Regenerate a policy trajectory GIF and static plots from saved data.

Usage: python -m scripts.animate_policy_trajectory RUN_DIR [--output FILE.gif]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.common.animate_policy_trajectory import animate
from experiments.common.io import load_config_dict
from experiments.common.view import plot_single_run_from_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path,
                        help="Output GIF (default: RUN_DIR/policy_trajectory_animation.gif)")
    parser.add_argument("--duration-ms", type=int, default=1800,
                        help="Milliseconds per target or onward frame")
    args = parser.parse_args()
    animate(args.run_dir, args.output or args.run_dir / "policy_trajectory_animation.gif", args.duration_ms)
    config = load_config_dict(args.run_dir) if (args.run_dir / "config.json").exists() else {}
    dpi = config.get("output", {}).get("image_dpi", 150)
    title_prefix = "MG " if "num_agents" in config.get("instance", {}) else ""
    figures = plot_single_run_from_dir(args.run_dir, title_prefix=title_prefix)
    try:
        for fmt in ("png", "pdf"):
            path = args.run_dir / f"policy_trajectory.{fmt}"
            figures["policy_trajectory"].savefig(path, dpi=dpi)
            print(f"Saved {path}")
    finally:
        for figure in figures.values():
            plt.close(figure)


if __name__ == "__main__":
    main()
