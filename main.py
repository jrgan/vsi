"""
CLI dispatcher for VSI demos.

Each domain exposes a registry of named scenarios. Use this script to list
available scenarios and run them by name. The full value-set domains
(``scmdp``, ``markovgame``) run the vertex-representation engine; the
singleton domains (``scmdp_singleton``, ``markovgame_singleton``) run the
cheaper singleton inner loop and scale to larger instances.

Usage:
    python main.py --list
    python main.py scmdp single_run
    python main.py scmdp --all
    python main.py scmdp_singleton scaling_states
    python main.py markovgame single_run
    python main.py markovgame_singleton scaling_num_agents
    python main.py scmdp single_run --gamma 0.9 --eps 0.1 --delta 0.1
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from demos import scmdp as scmdp_demos
from demos import scmdp_singleton as scmdp_singleton_demos
from demos import markovgame as mg_demos
from demos import markovgame_singleton as mg_singleton_demos


DOMAINS = {
    "scmdp":                scmdp_demos.SCENARIOS,
    "scmdp_singleton":      scmdp_singleton_demos.SCENARIOS,
    "markovgame":           mg_demos.SCENARIOS,
    "markovgame_singleton": mg_singleton_demos.SCENARIOS,
}


def _format_listing() -> str:
    lines = ["Available demos:"]
    for domain, scenarios in DOMAINS.items():
        lines.append(f"  {domain}:")
        for name in scenarios:
            lines.append(f"    - {name}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a VSI demo scenario.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_format_listing(),
    )
    parser.add_argument(
        "domain", nargs="?", choices=DOMAINS.keys(),
        help="Problem domain to run a scenario in.",
    )
    parser.add_argument(
        "scenario", nargs="?",
        help="Scenario name within the chosen domain.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all scenarios for the chosen domain.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available domains and scenarios, then exit.",
    )
    parser.add_argument(
        "--gamma", type=float,
        help="Override discount factor for the selected scenario(s).",
    )
    parser.add_argument(
        "--eps", type=float,
        help="Override approximation parameter eps for the selected scenario(s).",
    )
    parser.add_argument(
        "--delta", type=float,
        help="Override approximation parameter delta for the selected scenario(s).",
    )
    parser.add_argument(
        "--seed", type=int,
        help="Override instance random seed for the selected scenario(s).",
    )
    parser.add_argument(
        "--max-iters", type=int,
        help="Override max iterations (algorithm-level and singleton-view cap).",
    )
    parser.add_argument(
        "--output-dir", type=str,
        help="Override base output directory for selected scenario(s).",
    )
    args = parser.parse_args()

    if args.list or args.domain is None:
        print(_format_listing())
        return 0

    overrides: dict[str, Any] = {
        "gamma": args.gamma,
        "eps": args.eps,
        "delta": args.delta,
        "seed": args.seed,
        "max_iters": args.max_iters,
        "output_dir": args.output_dir,
    }

    scenarios = DOMAINS[args.domain]

    if args.all:
        to_run = list(scenarios.items())
    else:
        if args.scenario is None:
            parser.error(
                f"scenario is required for domain '{args.domain}' "
                f"(or pass --all). Available: {', '.join(scenarios)}"
            )
        if args.scenario not in scenarios:
            parser.error(
                f"unknown scenario '{args.scenario}' for domain "
                f"'{args.domain}'. Available: {', '.join(scenarios)}"
            )
        to_run = [(args.scenario, scenarios[args.scenario])]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    log = logging.getLogger(__name__)
    for name, fn in to_run:
        log.info("=== %s: %s ===", args.domain, name)
        fn(overrides=overrides)
    return 0


if __name__ == "__main__":
    sys.exit(main())
