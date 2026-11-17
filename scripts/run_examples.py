"""Run the worked examples, optionally selecting one domain or example.

Usage: python -m scripts.run_examples [all|scmdp|markovgame|DOMAIN.EXAMPLE]
       python -m scripts.run_examples --list
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from experiments.common.io import clear_run_artifacts


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = (
    "scmdp.two_state",
    "scmdp.nonstationary_advantage",
    "markovgame.two_state",
    "markovgame.prisoners_dilemma",
)


def run_examples(selection: str) -> int:
    """Run selected examples, clearing only their generated artifacts first."""
    selected = [name for name in EXAMPLES
                if selection == "all" or name == selection or name.startswith(selection + ".")]
    failures = 0
    for name in selected:
        domain, example = name.split(".")
        clear_run_artifacts(ROOT / "results" / "examples" / domain / example)
        print(f"[run ] {name}", flush=True)
        result = subprocess.run([sys.executable, "-m", f"examples.{name}"], cwd=ROOT)
        if result.returncode:
            print(f"[FAIL] {name} (exit {result.returncode})", flush=True)
            failures += 1
        else:
            print(f"[ok  ] {name}", flush=True)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", nargs="?", default="all",
                        choices=("all", "scmdp", "markovgame", *EXAMPLES),
                        help="Domain or individual example (default: all)")
    parser.add_argument("--list", action="store_true", help="List available examples")
    args = parser.parse_args()
    if args.list:
        print("\n".join(EXAMPLES))
        return 0
    return run_examples(args.selection)


if __name__ == "__main__":
    raise SystemExit(main())
