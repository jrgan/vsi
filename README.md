# VSI — Value-Set Iteration

This repository implements a **Value-Set Iteration (VSI)** framework for computing achievable value sets in multi-objective Markov decision processes (MDPs) and Markov games, with grid-based representations and Gurobi as the LP/MILP solver.

## Associated paper

### Subprocess-Constrained MDP (SCMDP) paper

This repository accompanies the SCMDP paper:

Jiarui Gan and Debmalya Mandal. *Subprocess-Constrained Markov Decision
Processes* (2026).
[arXiv:2611.21471](https://arxiv.org/abs/2611.21471).
(To appear in NeurIPS 2026.)

```bibtex
@misc{gan2026subprocess,
  title         = {Subprocess-Constrained Markov Decision Processes},
  author        = {Jiarui Gan and Debmalya Mandal},
  year          = {2026},
  eprint        = {2611.21471},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  doi           = {10.48550/arXiv.2611.21471},
  url           = {https://arxiv.org/abs/2611.21471}
}
```

### Related Markov games paper

The value-set iteration framework and the Markov games domain are described in:

Jiarui Gan and Rupak Majumdar. *Value-Set Iteration: Computing Optimal Correlated
Equilibria in Infinite-Horizon Multi-Player Stochastic Games* (2025).
[arXiv:2506.07186](https://arxiv.org/abs/2506.07186).

```bibtex
@misc{gan2025valuesetiteration,
  title         = {Value-Set Iteration: Computing Optimal Correlated Equilibria in Infinite-Horizon Multi-Player Stochastic Games},
  author        = {Jiarui Gan and Rupak Majumdar},
  year          = {2025},
  eprint        = {2506.07186},
  archivePrefix = {arXiv},
  primaryClass  = {cs.GT},
  doi           = {10.48550/arXiv.2506.07186},
  url           = {https://arxiv.org/abs/2506.07186}
}
```

If you use this implementation in research, please cite the SCMDP paper. For work
using the Markov games domain, please also cite the Markov games paper.

## Overview

VSI generalizes classical value iteration to multi-objective and constrained settings by propagating entire *sets* of achievable value vectors rather than scalar values. Value sets are stored in a **vertex representation** on a grid over value space, and one of three interchangeable engines computes each iteration step:

| Engine | Description |
|---|---|
| `VertexRepEngine` | Integer hull of each achievable value set via MILP facet separation (high-fidelity default) |
| `SingletonEngine` | A single feasible point per state (LP/QP) |
| `GridEnumEngine` | Exhaustive enumeration of grid points (baseline for comparison) |

All three use the same domain LP factory and runner interface. The grid-based
engines take a `Grid`; `SingletonEngine` needs only the factory. See
[docs/guides/vsi-design.md](docs/guides/vsi-design.md) for the architecture and
animations of value-set iteration, policy rollout, and MILP facet search.

`SingletonEngine` is an experimental implementation extension, not part of the
associated paper.

The framework currently supports two problem domains:

- **SCMDP** — Subprocess-constrained Markov decision processes  
- **Markov games** — Multi-player general-sum Markov games (stochastic games)

## Requirements

- Python >= 3.10
- A valid [Gurobi license](https://www.gurobi.com/downloads/) (free academic licenses available)

## Installation

```bash
git clone https://github.com/jrgan/VSI.git
cd VSI
python3 --version  # must be 3.10 or newer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

An existing virtual environment keeps the Python version used to create it.
If it uses Python 3.9 or older, recreate it with a supported interpreter.

The editable installation makes `vsi`, `scmdp`, and `markovgame` importable from
any directory and uses the dependencies in `requirements.txt`. Run demos and
examples from the repository root.

## Running the Demos

A single CLI dispatcher exposes all curated demo scenarios:

```bash
python3 main.py --list                          # list available scenarios
python3 main.py scmdp single_run                # run one SCMDP scenario
python3 main.py markovgame single_run           # run one Markov game scenario
python3 main.py scmdp --all                     # run every SCMDP scenario
python3 main.py scmdp_singleton --all           # run every SCMDP singleton scenario

# Optional lightweight overrides for sweeps/debugging
python3 main.py scmdp single_run --gamma 0.9 --eps 0.1 --delta 0.1
python3 main.py markovgame_singleton single_run --max-iters 200 --seed 7
```

The first positional argument selects the domain (`scmdp`, `scmdp_singleton`,
`markovgame`, or `markovgame_singleton`) and the second selects the scenario;
with no domain, `main.py` prints the listing and exits. The `*_singleton`
domains run the cheaper singleton inner loop and scale to larger instances.
The dispatcher covers demos only — the worked examples under `examples/` are
run directly (see below).

Supported override flags: `--gamma`, `--eps`, `--delta`, `--seed`,
`--max-iters`, `--output-dir` (interpreted as a base directory; each
scenario still writes to a `<base>/<scenario>` subdirectory).

Available scenarios:

| Domain                 | Scenarios                                                                            |
| ---------------------- | ------------------------------------------------------------------------------------ |
| `scmdp`                | `single_run`, `bound_sweep`, `scaling_states`, `scaling_actions`                     |
| `scmdp_singleton`      | `single_run`, `scaling_states`, `scaling_actions`, `scaling_reward_dim`              |
| `markovgame`           | `single_run`, `scaling_states`, `scaling_actions`                                    |
| `markovgame_singleton` | `single_run`, `scaling_states`, `scaling_actions`, `scaling_num_agents`              |

Each scenario writes its figures and plot-input artifacts under
`results/demos/<domain>/<scenario>/<run_label>/`. The saved
artifacts (`final_value_sets.npz`, `history/`, `trajectory.npz`, and
`policy_expansions.npz`) are
sufficient to re-render figures without re-running VSI via
`experiments.{scmdp,markovgame}.views.plot_single_run_from_dir`.
The full `scmdp` and `markovgame` `single_run` demos also generate
`iteration_animation.gif` and `policy_trajectory_animation.gif` alongside
their matching PNG/PDF figures. `policy_expansions.npz` stores the exact
target and onward values used in both the static policy plot and animation.
For singleton trajectories, use `experiments.common.view.plot_singleton_from_dir`;
see the [artifact guide](docs/guides/experiments.md#4-views-and-artifacts).

## Parameters and convergence

The accuracy and termination knobs:

- `eps`, `delta` — target accuracy. The domain grid helper turns them into the
  grid spacing `(1 - gamma) * min(eps, delta/2)`; smaller values give a finer grid
  and a larger per-state MILP.
- `convergence_tol` (`IterationRunner`, default `1e-9`) — stop when the distance
  between successive iterates is at or below it. The integer-indexed grid makes this
  distance reach exactly `0` at a fixed point, so a tiny tolerance is safe.
- `max_iters` — hard cap; iteration stops with status `MAX_ITER_REACHED` if the
  fixed point is not reached first.

`eps`, `delta`, and grid spacing must be finite and positive; `max_iters` and
`max_workers` must be positive integers.

Per-state progress bars are enabled by default. When using an engine directly,
`engine.set_param(show_progress=False)` disables them without changing ordinary
logging or solver results. State-level parallelism is controlled separately by
`engine.set_param(vsi_operator_parallel=True)`; see the
[architecture guide](docs/guides/vsi-design.md) for worker environment ownership.

Solver settings belong to the domain LP factory; see the
[configuration example](docs/guides/vsi-design.md#solver-settings).
Gurobi starts quietly when first needed; no initialization call is required.

After a run, `IterationRunner.status` is one of `CONVERGED`, `MAX_ITER_REACHED`,
`LOOP_DETECTED`, or `RUNNING` / `INITIATED`. Only extract policies
(`PolicyRunner`) from **converged** value sets — rolling out from under-converged
or capped sets can raise `InfeasiblePolicyError`. Note that `SingletonEngine`
carries no set-convergence guarantee (it may not converge, or may converge to an
empty set).

For the algorithmic theory — the grid construction, the `(eps, delta)` guarantee,
and the contraction/convergence analysis — see the [associated paper](#associated-paper).

## Documentation

- [VSI design and architecture](docs/guides/vsi-design.md) — how `vsi/`, the engines, and the domain packages fit together.
- [Adding a domain](docs/guides/adding-a-domain.md) — the three pieces a new problem domain supplies to plug into the engines.
- [Experiments and demos](docs/guides/experiments.md) — how `experiments/`, the runners, the views, and the demos fit together.
- [Reproducing results](docs/guides/reproducibility.md) — tested dependency versions and paper-figure commands.

## Worked Examples

Self-contained, hand-built toy instances live under `examples/`:

```bash
python3 -m examples.scmdp.two_state         # 2-state SCMDP example
python3 -m examples.scmdp.nonstationary_advantage
python3 -m examples.markovgame.two_state    # 2-state, 2-agent MG example
python3 -m examples.markovgame.prisoners_dilemma
```

To run a whole domain (or all four examples), use the batch runner:

```bash
python3 -m scripts.run_examples scmdp       # or markovgame, or all
python3 -m scripts.run_examples --list       # show individual example names
```

The runner clears only the selected examples' generated files before each run;
it leaves unrelated files and the other domain's results untouched.

Examples are intended to be run as modules (`python3 -m ...`), not as
direct script paths.

The examples run `VertexRepEngine` on hand-built instances with
`eps = delta = 0.1`. Each saves iteration snapshots, a sampled policy
trajectory, both GIFs, and their underlying data under
`results/examples/<domain>/<example_name>/`. State panels share x/y limits.

For questions or bug reports, please open an issue and
include the command, Python/Gurobi versions, and full traceback or run config.

## Repository Structure

```
vsi/                  Core VSI engine (domain-agnostic)
scmdp/                SCMDP environment and value-set LP factory
markovgame/           Markov game environment and value-set LP factory
experiments/
  common/             Shared config, I/O, results, and visualisation
  scmdp/              SCMDP experiment runners and view classes
  markovgame/         Markov game experiment runners and view classes
demos/                CLI-dispatched demo scenarios (full + singleton domains)
examples/             Hand-built toy instances and walkthroughs
docs/guides/          Design and architecture documentation
scripts/              Example batch runner and animation regeneration
main.py               CLI dispatcher for demos
```

## License

MIT — see [LICENSE](LICENSE).
