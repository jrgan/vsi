# Experiments and Demos

This guide explains how the `experiments/` package is organised, how a demo turns
into a figure on disk, and how to add or re-run scenarios. It complements
[vsi-design.md](vsi-design.md) (the engine) and
[adding-a-domain.md](adding-a-domain.md) (the domain plug-ins).

`experiments/` is the harness **around** the engine: it builds problem instances,
drives `IterationRunner` / `PolicyRunner`, saves artifacts, and renders figures.
The engine knows nothing about it.

---

## 1. The layers

```
main.py                     CLI dispatcher — picks (domain, scenario)
  │
demos/                      scenario registries: build a config, run a View
  ├── scmdp.py              SCENARIOS = {single_run, bound_sweep, scaling_*}
  ├── scmdp_singleton.py
  ├── markovgame.py
  └── markovgame_singleton.py
  │
experiments/
  ├── common/               domain-agnostic harness
  │   ├── config.py         BaseConfig / BaseAlgorithmConfig / OutputConfig / PolicyConfig / VertexSetMode
  │   ├── runner.py         BaseRunner (template method) + ENGINE_BY_MODE + SEED_FN
  │   ├── view.py           BaseView (compute vs. persist) + plot_single_run_from_dir
  │   ├── io.py             run_dir + save/load artifacts (npz, json)
  │   ├── results.py        RunResult / PolicyResult / SingletonResult containers
  │   ├── summary.py        summary metrics
  │   └── visualize/        convergence / point_sets / sweep / trajectory plots
  ├── scmdp/                SCMDP-specific config, instance, runner, views, summary
  └── markovgame/           Markov-game-specific counterparts
```

Each layer has a single job:

- **`main.py`** maps a `(domain, scenario)` pair to a callable in `demos/`.
- **`demos/`** builds an `ExperimentConfig` and hands it to a **View**.
- **Views** (`BaseView`) separate *computation* from *persistence*: `run()`
  returns `{figure_name: Figure}`; `run_and_save()` also writes figures and
  artifacts to the run directory. Views call runner functions to do the heavy
  lifting.
- **Runners** (`BaseRunner`) hold the domain-agnostic VSI loop; domain subclasses
  fill in a few hooks.

---

## 2. Config

Configuration is a tree of dataclasses rooted at `ExperimentConfig`. Each domain
(`experiments/scmdp/config.py`, `experiments/markovgame/config.py`) defines:

- `InstanceConfig` — how to generate the problem (states, actions, reward dim,
  gamma, seed, …);
- `AlgorithmConfig` — extends `BaseAlgorithmConfig` (`eps`, `delta`, `max_iters`,
  and `parallel_by_state` — solve the per-state operator across a thread pool;
  see the parallelism note in [vsi-design.md](vsi-design.md)).

The shared pieces in
[experiments/common/config.py](../../experiments/common/config.py) are:

- `PolicyConfig` — policy-extraction settings (scalarisation `objective`,
  `init_state`, `num_steps`, `seed`);
- `OutputConfig` — what to save and where (`output_dir`, `save_history`,
  `image_formats` (PNG and PDF by default), `image_dpi`, …).

`run_label()` turns a config into a stable, human-readable directory name, and
`BaseConfig` provides JSON (de)serialisation so every run records the exact config
that produced it.

### Selecting the engine: `VertexSetMode`

The experiment layer never names an engine class directly. It tags the config with
a `VertexSetMode`, resolved in the runner:

| `VertexSetMode` | Engine | Seed points |
| --- | --- | --- |
| `GRID_VERTICES` | `VertexRepEngine` | box corners |
| `SINGLE_POINT` | `SingletonEngine` | box centre |
| `ALL_GRID_POINTS` | `GridEnumEngine` | full grid mesh |

`ENGINE_BY_MODE` and `SEED_FN` in
[experiments/common/runner.py](../../experiments/common/runner.py) hold these two
mappings.

---

## 3. Runners: the template method

`BaseRunner` implements the domain-agnostic flow (build instance → build grid →
seed → iterate → optionally roll out a policy → summarise → save). Domain
subclasses (`SCMDPExperimentRunner`, `MGExperimentRunner`) supply a small
set of hooks:

- `build_instance()` — construct the problem from `config.instance`;
- `make_grid(instance)` — call the domain grid helper (`scmdp_grid` / `markovgame_grid`);
- `make_engine(...)` — wrap the domain `ValueSetLPFactory` in the engine chosen by
  `VertexSetMode`;
- `make_step_outcome(...)` — build the domain's `StepOutcome` for policy rollout;
- `compute_summary(...)` — domain-specific summary metrics;
- `save_instance(...)` — persist the generated instance.

Everything else — running `IterationRunner`, timing, loop detection, policy
rollout via `PolicyRunner`, writing artifacts — lives once in `BaseRunner`.
`run_parameter_sweep` wraps a single-run function to produce the scaling / sweep
scenarios.

Results come back as typed containers from
[experiments/common/results.py](../../experiments/common/results.py):
`RunResult` (iteration history + final sets + status/timing), `PolicyResult`
(trajectory), and `SingletonResult` (singleton-engine history).

---

## 4. Views and artifacts

A `BaseView` subclass is a thin figure builder. Its `run()` calls the appropriate
runner function(s) and returns named figures; `run_and_save()` persists them in the
run directory in every configured image format, alongside the raw artifacts.
For full SCMDP and Markov game `single_run` views, it also renders both GIFs by
default. Call `run_and_save(generate_animations=False)` in Python to skip GIF
rendering when only static figures are needed.

The saved artifacts are enough to re-render figures **without re-running VSI**:

- `final_value_sets.npz` — the converged result (not written if iteration stops
  without convergence);
- `history/` — per-iteration value sets, including the last iterate when saved;
- `trajectory.npz` — the policy rollout;
- `policy_expansions.npz` — the actual targets, achieved values, and onward
  values from that rollout (the static trajectory and animation use its targets);
- `summary.json` — runtime, iteration count, and termination status;
- `config.json` / instance file — exact inputs.

`plot_single_run_from_dir` (and the domain `views.plot_single_run_from_dir`
wrappers) reload these and rebuild the standard figures. The shared plot builders
live in [experiments/common/visualize/](../../experiments/common/visualize):
`convergence.py`, `point_sets.py`, `sweep.py`, `trajectory.py`.

For full `single_run` results, the figure and animation files are:

| Static PNG/PDF stem | GIF | Contents |
| --- | --- | --- |
| `iteration_snapshots` | `iteration_animation.gif` | Value sets from iteration 0 through convergence, with the current inner boundary highlighted. |
| `policy_trajectory` | `policy_trajectory_animation.gif` | Each policy expansion in two frames: current target, then its onward values. |
| `convergence` | — | Point counts across iterations. |

The renderers live in `experiments/common/animate_value_sets.py` and
`experiments/common/animate_policy_trajectory.py`; the matching `scripts/`
modules are command-line wrappers. They recreate each GIF and its matching
static PNG/PDF plots from saved data. The [design guide](vsi-design.md) embeds
both GIFs from the SCMDP single-run demo and a separate facet-search schematic.
From the repository root, regenerate the run-specific plots without rerunning
value-set iteration:

```bash
RUN_DIR="results/demos/scmdp/single_run/YOUR_RUN_LABEL"
python -m scripts.animate_value_sets "$RUN_DIR"
python -m scripts.animate_policy_trajectory "$RUN_DIR"
```

Replace `YOUR_RUN_LABEL` with the directory name from your result. Use a
`markovgame` run directory for the corresponding Markov game animations. The
same commands work for example directories under `results/examples/`.
The iteration command needs `history/iter_*.npz`. The policy command needs
`trajectory.npz`, `final_value_sets.npz`, and `policy_expansions.npz`. For older
experiment runs without the expansion file, the policy command reconstructs a
feasible decomposition using Gurobi and saves a cache; it may differ from the
one used in the original rollout. New runs save the exact expansion directly.
Rendering GIFs adds time, disk use, and memory use, especially for long runs.

Singleton experiments use the same artifact layout. Their summaries also record
whether a loop was detected and its starting iteration. The singleton
`single_run` demos save history by default, so their trajectories can be replotted
without running the solver:

```python
from experiments.common.view import plot_singleton_from_dir

figures = plot_singleton_from_dir("results/demos/scmdp_singleton/single_run/<run_label>")
figures["singleton_trajectory"].savefig("singleton_trajectory.pdf")
```

This helper works for both domains and restores any saved background hull.
Singleton scaling demos save config, instance, and summary in a separate run
directory for each size, but leave history saving disabled to limit output size.
The summaries' `performance.runtime_seconds` and `performance.num_iterations`
provide the data for `experiments.common.visualize.plot_scaling`.

Value-space figures use common x/y limits across all state panels in the same
figure, with tick labels on every panel. This includes iteration snapshots,
bound sweeps, and policy/singleton trajectories, with or without background value
sets. Runtime and point-count convergence plots keep their own scales because
they show different quantities.

---

## 5. How a demo maps onto the harness

Taking `python3 main.py scmdp single_run`:

1. `main.py` looks up `DOMAINS["scmdp"]` → `demos.scmdp.SCENARIOS["single_run"]`.
2. The scenario function builds an `ExperimentConfig` (paper-aligned base instance
   `S=4, A=5, d=2, gamma=0.8, bound=2.0`; `eps = delta = 0.1`) and applies any CLI
   overrides (`--gamma`, `--eps`, `--delta`, `--seed`, `--max-iters`,
   `--output-dir`).
3. It instantiates the matching `BaseView` subclass (`SingleRunView`,
   `BoundSweepView`, `ScalingView`, or `SingletonView`) and calls `run_and_save()`.
4. The view calls runner functions, which build the SCMDP, its grid, and the
   engine (via `VertexSetMode`), run `IterationRunner`, optionally roll out a
   policy, and write artifacts.
5. Figures and artifacts land under
   `results/demos/<domain>/<scenario>/<run_label>/`.

The scenario catalogue per domain:

| Domain | Scenarios |
| --- | --- |
| `scmdp` | `single_run`, `bound_sweep`, `scaling_states`, `scaling_actions` |
| `scmdp_singleton` | `single_run`, `scaling_states`, `scaling_actions`, `scaling_reward_dim` |
| `markovgame` | `single_run`, `scaling_states`, `scaling_actions` |
| `markovgame_singleton` | `single_run`, `scaling_states`, `scaling_actions`, `scaling_num_agents` |

The `*_singleton` domains swap in `VertexSetMode.SINGLE_POINT` to run the cheaper
LP/QP inner loop (which keeps a single point per state and carries no
set-convergence guarantee; see the `SingletonEngine` notes in
[vsi-design.md](vsi-design.md)). Markov games have no `bound_sweep` (no scalar
constraint bound to sweep).

---

## 6. Adding a scenario

To add a scenario to an existing domain:

1. Add a builder function in `demos/<domain>.py` that returns a configured
   `ExperimentConfig` (reuse `_base_instance` / `_apply_overrides`).
2. Point it at an existing View, or add a new `BaseView` subclass in
   `experiments/<domain>/views.py` if you need a new figure.
3. Register it in that module's `SCENARIOS` dict — it is then dispatchable via
   `main.py`.

To add a scenario for a **new** domain, first follow
[adding-a-domain.md](adding-a-domain.md) to supply the model, factory, and grid,
then add `experiments/<domain>/` (config, instance, runner subclass, views) and a
`demos/<domain>.py` registry mirroring the SCMDP layout.
