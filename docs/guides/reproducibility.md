# Reproducing results

The primary reference for this repository is the
[SCMDP paper](https://arxiv.org/abs/2611.21471). The Markov-games domain relates
to the separate [Markov-games paper](https://arxiv.org/abs/2506.07186).

## Environment

The project has been run with CPython 3.14.3 on macOS 15.4.1 (x86_64) and
Gurobi/gurobipy 13.0.1. Install the project as described in the
[README](../../README.md#installation); its dependencies are listed in
[requirements.txt](../../requirements.txt). A working Gurobi license is needed
to run the experiments.

Record the package and solver versions used for your own runs when comparing
results across environments.

## SCMDP paper figures

Run the following commands from the repository root, with the environment active
and Gurobi licensed:

| Paper figure | Command | Output figure |
| --- | --- | --- |
| Figure 2: value sets across iterations | `python main.py scmdp single_run` | `iteration_snapshots.png` / `iteration_snapshots.pdf` / `iteration_animation.gif` |
| Figure 3: policy target-value trajectory | same `single_run` command | `policy_trajectory.png` / `policy_trajectory.pdf` / `policy_trajectory_animation.gif` |
| Figure 4: cost-bound sweep | `python main.py scmdp bound_sweep` | `bound_sweep.png` / `bound_sweep.pdf` |

Figure numbers refer to the current manuscript; plot styling may differ.

Paper-scale runs can take considerably longer than the worked examples. Check
that they converge before using the results.

The base instance has 4 states, 5 actions, 2 value dimensions, discount factor
`gamma = 0.8`, reward entries sampled uniformly from `[-1, 1]`, and instance seed
`42`. The single run uses cost bound `2.0`, `eps = delta = 0.1`, and state-level
parallelism. Its policy rollout maximizes the reward coordinate, starts in state
0, and records 50 steps with sampling seed `123`.

The bound sweep uses the same instance-generation settings and accuracy, with
cost bounds `5, 3, 1, -1, -3`. These settings
are defined in [demos/scmdp.py](../../demos/scmdp.py).

Outputs are written under `results/demos/scmdp/<scenario>/<run_label>/`, including
PNG/PDF figures, config, generated instance, summary, and available value-set and
trajectory artifacts. The `single_run` also writes both GIFs and
`policy_expansions.npz`; history is saved for the single run, but not the bound
sweep. See [experiments.md](experiments.md#4-views-and-artifacts) for the
animation inputs and commands to regenerate them without rerunning VSI.

Rerunning the same run label replaces its generated data and single-run figures;
copy results elsewhere first if you want to keep an earlier run. Policy extraction
is skipped if value-set iteration does not converge.

Preserve the config, generated instance, and Git commit with results. Hardware,
solver versions, and alternative optimal LP solutions can affect runtime and
sampled policy trajectories. To record the commit:

```bash
git rev-parse HEAD
```

The repository includes selected PNG, PDF, and GIF plots under `results/`.
Generated raw artifacts are ignored by Git; rerun the experiments to obtain
their configs, histories, and value-set data, or keep those files in a separate
archive if you need to preserve a particular run.
