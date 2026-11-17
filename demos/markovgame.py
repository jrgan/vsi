"""
Markov game demo scenarios (full value-set iteration).

Each scenario builds a config and runs the corresponding view class from
``experiments.markovgame.views``:

- ``single_run``     End-to-end VSI value-set iteration + policy rollout.
- ``scaling_states``  Runtime / iteration scaling with the state count.
- ``scaling_actions`` Runtime / iteration scaling with the actions per agent.

Singleton-iteration counterparts (and the number-of-agents scaling sweep)
live in ``demos.markovgame_singleton``.

Non-scaling scenarios use ``eps = delta = 0.2``; scaling scenarios use the
coarser ``eps = delta = 0.5`` to keep sweep runtimes manageable. All
scenarios use ``gamma = 0.8``.

Scenarios are registered in ``SCENARIOS`` for CLI dispatch via ``main.py``.
"""
from __future__ import annotations

from typing import Any

from experiments.common.config import OutputConfig, PolicyConfig
from experiments.markovgame.config import ExperimentConfig, AlgorithmConfig, InstanceConfig
from experiments.markovgame.views import (
    SingleRunView, ScalingView,
)

_OUTPUT_ROOT = "results/demos/markovgame"


def _output_dir(subdir: str, overrides: dict[str, Any] | None) -> str:
    base = overrides.get("output_dir") if overrides else None
    return f"{base}/{subdir}" if base else f"{_OUTPUT_ROOT}/{subdir}"


def _apply_overrides(config: ExperimentConfig, overrides: dict[str, Any] | None) -> None:
    if not overrides:
        return

    if overrides.get("gamma") is not None:
        config.instance.gamma = float(overrides["gamma"])
    if overrides.get("eps") is not None:
        config.algorithm.eps = float(overrides["eps"])
    if overrides.get("delta") is not None:
        config.algorithm.delta = float(overrides["delta"])
    if overrides.get("seed") is not None:
        config.instance.seed = int(overrides["seed"])
    if overrides.get("max_iters") is not None:
        config.algorithm.max_iters = int(overrides["max_iters"])


def single_run(overrides: dict[str, Any] | None = None):
    """End-to-end value-set iteration with policy rollout (4 states, 2 agents)."""
    config = ExperimentConfig(
        name="mg_single_run",
        instance=InstanceConfig(
            method="random_symmetric",
            num_states=4,
            num_agents=2,
            num_actions_per_agent=5,
            reward_min=-1.0,
            reward_max=1.0,
            gamma=0.8,
            seed=42,
        ),
        algorithm=AlgorithmConfig(eps=0.2, delta=0.2, parallel_by_state=True),
        policy=PolicyConfig(init_state=0, num_steps=50, seed=123),
        output=OutputConfig(
            output_dir=_output_dir("single_run", overrides), save_history=True,
        ),
    )
    _apply_overrides(config, overrides)
    SingleRunView(config).run_and_save()


def _scaling_config(name: str, subdir: str, overrides: dict[str, Any] | None):
    """Scaling base config: 2 agents, coarser eps = delta = 0.5."""
    config = ExperimentConfig(
        name=name,
        instance=InstanceConfig(
            method="random_symmetric",
            num_states=3,
            num_agents=2,
            num_actions_per_agent=3,
            reward_min=-1.0,
            reward_max=1.0,
            gamma=0.8,
            seed=42,
        ),
        algorithm=AlgorithmConfig(eps=0.5, delta=0.5, parallel_by_state=False),
        output=OutputConfig(
            output_dir=_output_dir(subdir, overrides), save_history=False,
        ),
    )
    _apply_overrides(config, overrides)
    return config


def scaling_states(overrides: dict[str, Any] | None = None):
    """Runtime / iteration scaling with the number of states."""
    config = _scaling_config("mg_scaling_states", "scaling_states", overrides)
    ScalingView(config, sizes=[2, 3, 4, 5], axis="states").run_and_save()


def scaling_actions(overrides: dict[str, Any] | None = None):
    """Runtime / iteration scaling with the number of actions per agent."""
    config = _scaling_config("mg_scaling_actions", "scaling_actions", overrides)
    ScalingView(config, sizes=[2, 3, 4, 5], axis="actions").run_and_save()


SCENARIOS = {
    "single_run":      single_run,
    "scaling_states":  scaling_states,
    "scaling_actions": scaling_actions,
}
