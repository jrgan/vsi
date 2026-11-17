"""
Markov game singleton demo scenarios.

These scenarios use the cheaper singleton value-set iteration (a single
representative point per state instead of the full vertex set), which lets
the scaling sweeps reach larger instances than the full VSI engine.

- ``single_run``        Singleton-iteration trajectory on a small instance.
- ``scaling_states``    Runtime / iteration scaling with the state count.
- ``scaling_actions``   Runtime / iteration scaling with the actions per agent.
- ``scaling_num_agents`` Runtime / iteration scaling with the number of agents.

Full value-set counterparts live in ``demos.markovgame``.

``single_run`` uses ``eps = delta = 0.2``; scaling scenarios use the
coarser ``eps = delta = 0.5``. All scenarios use ``gamma = 0.8``.

Scenarios are registered in ``SCENARIOS`` for CLI dispatch via ``main.py``.
"""
from __future__ import annotations

from typing import Any

from experiments.common.config import OutputConfig
from experiments.markovgame.config import ExperimentConfig, AlgorithmConfig, InstanceConfig
from experiments.markovgame.views import ScalingView, SingletonView

_OUTPUT_ROOT = "results/demos/markovgame_singleton"


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


def _max_iters(overrides: dict[str, Any] | None, default: int = 1000) -> int:
    if overrides and overrides.get("max_iters") is not None:
        return int(overrides["max_iters"])
    return default


def single_run(overrides: dict[str, Any] | None = None):
    """Singleton-iteration trajectory (3 states, 2 agents, 3 actions each)."""
    config = ExperimentConfig(
        name="mg_singleton_run",
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
        algorithm=AlgorithmConfig(eps=0.2, delta=0.2, parallel_by_state=True),
        output=OutputConfig(
            output_dir=_output_dir("single_run", overrides), save_history=True,
        ),
    )
    _apply_overrides(config, overrides)
    SingletonView(
        config,
        max_iters=_max_iters(overrides),
        show_background_hull=False,
    ).run_and_save()


def _scaling_config(
    name: str,
    subdir: str,
    overrides: dict[str, Any] | None,
    **instance: Any,
) -> ExperimentConfig:
    """Singleton scaling base config with coarser eps = delta = 0.5."""
    params: dict[str, Any] = dict(
        method="random_symmetric",
        num_states=3,
        num_agents=2,
        num_actions_per_agent=3,
        reward_min=-1.0,
        reward_max=1.0,
        gamma=0.8,
        seed=42,
    )
    params.update(instance)
    config = ExperimentConfig(
        name=name,
        instance=InstanceConfig(**params),
        algorithm=AlgorithmConfig(eps=0.5, delta=0.5, parallel_by_state=False),
        output=OutputConfig(
            output_dir=_output_dir(subdir, overrides), save_history=False,
        ),
    )
    _apply_overrides(config, overrides)
    return config


def scaling_states(overrides: dict[str, Any] | None = None):
    """Singleton runtime / iteration scaling with the number of states."""
    config = _scaling_config("mg_singleton_scaling_states", "scaling_states", overrides)
    ScalingView(
        config, sizes=[3, 5, 7, 9], axis="states",
        singleton=True, max_iters=_max_iters(overrides),
    ).run_and_save()


def scaling_actions(overrides: dict[str, Any] | None = None):
    """Singleton runtime / iteration scaling with the actions per agent."""
    config = _scaling_config("mg_singleton_scaling_actions", "scaling_actions", overrides)
    ScalingView(
        config, sizes=[3, 5, 7, 9], axis="actions",
        singleton=True, max_iters=_max_iters(overrides),
    ).run_and_save()


def scaling_num_agents(overrides: dict[str, Any] | None = None):
    """Singleton runtime / iteration scaling with the number of agents."""
    config = _scaling_config("mg_singleton_scaling_num_agents", "scaling_num_agents", overrides)
    ScalingView(
        config, sizes=[2, 3, 4, 5], axis="num_agents",
        singleton=True, max_iters=_max_iters(overrides),
    ).run_and_save()


SCENARIOS = {
    "single_run":        single_run,
    "scaling_states":    scaling_states,
    "scaling_actions":   scaling_actions,
    "scaling_num_agents": scaling_num_agents,
}
