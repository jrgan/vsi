"""
SCMDP demo scenarios (full value-set iteration).

Each scenario builds a config and runs the corresponding view class from
``experiments.scmdp.views``. Each scenario showcases a different VSI
capability:

- ``single_run``      End-to-end VSI value-set iteration + policy rollout.
- ``bound_sweep``     Effect of the constraint bound on the achievable set.
- ``scaling_states``  Runtime / iteration scaling with the state count.
- ``scaling_actions`` Runtime / iteration scaling with the action count.

Singleton-iteration counterparts (and the cheaper reward-dimension scaling
sweep) live in ``demos.scmdp_singleton``.

Non-scaling scenarios use ``eps = delta = 0.1``; scaling scenarios use the
coarser ``eps = delta = 0.2`` to keep sweep runtimes manageable. The base
instance (``S=4, A=5, d=2, gamma=0.8, bound=2.0``) matches the paper baseline.

Scenarios are registered in ``SCENARIOS`` for CLI dispatch via ``main.py``.
"""
from __future__ import annotations

from typing import Any

from experiments.common.config import OutputConfig, PolicyConfig
from experiments.scmdp.config import ExperimentConfig, AlgorithmConfig, InstanceConfig
from experiments.scmdp.views import (
    SingleRunView, BoundSweepView, ScalingView,
)

_OUTPUT_ROOT = "results/demos/scmdp"


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


def _base_instance(**overrides: Any) -> InstanceConfig:
    """Paper-aligned base instance: S=4, A=5, d=2, gamma=0.8, bound=2.0."""
    params: dict[str, Any] = dict(
        method="uniform_const",
        num_states=4,
        num_actions=5,
        reward_dim=2,
        reward_min=-1.0,
        reward_max=1.0,
        gamma=0.8,
        bound=2.0,
        seed=42,
    )
    params.update(overrides)
    return InstanceConfig(**params)


def single_run(overrides: dict[str, Any] | None = None):
    """End-to-end value-set iteration with policy rollout (paper baseline)."""
    config = ExperimentConfig(
        name="scmdp_single_run",
        instance=_base_instance(),
        algorithm=AlgorithmConfig(eps=0.1, delta=0.1, parallel_by_state=True),
        policy=PolicyConfig(
            objective=[1.0, 0.0], init_state=0, num_steps=50, seed=123,
        ),
        output=OutputConfig(
            output_dir=_output_dir("single_run", overrides), save_history=True,
        ),
    )
    _apply_overrides(config, overrides)
    SingleRunView(config).run_and_save()


def bound_sweep(overrides: dict[str, Any] | None = None):
    """Overlay achievable sets for a sweep of constraint bounds."""
    reward_max, reward_min = 1.0, -1.0
    config = ExperimentConfig(
        name="scmdp_bound_sweep",
        instance=_base_instance(),
        algorithm=AlgorithmConfig(eps=0.1, delta=0.1, parallel_by_state=True),
        output=OutputConfig(
            output_dir=_output_dir("bound_sweep", overrides), save_history=False,
        ),
    )
    _apply_overrides(config, overrides)
    gamma = config.instance.gamma
    ratios = [1.0, 0.8, 0.6, 0.4, 0.2]
    bounds = [
        (r * reward_max + (1 - r) * reward_min) / (1 - gamma) for r in ratios
    ]
    BoundSweepView(config, bounds).run_and_save()


def _scaling_config(name: str, subdir: str, overrides: dict[str, Any] | None):
    """Scaling base config: paper instance with coarser eps = delta = 0.2."""
    config = ExperimentConfig(
        name=name,
        instance=_base_instance(),
        algorithm=AlgorithmConfig(eps=0.2, delta=0.2, parallel_by_state=True),
        output=OutputConfig(
            output_dir=_output_dir(subdir, overrides), save_history=False,
        ),
    )
    _apply_overrides(config, overrides)
    return config


def scaling_states(overrides: dict[str, Any] | None = None):
    """Runtime / iteration scaling with the number of states."""
    config = _scaling_config("scmdp_scaling_states", "scaling_states", overrides)
    ScalingView(config, sizes=[3, 5, 7, 9], axis="states").run_and_save()


def scaling_actions(overrides: dict[str, Any] | None = None):
    """Runtime / iteration scaling with the number of actions."""
    config = _scaling_config("scmdp_scaling_actions", "scaling_actions", overrides)
    ScalingView(config, sizes=[3, 5, 7, 9], axis="actions").run_and_save()


SCENARIOS = {
    "single_run":     single_run,
    "bound_sweep":    bound_sweep,
    "scaling_states": scaling_states,
    "scaling_actions": scaling_actions,
}
