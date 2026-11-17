"""
Markov game experiment configuration.

Mirrors the SCMDP configuration layout so that the experiment runner
and the generic orchestration utilities work identically.
"""
from dataclasses import dataclass, field
from typing import Optional

from experiments.common.config import (
    BaseConfig, BaseAlgorithmConfig, BaseExperimentConfig, OutputConfig, PolicyConfig,
)


@dataclass
class InstanceConfig(BaseConfig):
    """
    Parameters for generating a Markov game instance.

    Attributes:
        method: Instance generation method (e.g. "random_symmetric").
        num_states: Number of states.
        num_agents: Number of agents.
        num_actions_per_agent: Number of actions per agent (assumed uniform).
        gamma: Discount factor.
        seed: Random seed for reproducibility.
        reward_min: Lower bound for uniform reward sampling.
        reward_max: Upper bound for uniform reward sampling.
    """
    method: str = "random_symmetric"
    num_states: int = 3
    num_agents: int = 2
    num_actions_per_agent: int = 2
    gamma: float = 0.9
    seed: Optional[int] = None
    reward_min: float = -1.0
    reward_max: float = 1.0


@dataclass
class AlgorithmConfig(BaseAlgorithmConfig):
    """Markov game algorithm parameters."""
    pass


@dataclass
class ExperimentConfig(BaseExperimentConfig):
    """Complete Markov game experiment configuration."""
    name: str = "mg_experiment"
    instance: InstanceConfig = field(default_factory=InstanceConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        return cls(
            name=d.get("name", "mg_experiment"),
            instance=InstanceConfig.from_dict(d.get("instance", {})),
            algorithm=AlgorithmConfig.from_dict(d.get("algorithm", {})),
            policy=PolicyConfig.from_dict(d.get("policy", {})),
            output=OutputConfig.from_dict(d.get("output", {})),
        )

    def run_label(self) -> str:
        ic = self.instance
        return (
            f"{self.name}_{ic.method}"
            f"_S{ic.num_states}_N{ic.num_agents}"
            f"_A{ic.num_actions_per_agent}_g{ic.gamma}_{self.uid()}"
        )

