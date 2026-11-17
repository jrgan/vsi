"""
SCMDP experiment configuration.

Defines the structure of an experiment configuration, including instance
settings, algorithm parameters, and output options. Provides utilities
to serialize or uniquely identify a configuration.
"""
from dataclasses import dataclass, field
from typing import Optional

from experiments.common.config import (
    BaseConfig, BaseAlgorithmConfig, BaseExperimentConfig, OutputConfig, PolicyConfig,
)


@dataclass
class InstanceConfig(BaseConfig):
    """
    Parameters for generating an SCMDP instance.

    Attributes:
        method: Instance generation method name (e.g. "uniform_const").
        num_states: Number of states.
        num_actions: Number of actions.
        reward_dim: Number of reward dimensions.
        gamma: Discount factor.
        seed: Random seed for reproducibility.
        reward_min: Lower bound for uniform reward sampling.
        reward_max: Upper bound for uniform reward sampling.
        bound: Uniform constraint bound.
    """
    method: str = "unspecified_method"
    num_states: int = 3
    num_actions: int = 2
    reward_dim: int = 2
    gamma: float = 0.9
    seed: Optional[int] = None
    reward_min: float = -1.0
    reward_max: float = 1.0
    bound: float = 10.0


@dataclass
class AlgorithmConfig(BaseAlgorithmConfig):
    """SCMDP algorithm parameters."""
    pass


@dataclass
class ExperimentConfig(BaseExperimentConfig):
    """Complete SCMDP experiment configuration."""
    instance: InstanceConfig = field(default_factory=InstanceConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        """Deserialize configuration from a dictionary."""
        return cls(
            name=d.get("name", "experiment"),
            instance=InstanceConfig.from_dict(d.get("instance", {})),
            algorithm=AlgorithmConfig.from_dict(d.get("algorithm", {})),
            policy=PolicyConfig.from_dict(d.get("policy", {})),
            output=OutputConfig.from_dict(d.get("output", {})),
        )

    def run_label(self) -> str:
        """Return a human-readable label for this run."""
        ic = self.instance
        return (
            f"{self.name}_{ic.method}"
            f"_S{ic.num_states}_A{ic.num_actions}"
            f"_d{ic.reward_dim}_g{ic.gamma}_{self.uid()}"
        )
