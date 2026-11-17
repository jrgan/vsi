"""
Base configuration classes for experiments.

Provides generic dataclass bases for instance, algorithm, and output
configuration. Problem-specific modules subclass or compose these.
"""
import hashlib
import json
from dataclasses import dataclass, asdict
from enum import Enum, auto
from typing import Optional

import numpy as np


class VertexSetMode(Enum):
    """
    Experiment-level tag selecting which VSI engine to instantiate.

    Resolved to a concrete engine class in
    :mod:`experiments.common.runner` (``ENGINE_BY_MODE``):

    * ``GRID_VERTICES`` -> :class:`vsi.VertexRepEngine`
    * ``SINGLE_POINT``  -> :class:`vsi.SingletonEngine`
    * ``ALL_GRID_POINTS`` -> :class:`vsi.vertex_rep.baselines.GridEnumEngine`
    """
    GRID_VERTICES = auto()
    SINGLE_POINT = auto()
    ALL_GRID_POINTS = auto()


@dataclass
class BaseConfig:
    """Abstract base for all configuration dataclasses."""

    @classmethod
    def from_dict(cls, d: dict) -> "BaseConfig":
        """Construct an instance from a dictionary."""
        return cls(**d)

    def to_dict(self) -> dict:
        """Serialize to a dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=_json_default)

    @classmethod
    def from_json(cls, s: str) -> "BaseConfig":
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(s))


@dataclass
class OutputConfig(BaseConfig):
    """Options controlling what to record and where to save."""
    output_dir: str = "results"
    save_history: bool = True
    save_final: bool = True
    save_instance: bool = True
    save_config: bool = True
    image_formats: tuple[str, ...] = ("png", "pdf")
    image_dpi: int = 150


@dataclass
class PolicyConfig(BaseConfig):
    """Parameters for policy extraction via linear scalarisation.

    Attributes:
        objective: Weight vector for linear scalarisation.  If ``None``,
            defaults to maximising the first reward dimension.
        init_state: Starting state for the policy rollout.
        num_steps: Number of trajectory steps to record.
        seed: Random seed for policy rollout.
    """
    objective: Optional[list[float]] = None
    init_state: int = 0
    num_steps: int = 50
    seed: int = 123

    def objective_array(self, reward_dim: int) -> np.ndarray:
        """Return the objective as a numpy array.

        Falls back to maximising the first dimension when *objective*
        is ``None``.
        """
        if self.objective is not None:
            return np.asarray(self.objective, dtype=float)
        obj = np.zeros(reward_dim)
        obj[0] = 1.0
        return obj


def _json_default(obj):
    """JSON serializer for numpy types and enums."""
    if isinstance(obj, Enum):
        return obj.name
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


@dataclass
class BaseAlgorithmConfig(BaseConfig):
    """Shared algorithm parameters for VSI experiments."""
    eps: float = 0.5
    delta: float = 0.5
    max_iters: Optional[int] = None
    parallel_by_state: bool = False
    mode: VertexSetMode = VertexSetMode.GRID_VERTICES

    @classmethod
    def from_dict(cls, d: dict) -> "BaseAlgorithmConfig":
        d = dict(d)
        if "mode" in d and isinstance(d["mode"], str):
            d["mode"] = VertexSetMode[d["mode"]]
        return cls(**d)


@dataclass
class BaseExperimentConfig(BaseConfig):
    """Base experiment configuration providing ``uid()``."""
    name: str = "experiment"

    def uid(self) -> str:
        """Return a unique identifier (short hash) for this configuration."""
        content = json.dumps(self.to_dict(), sort_keys=True, default=_json_default)
        return hashlib.sha256(content.encode()).hexdigest()[:12]
