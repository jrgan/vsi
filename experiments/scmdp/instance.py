"""
SCMDP instance generation and persistence utilities.

Provides factory functions for creating SCMDP instances with various
constraint structures, and I/O for saving/loading instances to disk.
"""
import logging
from pathlib import Path

import numpy as np

from scmdp import SCMDP
from experiments.scmdp.config import InstanceConfig

logger = logging.getLogger(__name__)


# =========================================================================== #
#  Instance generation
# =========================================================================== #

def generate_instance(config: InstanceConfig) -> SCMDP:
    """
    Generate an SCMDP instance based on the configured method.

    Dispatches to the appropriate generator function based on
    ``config.method``.

    Args:
        config: Instance configuration with method discriminator and params.

    Returns:
        A generated SCMDP instance.

    Raises:
        ValueError: If ``config.method`` is not recognized.
    """
    if config.method == "uniform_const":
        return _generate_uniform_const_instance(
            num_states=config.num_states,
            num_actions=config.num_actions,
            rewards_dim=config.reward_dim,
            gamma=config.gamma,
            bound=config.bound,
            reward_min=config.reward_min,
            reward_max=config.reward_max,
            seed=config.seed,
        )
    else:
        raise ValueError(
            f"Unknown instance generation method: '{config.method}'. "
            f"Supported methods: 'uniform_const'."
        )


def _generate_uniform_const_instance(
    num_states: int,
    num_actions: int,
    rewards_dim: int,
    gamma: float,
    bound: float,
    reward_min: float = -1.0,
    reward_max: float = 1.0,
    seed: int | None = None,
) -> SCMDP:
    """
    Generate a random SCMDP with uniform box constraints v[i] <= bound.

    Rewards are drawn uniformly from [reward_min, reward_max] and transition
    probabilities are drawn from a Dirichlet(1,...,1) distribution.

    The constraints are of the form v[i] <= bound for i = 1, ..., d-1 at every
    state (dimension 0 is left unconstrained), i.e. cons_coeffs[s] selects
    dimensions 1..d-1 and cons_bounds[s] = [bound, ..., bound].

    Args:
        num_states: Number of states (S).
        num_actions: Number of actions (A).
        rewards_dim: Number of reward dimensions (d).
        gamma: Discount factor.
        bound: Upper bound for each value dimension at every state.
        reward_min: Lower bound for uniform reward sampling. Default: -1.0.
        reward_max: Upper bound for uniform reward sampling. Default: 1.0.
        seed: Random seed for reproducibility. Default: None.

    Returns:
        A randomly generated SCMDP instance.
    """
    rng = np.random.default_rng(seed)
    S, A, d = num_states, num_actions, rewards_dim

    # Random rewards: (d, S, A), uniform in [reward_min, reward_max]
    rewards = rng.uniform(reward_min, reward_max, size=(d, S, A))

    # Random transitions: (S, A, S), each row drawn from Dirichlet(1,...,1)
    transitions = rng.dirichlet(np.ones(S), size=(S, A))

    # Uniform box constraints: v[i] <= bound for i = 1, ..., d-1 (skip dim 0)
    cons_coeffs = np.zeros((S, d-1, d))
    cons_coeffs[:, np.arange(d-1), np.arange(1, d)] = 1.0
    cons_bounds = np.full((S, d-1), bound)

    return SCMDP(rewards, transitions, cons_coeffs, cons_bounds, gamma)


# =========================================================================== #
#  Instance I/O
# =========================================================================== #

def save_instance(scmdp: SCMDP, directory) -> Path:
    """Save an SCMDP instance to ``directory/instance.npz``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "instance.npz"
    np.savez(
        path,
        rewards=scmdp.rewards,
        transitions=scmdp.transitions,
        cons_coeffs=scmdp.cons_coeffs,
        cons_bounds=scmdp.cons_bounds,
        gamma=np.array([scmdp.gamma]),
        prob_tol=np.array([scmdp.prob_tol]),
    )
    logger.info("Instance saved to %s", path)
    return path


def load_instance(directory) -> SCMDP:
    """Load an SCMDP instance from ``directory/instance.npz``."""
    data = np.load(Path(directory) / "instance.npz")
    return SCMDP(
        rewards=data["rewards"],
        transitions=data["transitions"],
        cons_coeffs=data["cons_coeffs"],
        cons_bounds=data["cons_bounds"],
        gamma=float(data["gamma"][0]),
        prob_tol=float(data["prob_tol"][0]) if "prob_tol" in data else 1e-12,
    )

