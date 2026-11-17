"""
Markov game instance generation and persistence utilities.
"""
import logging
from pathlib import Path

import numpy as np

from markovgame import MarkovGame
from experiments.markovgame.config import InstanceConfig

logger = logging.getLogger(__name__)


# =========================================================================== #
#  Instance generation
# =========================================================================== #

def generate_instance(config: InstanceConfig) -> MarkovGame:
    """Generate a Markov game instance based on the configured method.

    Args:
        config: Instance configuration with method discriminator and params.

    Returns:
        A generated MarkovGame instance.

    Raises:
        ValueError: If ``config.method`` is not recognized.
    """
    if config.method == "random_symmetric":
        return _generate_random_symmetric(
            num_states=config.num_states,
            num_agents=config.num_agents,
            num_actions_per_agent=config.num_actions_per_agent,
            gamma=config.gamma,
            reward_min=config.reward_min,
            reward_max=config.reward_max,
            seed=config.seed,
        )
    else:
        raise ValueError(
            f"Unknown instance generation method: '{config.method}'. "
            f"Supported methods: 'random_symmetric'."
        )


def _generate_random_symmetric(
    num_states: int,
    num_agents: int,
    num_actions_per_agent: int,
    gamma: float,
    reward_min: float = -1.0,
    reward_max: float = 1.0,
    seed: int | None = None,
) -> MarkovGame:
    """Generate a random Markov game where every agent has the same action count.

    Rewards are drawn uniformly from [reward_min, reward_max] and transition
    probabilities are drawn from a Dirichlet(1,...,1) distribution.

    Args:
        num_states: Number of states (S).
        num_agents: Number of agents (N).
        num_actions_per_agent: Individual actions per agent.
        gamma: Discount factor.
        reward_min: Lower bound for uniform reward sampling.
        reward_max: Upper bound for uniform reward sampling.
        seed: Random seed.

    Returns:
        A randomly generated MarkovGame instance.
    """
    rng = np.random.default_rng(seed)
    S = num_states
    N = num_agents

    actions_list = [num_actions_per_agent] * N
    A = num_actions_per_agent ** N  # number of joint actions

    # Random rewards: (N, S, A)
    rewards = rng.uniform(reward_min, reward_max, size=(N, S, A))

    # Random transitions: (S, A, S)
    transitions = rng.dirichlet(np.ones(S), size=(S, A))

    return MarkovGame(rewards, transitions, actions_list, gamma)


# =========================================================================== #
#  Instance I/O
# =========================================================================== #

def save_instance(game: MarkovGame, directory) -> Path:
    """Save a MarkovGame instance to ``directory/instance.npz``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "instance.npz"
    np.savez(
        path,
        rewards=game.rewards,
        transitions=game.transitions,
        num_actions_per_agent=np.array(game.num_actions_per_agent),
        gamma=np.array([game.gamma]),
        prob_tol=np.array([game.prob_tol]),
    )
    logger.info("Instance saved to %s", path)
    return path


def load_instance(directory) -> MarkovGame:
    """Load a MarkovGame instance from ``directory/instance.npz``."""
    data = np.load(Path(directory) / "instance.npz")
    return MarkovGame(
        rewards=data["rewards"],
        transitions=data["transitions"],
        num_actions_per_agent=data["num_actions_per_agent"].tolist(),
        gamma=float(data["gamma"][0]),
        prob_tol=float(data["prob_tol"][0]) if "prob_tol" in data else 1e-12,
    )

