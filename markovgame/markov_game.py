"""
Tabular Markov game (stochastic game) in explicit tensor form.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
import itertools
import numpy as np

# Type aliases
Action = int
JointVec = tuple[Action, ...]
State = int
Player = int


# ============================================================
# Markov Game Class
# ============================================================

@dataclass
class MarkovGame:
    """
    Tabular Markov game (stochastic game) in explicit tensor form.

    After validation, transition probabilities of magnitude at most ``prob_tol``
    are set to zero and each row is normalised.

    Attributes:
        rewards:              Reward tensor, shape ``(N, S, A)``, where
                              ``N`` is the number of agents, ``S`` the
                              number of states, and ``A`` the number of
                              joint actions.
        transitions:          Transition tensor, shape ``(S, A, S)``.
        num_actions_per_agent: Number of individual actions for each agent.
        gamma:                Discount factor in ``[0, 1)``.
        prob_tol:             Finite, nonnegative transition cleanup tolerance.
                              Zero preserves positive probabilities and rejects negatives.
    """
    rewards: np.ndarray  # (N, S, A)
    transitions: np.ndarray  # (S, A, S)
    num_actions_per_agent: list[int]
    gamma: float  # Discount factor
    prob_tol: float = field(default=1e-12, kw_only=True)

    # ---------- Computed after init ----------
    joint_idx_to_vec: dict[int, JointVec] = field(init=False, repr=False)
    joint_vec_to_idx: dict[JointVec, int] = field(init=False, repr=False)

    def __post_init__(self):
        """Convert inputs to arrays and validate the tabular model."""
        self.rewards = np.asarray(self.rewards, dtype=float)
        self.transitions = np.asarray(self.transitions, dtype=float)
        self.gamma = float(self.gamma)
        self.prob_tol = float(self.prob_tol)

        if self.rewards.ndim != 3:
            raise ValueError(
                f"rewards must have shape (N, S, A), got {self.rewards.shape}"
            )
        N, S, A = self.rewards.shape
        if N == 0 or S == 0 or A == 0:
            raise ValueError("rewards dimensions N, S, and A must be positive")

        if self.transitions.shape != (S, A, S):
            raise ValueError(
                f"transitions must have shape ({S}, {A}, {S}), "
                f"got {self.transitions.shape}"
            )

        for name, values in (
            ("rewards", self.rewards),
            ("transitions", self.transitions),
        ):
            if not np.isfinite(values).all():
                raise ValueError(f"{name} must contain only finite values")
        if not np.isfinite(self.gamma) or not 0 <= self.gamma < 1:
            raise ValueError("gamma must be finite and in [0, 1)")
        if not np.isfinite(self.prob_tol) or self.prob_tol < 0:
            raise ValueError("prob_tol must be finite and nonnegative")

        if len(self.num_actions_per_agent) != N or any(
            not isinstance(n, (int, np.integer)) or n <= 0
            for n in self.num_actions_per_agent
        ):
            raise ValueError(
                "num_actions_per_agent must contain one positive integer "
                "per agent"
            )
        self.num_actions_per_agent = [
            int(n) for n in self.num_actions_per_agent
        ]

        # Build joint action index-vector maps
        self.joint_idx_to_vec = {
            idx: vec
            for idx, vec in enumerate(
                itertools.product(*(range(n) for n in self.num_actions_per_agent))
            )
        }
        self.joint_vec_to_idx = {
            vec: idx
            for idx, vec in self.joint_idx_to_vec.items()
        }

        expected_A = len(self.joint_idx_to_vec)
        if A != expected_A:
            raise ValueError(
                f"rewards has {A} joint actions, but num_actions_per_agent "
                f"implies {expected_A} joint actions."
            )

        if (self.transitions < -self.prob_tol).any():
            raise ValueError("transitions contains negative entries")
        if not np.allclose(
            self.transitions.sum(axis=2), 1.0, atol=1e-10, rtol=0,
        ):
            raise ValueError("Each transitions[s, a, :] must sum to 1")

        # Remove roundoff-sized probabilities and normalise each row so
        # optimisation and rollout sampling use the same valid distribution.
        self.transitions = np.where(
            self.transitions > self.prob_tol, self.transitions, 0.0,
        )
        totals = self.transitions.sum(axis=2, keepdims=True)
        if (totals == 0).any():
            raise ValueError("prob_tol trims all probabilities in a transition row")
        self.transitions /= totals

    # ---------- Dimensions ----------
    @property
    def num_states(self) -> int:
        """Number of states ``S`` (``rewards.shape[1]``)."""
        return self.rewards.shape[1]

    @property
    def num_agents(self) -> int:
        """Number of agents ``N`` (``rewards.shape[0]``)."""
        return self.rewards.shape[0]

    # ---------- Build from functions ----------
    @classmethod
    def from_functions(
            cls,
            num_states: int,
            num_agents: int,
            num_actions_per_agent: list[int],
            trans_fn: Callable[[State, JointVec], np.ndarray],  # (s, a_joint) -> s' probs
            reward_fn: Callable[[State, JointVec], np.ndarray],  # (s, a_joint) -> r vector
            gamma: float,
            *,
            prob_tol: float = 1e-12,
    ) -> "MarkovGame":
        """
        Materialize transitions and rewards by enumerating (s, joint a)
        using user functions.
        """
        joint_vecs = list(
            itertools.product(*(range(n) for n in num_actions_per_agent))
        )

        S = num_states
        A = len(joint_vecs)
        n = num_agents

        transitions = np.zeros((S, A, S), dtype=float)
        rewards = np.zeros((n, S, A), dtype=float)

        for s in range(S):
            for idx, jv in enumerate(joint_vecs):
                transitions[s, idx, :] = np.asarray(trans_fn(s, jv), dtype=float).ravel()
                rewards[:, s, idx] = np.asarray(reward_fn(s, jv), dtype=float).ravel()

        return cls(
            rewards=rewards,
            transitions=transitions,
            num_actions_per_agent=num_actions_per_agent,
            gamma=gamma,
            prob_tol=prob_tol,
        )

    # ---------- Joint action index-vector mapping ----------
    def get_joint_idx(self, joint_vec: JointVec) -> int:
        """
        Get index of a joint action.
        """
        return self.joint_vec_to_idx[joint_vec]

    def get_joint_vec(self, joint_idx: int) -> JointVec:
        """
        Get joint action vector for a joint action index.
        """
        return self.joint_idx_to_vec[joint_idx]

    # ---------- helper methods ----------
    def joint_idx_after_deviation(
            self,
            joint_idx: int,
            player: Player,
            deviate_to: Action,
    ) -> int:
        """
        Return the joint-action index after *player* deviates to *deviate_to*.
        """

        vec = self.joint_idx_to_vec[joint_idx]

        new_vec = list(vec)
        new_vec[player] = deviate_to

        return self.joint_vec_to_idx[
            tuple(new_vec)
        ]
