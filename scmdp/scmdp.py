"""
Tabular subprocess-constrained MDP (SCMDP) in explicit tensor form.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np


# ============================================================
# SCMDP Class
# ============================================================
@dataclass
class SCMDP:
    """
    Tabular subprocess-constrained MDP in explicit tensor form.

    After validation, transition probabilities of magnitude at most ``prob_tol``
    are set to zero and each row is normalised.

    Attributes:
        rewards:     Reward tensor, shape ``(d, S, A)``, where ``d`` is
                     the reward dimension, ``S`` the number of states,
                     and ``A`` the number of actions.
        transitions: Transition tensor, shape ``(S, A, S)``.
        cons_coeffs: Per-state cost-constraint coefficients, shape
                     ``(S, m, d)`` for ``m`` constraints.
        cons_bounds: Per-state cost-constraint bounds, shape ``(S, m)``.
        gamma:       Discount factor in ``[0, 1)``.
        prob_tol:    Finite, nonnegative transition cleanup tolerance. Zero
                     preserves all positive probabilities and rejects negatives.
    """
    rewards: np.ndarray  # (d, S, A)
    transitions: np.ndarray  # (S, A, S)
    cons_coeffs: np.ndarray  # (S, m, d)
    cons_bounds: np.ndarray  # (S, m)
    gamma: float  # Discount factor
    prob_tol: float = field(default=1e-12, kw_only=True)

    # ---------- Computed after init ----------
    def __post_init__(self):
        """Convert inputs to arrays and validate the tabular model."""
        self.rewards = np.asarray(self.rewards, dtype=float)
        self.transitions = np.asarray(self.transitions, dtype=float)
        self.cons_coeffs = np.asarray(self.cons_coeffs, dtype=float)
        self.cons_bounds = np.asarray(self.cons_bounds, dtype=float)
        self.gamma = float(self.gamma)
        self.prob_tol = float(self.prob_tol)

        if self.rewards.ndim != 3:
            raise ValueError(
                f"rewards must have shape (d, S, A), got {self.rewards.shape}"
            )
        d, S, A = self.rewards.shape
        if d == 0 or S == 0 or A == 0:
            raise ValueError("rewards dimensions d, S, and A must be positive")

        if self.cons_coeffs.ndim != 3:
            raise ValueError(
                "cons_coeffs must have shape (S, m, d), "
                f"got {self.cons_coeffs.shape}"
            )
        m = self.cons_coeffs.shape[1]

        if self.transitions.shape != (S, A, S):
            raise ValueError(
                f"transitions must have shape ({S}, {A}, {S}), "
                f"got {self.transitions.shape}"
            )
        if self.cons_coeffs.shape != (S, m, d):
            raise ValueError(
                f"cons_coeffs must have shape ({S}, {m}, {d}), "
                f"got {self.cons_coeffs.shape}"
            )
        if self.cons_bounds.shape != (S, m):
            raise ValueError(
                f"cons_bounds must have shape ({S}, {m}), "
                f"got {self.cons_bounds.shape}"
            )

        for name, values in (
            ("rewards", self.rewards),
            ("transitions", self.transitions),
            ("cons_coeffs", self.cons_coeffs),
            ("cons_bounds", self.cons_bounds),
        ):
            if not np.isfinite(values).all():
                raise ValueError(f"{name} must contain only finite values")

        if not np.isfinite(self.gamma) or not 0 <= self.gamma < 1:
            raise ValueError("gamma must be finite and in [0, 1)")
        if not np.isfinite(self.prob_tol) or self.prob_tol < 0:
            raise ValueError("prob_tol must be finite and nonnegative")

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
    def reward_dim(self) -> int:
        """Reward dimension ``d`` (``rewards.shape[0]``)."""
        return self.rewards.shape[0]

    # ---------- Build from functions ----------
    @classmethod
    def from_functions(
            cls,
            num_states: int,
            num_actions: int,
            reward_dim: int,
            trans_fn: Callable[[int, int], np.ndarray],  # (s, a) -> s' probs
            reward_fn: Callable[[int, int], np.ndarray],  # (s, a) -> r vector
            cons_coeffs: np.ndarray,
            cons_bounds: np.ndarray,
            gamma: float,
            *,
            prob_tol: float = 1e-12,
    ) -> "SCMDP":
        """
        Materialize transitions and rewards by enumerating (s, a) using
        user functions.
        """
        S = num_states
        A = num_actions
        d = reward_dim

        transitions = np.zeros((S, A, S), dtype=float)
        rewards = np.zeros((d, S, A), dtype=float)

        for s in range(S):
            for a in range(A):
                transitions[s, a, :] = np.asarray(trans_fn(s, a), dtype=float).ravel()
                rewards[:, s, a] = np.asarray(reward_fn(s, a), dtype=float).ravel()

        return cls(
            rewards=rewards,
            transitions=transitions,
            cons_coeffs=cons_coeffs,
            cons_bounds=cons_bounds,
            gamma=gamma,
            prob_tol=prob_tol,
        )
