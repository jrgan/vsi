"""Policy rollout runner over a ValueSetEngine.

Given achievable value sets and an initial policy, it offers stateful,
step-by-step rollout utilities.
"""
from __future__ import annotations

from typing import Generic, Any

import numpy as np

from .value_set_engine import (
    ValueSetEngine,
    ValueSetsT,
    Policy,
    StepOutcome,
    ExpansionResult,
)


# ============================================================
# Exceptions
# ============================================================

class InfeasiblePolicyError(Exception):
    """Raised when an iteration step cannot proceed because an optimization is infeasible."""
    pass


# ============================================================
# PolicyRunner Class
# ============================================================

class PolicyRunner(Generic[ValueSetsT]):
    """
    Wraps a ValueSetEngine, achievable value sets, an initial policy, and a
    seed to provide stateful, step-by-step policy rollout.

    Read-only properties expose the rollout state: ``next_policy``,
    ``history``, ``rollout_history``, and ``step_count``.
    """

    def __init__(
            self,
            engine: ValueSetEngine[ValueSetsT, Any],
            achievable_value_sets: ValueSetsT,
            policy: Policy,
            seed: int | None = None,
    ) -> None:
        self._engine = engine
        self._achievable_value_sets = achievable_value_sets
        self._initial_policy = policy
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # Mutable rollout state
        self._next_policy: Policy = policy
        self._history: list[StepOutcome] = []
        self._rollout_history: list[ExpansionResult] = []
        self._awaiting_outcome = False

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def engine(self) -> ValueSetEngine[ValueSetsT, Any]:
        """The engine whose ``expand_value`` drives this rollout."""
        return self._engine

    @property
    def achievable_value_sets(self) -> ValueSetsT:
        """The converged value sets the policy is expanded against."""
        return self._achievable_value_sets

    @property
    def next_policy(self) -> Policy:
        """The policy to be expanded on the next call to ``next_step``."""
        return self._next_policy

    @property
    def history(self) -> list[StepOutcome]:
        """Step outcomes recorded by ``receive_step_outcome``."""
        return self._history

    @property
    def rollout_history(self) -> list[ExpansionResult]:
        """``ExpansionResult`` objects produced by ``next_step``."""
        return self._rollout_history

    @property
    def step_count(self) -> int:
        """Number of completed steps (``next_step`` + ``receive_step_outcome`` pairs)."""
        return len(self._history)

    # ------------------------------------------------------------------ #
    # Core methods
    # ------------------------------------------------------------------ #
    def next_step(self) -> ExpansionResult:
        """
        Expand the current policy by one step.

        Calls ``ValueSetEngine.expand_value`` with the current
        policy's value and initial state.

        Returns:
            expansion_result: an ``ExpansionResult`` with attributes
            ``action_probs``, ``onward_policies``, and ``value_achieved``.

        Raises:
            RuntimeError: if the preceding expansion has not yet received
                its step outcome.
            InfeasiblePolicyError: if the policy cannot be expanded.
        """
        if self._awaiting_outcome:
            raise RuntimeError(
                "next_step called before the preceding step outcome was received."
            )

        step = len(self._rollout_history) + 1
        expansion_result = self._engine.expand_value(
            self._next_policy.value,
            self._next_policy.init_state,
            self._achievable_value_sets,
        )

        if expansion_result is None:
            raise InfeasiblePolicyError(
                f"Policy infeasible at Step {step}."
            )

        self._rollout_history.append(expansion_result)
        self._awaiting_outcome = True
        return expansion_result

    def receive_step_outcome(self, step_outcome: StepOutcome) -> None:
        """
        Record observed step info and advance to the corresponding
        onward policy.

        Args:
            step_outcome: Key into the onward_policies dict returned by the
                       most recent call to next_step.

        Raises:
            RuntimeError: if called before next_step or called twice
                          without an intervening next_step.
        """
        if not self._awaiting_outcome:
            raise RuntimeError(
                "receive_step_outcome called without a preceding next_step."
            )

        last_result = self._rollout_history[-1]
        # Resolve the key before changing any runner state.  A caller may
        # recover from an invalid outcome and submit the correct one.
        next_policy = last_result.onward_policies[step_outcome]
        self._next_policy = next_policy
        self._history.append(step_outcome)
        self._awaiting_outcome = False

    def sample_action(self) -> int:
        """
        Sample an action index from the latest action distribution.
        """
        if not self._awaiting_outcome:
            raise RuntimeError(
                "sample_action requires an expansion awaiting its step outcome."
            )
        action_probs = self._rollout_history[-1].action_probs
        return int(self._rng.choice(len(action_probs), p=action_probs))

    def restart(self) -> None:
        """
        Reset the runner to its initial state.
        """
        self._next_policy = self._initial_policy
        self._history = []
        self._rollout_history = []
        self._awaiting_outcome = False
        self._rng = np.random.default_rng(self._seed)
