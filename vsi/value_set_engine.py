"""
Abstract base classes for value-set iteration (VSI).

Defines :class:`ValueSetEngine`, the abstract engine that drives VSI,
along with the core data classes :class:`Policy`,
:class:`ExpansionResult`, and :class:`StepOutcome`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Generic

import numpy as np


# Type aliases
ValueSetsT = TypeVar("ValueSetsT")
StepOutcomeT = TypeVar("StepOutcomeT", bound="StepOutcome")

State = int     # Integral state indices
Action = int    # Integral action indices
Point = np.ndarray
Points = np.ndarray


# ============================================================
# Dataclasses
# ============================================================

@dataclass(frozen=True)
class StepOutcome:
    """
    Base class for problem-specific step outcome during policy rollout.

    Frozen so that instances are hashable and can serve as dictionary keys
    in ``ExpansionResult.onward_policies``.

    Subclass and add fields for each problem domain.
    """
    action: Action
    state: State


@dataclass(frozen=True)
class Policy:
    """
    Policy in compact representation.
    """
    value: np.ndarray
    init_state: State


@dataclass(frozen=True)
class ExpansionResult:
    """
    Result of expanding a policy at one step.
    """
    action_probs: np.ndarray
    onward_policies: dict[StepOutcome, Policy]
    value_achieved: Point


# ============================================================
# ValueSetEngine (abstract)
# ============================================================

class ValueSetEngine(ABC, Generic[ValueSetsT, StepOutcomeT]):
    """
    Abstract engine for value-set iteration (VSI).

    Subclasses implement the core operations:

    * :meth:`iterate_once` — map onward value sets to achievable
      value sets (one application of the forward Bellman operator).
    * :meth:`expand_value` — decompose a single achievable value
      back into action probabilities and onward values.
    * :meth:`value_sets_distance` — distance between two value sets,
      used as the convergence criterion.
    """
    @abstractmethod
    def value_sets_distance(
            self,
            before: ValueSetsT,
            after: ValueSetsT,
    ) -> float:
        """Distance between two value sets (for convergence checks)."""
        ...

    @abstractmethod
    def iterate_once(self, value_sets: ValueSetsT) -> ValueSetsT:
        """
        Map onward value sets to achievable value sets.

        This is the core VSI operator: given the current onward value
        sets, compute the set of all achievable values at each state.
        """
        ...

    @abstractmethod
    def expand_value(
            self,
            value: np.ndarray,
            state: State,
            onward_value_sets: ValueSetsT,
    ) -> ExpansionResult | None:
        """
        Decompose an achievable value into onward components.

        Given a target value vector at a given state and the onward
        value sets, check whether the value is achievable and, if so,
        return the action distribution, onward policies, and the
        actual value achieved.

        Returns:
            An :class:`ExpansionResult` if the value is achievable,
            or ``None`` if it is not.
        """
        ...
