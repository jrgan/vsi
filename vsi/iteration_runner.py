"""Iteration runner for value-set iteration (VSI).

Applies the VSI operator until convergence, optional loop detection, or
a maximum iteration cap.
"""
from __future__ import annotations

import itertools
import logging
from copy import deepcopy
from enum import Enum, auto

from typing import Generic, Iterator, Any

from ._validation import as_nonnegative_float, as_positive_int
from .value_set_engine import ValueSetEngine, ValueSetsT

_module_logger = logging.getLogger(__name__)


# ============================================================
# RunnerStatus
# ============================================================

class RunnerStatus(Enum):
    """Status of an IterationRunner after a run call."""
    INITIATED = auto()
    RUNNING = auto()
    CONVERGED = auto()
    LOOP_DETECTED = auto()
    MAX_ITER_REACHED = auto()


# ============================================================
# IterationRunner
# ============================================================

class IterationRunner(Generic[ValueSetsT]):
    """
    Runner for value-set iteration.

    Iterates the VSI operator until convergence, loop detection, or a
    maximum iteration cap.  Results are exposed via read-only properties:
    ``status``, ``result``, ``sequence``, ``loop_back_iter``, and
    ``num_iterations``.
    """

    def __init__(
            self,
            vsi_engine: ValueSetEngine[ValueSetsT, Any],
            convergence_tol: float = 1e-9,
    ):
        convergence_tol = as_nonnegative_float(
            convergence_tol, "convergence_tol",
        )

        self._vsi_engine = vsi_engine
        self._convergence_tol = convergence_tol
        self._logger = _module_logger

        self._status: RunnerStatus = RunnerStatus.INITIATED

        self._result: ValueSetsT | None = None
        self._sequence: list[ValueSetsT] | None = None
        self._loop_back_iter: int | None = None
        self._num_iterations: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def status(self) -> RunnerStatus:
        """Status of the most recent run."""
        return self._status

    @property
    def result(self) -> ValueSetsT | None:
        """Deep copy of the final value sets from the last run."""
        if self._result is None:
            return None
        return deepcopy(self._result)

    @property
    def sequence(self) -> list[ValueSetsT] | None:
        """Sequence of value sets from the last run."""
        return self._sequence

    @property
    def loop_back_iter(self) -> int | None:
        """Iteration index that the run looped back to, or None."""
        return self._loop_back_iter

    @property
    def num_iterations(self) -> int:
        """Number of iterations completed in the last run."""
        return self._num_iterations

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _reset(self):
        """Clear all stored results and sequences."""
        self._result = None
        self._sequence = None
        self._loop_back_iter = None
        self._num_iterations = 0
        self._status = RunnerStatus.INITIATED

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, **kwargs) -> None:
        """Run VSI to completion (see ``run_iter``)."""
        for _ in self.run_iter(**kwargs):
            pass

    def run_iter(
            self,
            *,
            init_value_sets: ValueSetsT,
            max_iters: int | None = None,
            detect_loops: bool = False,
            keep_history: bool = False,
    ) -> Iterator[ValueSetsT]:
        """
        Generator that yields value sets at each iteration.

        Args:
            init_value_sets: Starting value sets for the iteration.
            max_iters: Stop after this many iterations. Must be a positive
                integer when provided.
            detect_loops: If ``True``, check each new iterate against
                all previous iterates (not just the immediately preceding
                one).  When a match to an earlier iterate is found, the
                run terminates with ``LOOP_DETECTED`` status.
            keep_history: If ``True``, store every iterate in
                ``sequence``.  Always ``True`` when *detect_loops*
                is enabled (required for correctness).

        Resets all results and sequences before starting.
        """
        if max_iters is not None:
            max_iters = as_positive_int(max_iters, "max_iters")

        self._reset()
        _store = keep_history or detect_loops
        if _store:
            self._sequence = []
        self._status = RunnerStatus.RUNNING

        vs = init_value_sets
        if _store:
            self._sequence.append(vs)

        for iter_idx in itertools.count(1):
            iteration = f"{iter_idx}/{max_iters}" if max_iters is not None else str(iter_idx)
            self._logger.info("---------- VSI iteration %s ----------", iteration)

            new_vs = self._vsi_engine.iterate_once(vs)

            diff = self._vsi_engine.value_sets_distance(vs, new_vs)
            converged = diff <= self._convergence_tol
            self._logger.info(
                "Distance: %.2e (tolerance %.2e)", diff, self._convergence_tol,
            )

            if detect_loops:
                # Check new_vs against every previous iterate.
                loop_target: int | None = None
                for past_idx, past_vs in enumerate(self._sequence):
                    past_diff = self._vsi_engine.value_sets_distance(past_vs, new_vs)
                    if past_diff <= self._convergence_tol:
                        loop_target = past_idx
                        break

                vs = new_vs
                if _store:
                    self._sequence.append(vs)
                yield vs

                if loop_target is not None:
                    if loop_target == iter_idx - 1:
                        self._logger.info(
                            "VSI converged at iteration %d (distance=%.2e).",
                            iter_idx, diff,
                        )
                        self._result = vs
                        self._status = RunnerStatus.CONVERGED
                    else:
                        self._loop_back_iter = loop_target
                        self._logger.info(
                            "VSI stopped at iteration %d: loop back to "
                            "iteration %d (distance=%.2e to that iterate).",
                            iter_idx, loop_target, past_diff,
                        )
                        self._result = vs
                        self._status = RunnerStatus.LOOP_DETECTED
                    self._num_iterations = iter_idx
                    return
            else:
                vs = new_vs
                if _store:
                    self._sequence.append(vs)
                yield vs

                if converged:
                    self._logger.info(
                        "VSI converged at iteration %d (distance=%.2e).",
                        iter_idx, diff,
                    )
                    self._result = vs
                    self._status = RunnerStatus.CONVERGED
                    self._num_iterations = iter_idx
                    return

            if max_iters is not None and iter_idx >= max_iters:
                self._logger.info(
                    "VSI stopped at iteration %d: limit reached without "
                    "convergence (distance=%.2e, tolerance=%.2e).",
                    iter_idx, diff, self._convergence_tol,
                )
                self._result = vs
                self._status = RunnerStatus.MAX_ITER_REACHED
                self._num_iterations = iter_idx
                return
