"""
SCMDP toy example showing nonstationary-policy advantage.

The key phenomenon is that optimal history-dependent strategies are
strictly better than stationary ones under state-dependent cost
budgets. The model has three non-terminal states ``s1``, ``s2``,
``s3``. Both ``s1`` and ``s2`` transition to ``s3``, where action
choice creates a reward-cost tradeoff.

To mirror the paper's randomized start (``s1`` or ``s2`` with equal
probability), this implementation uses an auxiliary pre-initial state
``s0`` whose transition (for every action) splits 50/50 into ``s1`` and
``s2`` with zero reward.

States (indices):

    0 = s0   auxiliary initial state (-> s1, s2 each w.p. 1/2)
    1 = s1
    2 = s2
    3 = s3
    4 = terminal (absorbing, zero reward)

Actions: ``a`` (0) and ``b`` (1).

Dynamics (deterministic apart from ``s0``):

    s0 -> {s1: 1/2, s2: 1/2}      (both actions)
    s1 -> s3                      (both actions)
    s2 -> s3                      (both actions)
    s3 -> terminal                (both actions)
    terminal -> terminal          (both actions)

Rewards are two-dimensional ``(reward, cost)``.  Everything is zero
except at ``s3`` where action ``b`` yields ``(1, 1)`` (action ``a``
yields ``(0, 0)``).

Cost constraints bound the achievable cumulative cost (the second
reward dimension) per state:

    s1: cost <= 0
    s2: cost <= gamma
    s3: unconstrained

with ``gamma`` defaulting to ``0.8``. Under the budget at ``s1`` the
policy cannot afford the costly action ``b`` downstream (its discounted
cost ``gamma`` exceeds 0), whereas from ``s2`` it can.

Because both states share the same downstream state ``s3`` but have
different feasible cost budgets, a stationary strategy at ``s3`` cannot
be optimal for both cases simultaneously. Conditioning on history
(whether the trajectory came from ``s1`` or ``s2``) attains strictly
better performance. States without a real constraint use a vacuous
``0 <= 0`` row.

Workflow:

    1. Define transition and reward functions over (state, action).
    2. Build an ``SCMDP`` with ``SCMDP.from_functions``.
    3. Run ``VertexRepEngine`` via ``IterationRunner``.
    4. Print a summary and save iteration and policy illustrations.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from scmdp import SCMDP, SCMDPLPFactory, scmdp_grid
from vsi import (
    VertexRepEngine, IterationRunner, box_corners, uniform_value_sets,
)
from vsi.value_set_engine import StepOutcome
from examples._support import value_sets_to_point_lists, save_example_run


# States.
S0, S1, S2, S3, TERMINAL = 0, 1, 2, 3, 4
NUM_STATES = 5
NUM_ACTIONS = 2          # a = 0, b = 1
REWARD_DIM = 2           # (reward, cost)
GAMMA = 0.8


def _reward_fn(s: int, a: int) -> np.ndarray:
    # Only the costly action b at s3 yields a non-zero (reward, cost).
    if s == S3 and a == 1:
        return np.array([1.0, 1.0])
    return np.zeros(REWARD_DIM)


def _trans_fn(s: int, a: int) -> np.ndarray:
    probs = np.zeros(NUM_STATES, dtype=float)
    if s == S0:
        probs[S1] = 0.5
        probs[S2] = 0.5
    elif s == S1 or s == S2:
        probs[S3] = 1.0
    elif s == S3:
        probs[TERMINAL] = 1.0
    else:  # TERMINAL is absorbing
        probs[TERMINAL] = 1.0
    return probs


def build_instance() -> SCMDP:
    """Construct the nonstationary-advantage SCMDP via ``SCMDP.from_functions``."""
    # Single constraint slot per state: cost (dim 1) <= bound.  A zero row
    # with bound 0 (the vacuous 0 <= 0) marks an unconstrained state.
    cons_coeffs = np.zeros((NUM_STATES, 1, REWARD_DIM))
    cons_bounds = np.zeros((NUM_STATES, 1))

    cost_selector = np.array([0.0, 1.0])  # picks out the cost dimension
    cons_coeffs[S1, 0] = cost_selector
    cons_bounds[S1, 0] = 0.0               # s1: cost <= 0
    cons_coeffs[S2, 0] = cost_selector
    cons_bounds[S2, 0] = GAMMA             # s2: cost <= gamma

    return SCMDP.from_functions(
        num_states=NUM_STATES,
        num_actions=NUM_ACTIONS,
        reward_dim=REWARD_DIM,
        trans_fn=_trans_fn,
        reward_fn=_reward_fn,
        cons_coeffs=cons_coeffs,
        cons_bounds=cons_bounds,
        gamma=GAMMA,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger(__name__)

    scmdp = build_instance()

    grid = scmdp_grid(scmdp, eps=0.1, delta=0.1)
    engine = VertexRepEngine(SCMDPLPFactory(scmdp), grid)
    engine.set_param(vsi_operator_parallel=True)

    runner = IterationRunner(engine)
    runner.run(
        init_value_sets=uniform_value_sets(box_corners(grid.lbs, grid.ubs), NUM_STATES),
        max_iters=200,
        keep_history=True,
    )
    log.info(
        "VSI finished: %s (%d iterations).", runner.status.name, runner.num_iterations,
    )

    point_lists = value_sets_to_point_lists(runner.result)
    for s, pts in enumerate(point_lists):
        log.info("State %d: %d vertices", s, len(pts))

    out_dir = Path("results/examples/scmdp")
    save_example_run(scmdp, engine, runner, out_dir, "nonstationary_advantage",
                     lambda action, state: StepOutcome(action, state))


if __name__ == "__main__":
    main()
