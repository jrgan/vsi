"""
Hand-built two-state SCMDP example.

Demonstrates the minimal path from problem definition to a value-set
iteration result:

    1. Define transition and reward functions over (state, action).
    2. Build an ``SCMDP`` with ``SCMDP.from_functions``.
    3. Drive ``VertexRepEngine`` directly via ``IterationRunner``.
    4. Print a summary and save iteration and policy illustrations.

The instance has 2 states, 2 actions, and 2 reward dimensions, which
makes it small enough to verify by hand.
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


NUM_STATES = 2
NUM_ACTIONS = 2
REWARD_DIM = 2
GAMMA = 0.7
BOUND = 3.0

# Reward table indexed as REWARDS[state][action] -> [r0, r1].
REWARDS = {
    (0, 0): [1.0,  0.0],
    (0, 1): [0.0,  1.0],
    (1, 0): [0.8,  0.2],
    (1, 1): [-0.2, 0.9],
}

# Transition probabilities to each next state: TRANSITIONS[state][action] -> P(s' | s, a).
TRANSITIONS = {
    (0, 0): [0.9, 0.1],
    (0, 1): [0.4, 0.6],
    (1, 0): [0.3, 0.7],
    (1, 1): [0.5, 0.5],
}


def _reward_fn(s: int, a: int) -> np.ndarray:
    return np.asarray(REWARDS[(s, a)], dtype=float)


def _trans_fn(s: int, a: int) -> np.ndarray:
    return np.asarray(TRANSITIONS[(s, a)], dtype=float)


def build_instance() -> SCMDP:
    """Construct the toy SCMDP via ``SCMDP.from_functions``."""
    # Box constraint v[i] <= BOUND for i = 1, ..., d-1 at every state
    # (matches the convention used in experiments/scmdp/instance.py).
    cons_coeffs = np.zeros((NUM_STATES, REWARD_DIM - 1, REWARD_DIM))
    cons_coeffs[:, np.arange(REWARD_DIM - 1), np.arange(1, REWARD_DIM)] = 1.0
    cons_bounds = np.full((NUM_STATES, REWARD_DIM - 1), BOUND)

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
    save_example_run(scmdp, engine, runner, out_dir, "two_state",
                     lambda action, state: StepOutcome(action, state))


if __name__ == "__main__":
    main()
