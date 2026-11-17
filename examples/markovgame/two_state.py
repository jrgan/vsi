"""
Hand-built two-state, two-agent Markov game example.

Both agents have two actions each, giving four joint actions per state.
Rewards are zero-sum-like over a 2D vector (one component per agent),
which is small enough to sketch on paper while still exercising the
joint-action machinery in ``MarkovGame``.

Workflow:

    1. Define transition and reward functions over (state, joint_action).
    2. Build a ``MarkovGame`` with ``MarkovGame.from_functions``.
    3. Run ``VertexRepEngine`` via ``IterationRunner``.
    4. Print a summary and save iteration and policy illustrations.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from markovgame import MarkovGame, MarkovGameLPFactory, markovgame_grid
from vsi import (
    VertexRepEngine, IterationRunner, box_corners, uniform_value_sets,
)
from markovgame.vslp_factory import StepOutcome
from examples._support import value_sets_to_point_lists, save_example_run


NUM_STATES = 2
NUM_AGENTS = 2
NUM_ACTIONS_PER_AGENT = [2, 2]
GAMMA = 0.7

# Per-agent reward table keyed by (state, joint_action_vec) -> [r_agent0, r_agent1].
REWARDS = {
    (0, (0, 0)): [1.0, -1.0],
    (0, (0, 1)): [-0.5, 0.5],
    (0, (1, 0)): [0.3, -0.3],
    (0, (1, 1)): [0.8,  0.2],
    (1, (0, 0)): [-0.2, 0.6],
    (1, (0, 1)): [0.4,  0.0],
    (1, (1, 0)): [0.1,  0.1],
    (1, (1, 1)): [0.9, -0.4],
}

# Next-state distribution keyed by (state, joint_action_vec) -> P(s' | s, joint a).
TRANSITIONS = {
    (0, (0, 0)): [0.8, 0.2],
    (0, (0, 1)): [0.5, 0.5],
    (0, (1, 0)): [0.3, 0.7],
    (0, (1, 1)): [0.6, 0.4],
    (1, (0, 0)): [0.4, 0.6],
    (1, (0, 1)): [0.7, 0.3],
    (1, (1, 0)): [0.2, 0.8],
    (1, (1, 1)): [0.5, 0.5],
}


def _reward_fn(s: int, jv: tuple[int, ...]) -> np.ndarray:
    return np.asarray(REWARDS[(s, jv)], dtype=float)


def _trans_fn(s: int, jv: tuple[int, ...]) -> np.ndarray:
    return np.asarray(TRANSITIONS[(s, jv)], dtype=float)


def build_instance() -> MarkovGame:
    """Construct the toy Markov game via ``MarkovGame.from_functions``."""
    return MarkovGame.from_functions(
        num_states=NUM_STATES,
        num_agents=NUM_AGENTS,
        num_actions_per_agent=NUM_ACTIONS_PER_AGENT,
        trans_fn=_trans_fn,
        reward_fn=_reward_fn,
        gamma=GAMMA,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger(__name__)

    game = build_instance()

    grid = markovgame_grid(game, eps=0.1, delta=0.1)
    engine = VertexRepEngine(MarkovGameLPFactory(game), grid)
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

    out_dir = Path("results/examples/markovgame")
    save_example_run(game, engine, runner, out_dir, "two_state",
                     lambda action, state: StepOutcome(action, state, action))


if __name__ == "__main__":
    main()
