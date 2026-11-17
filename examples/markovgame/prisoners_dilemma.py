"""
Repeated Prisoner's Dilemma as a two-agent Markov game.

Two states:

* ``s`` (index 0): the *play* state and initial state.
* ``t`` (index 1): an absorbing *terminal* state with zero reward.

Each agent has two actions, ``C`` (cooperate, index 0) and ``D``
(defect, index 1), giving four joint actions per state in the order
``(C,C), (C,D), (D,C), (D,D)``.

Dynamics: from ``s`` the game stays in ``s`` only when both agents
cooperate ``(C,C)``; any defection sends it to the terminal state
``t``.  Once in ``t`` it stays there forever (absorbing) with zero
reward, so its achievable value set converges to ``{0}``.

Stage rewards at ``s`` (one component per agent):

    (C,C) -> ( 1,  1)
    (C,D) -> (-1,  2)
    (D,C) -> ( 2, -1)
    (D,D) -> ( 0,  0)

The discount factor ``GAMMA`` is the one parameter not pinned down by
the problem statement; it is exposed below and defaults to ``0.67``
(high enough for cooperation to be sustainable in the repeated game).

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


# States: 0 = s (play / initial), 1 = t (terminal, absorbing).
PLAY, TERMINAL = 0, 1
NUM_STATES = 2
NUM_AGENTS = 2
NUM_ACTIONS_PER_AGENT = [2, 2]
GAMMA = 0.67

# Joint actions enumerate as (C,C), (C,D), (D,C), (D,D) with C=0, D=1.
COOPERATE = (0, 0)

# Per-agent stage rewards at the play state, keyed by joint action vector.
PLAY_REWARDS = {
    (0, 0): [ 1.0,  1.0],    # (C, C)
    (0, 1): [-1.0,  2.0],    # (C, D)
    (1, 0): [ 2.0, -1.0],    # (D, C)
    (1, 1): [ 0.0,  0.0],    # (D, D)
}


def _reward_fn(s: int, jv: tuple[int, ...]) -> np.ndarray:
    # Only the play state yields stage rewards; the terminal state is zero.
    if s == PLAY:
        return np.asarray(PLAY_REWARDS[jv], dtype=float)
    return np.zeros(NUM_AGENTS, dtype=float)


def _trans_fn(s: int, jv: tuple[int, ...]) -> np.ndarray:
    probs = np.zeros(NUM_STATES, dtype=float)
    if s == PLAY and jv == COOPERATE:
        probs[PLAY] = 1.0          # stay in the game only under mutual cooperation
    else:
        probs[TERMINAL] = 1.0      # any defection -> terminal; terminal is absorbing
    return probs


def build_instance() -> MarkovGame:
    """Construct the repeated Prisoner's Dilemma via ``MarkovGame.from_functions``."""
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
    save_example_run(game, engine, runner, out_dir, "prisoners_dilemma",
                     lambda action, state: StepOutcome(action, state, action))


if __name__ == "__main__":
    main()
