"""
Markov-game value-set LP factory for VSI.

Defines :class:`MarkovGameLPFactory`, the problem-bound factory
that creates the per-state LP encoding the Bellman equations for every
agent together with Nash incentive-compatibility (IC) constraints that
forbid any single agent from benefiting by a unilateral deviation.

Also defines the Markov-game :class:`StepOutcome` shape used by the
LP (extending :class:`vsi.value_set_engine.StepOutcome` with a
``action_played`` index).
"""
from __future__ import annotations

from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB

from markovgame.markov_game import MarkovGame
from vsi.value_set_engine import (
    State,
    StepOutcome as _StepOutcome,
)
from vsi.vertex_rep.value_set_lp import (
    LPSolverParams, ValueSetLPModel, ValueSetLPFactory,
)


@dataclass(frozen=True)
class StepOutcome(_StepOutcome):
    """
    Step outcome for a Markov game.

    Attributes:
        action (inherited):     Recommended joint-action index (on-path).
        state (inherited):      Successor state index.
        action_played:          Joint-action index actually played (may differ
                                from *action* under a unilateral deviation).
    """
    action_played: int


@dataclass(frozen=True)
class MarkovGameLPFactory(ValueSetLPFactory[StepOutcome]):
    """
    Factory for the per-state value-set LP of a Markov game.

    Binds one :class:`~markovgame.markov_game.MarkovGame` instance.
    On each :meth:`_build_model` call, the LP searches for a correlated
    strategy (distribution over joint actions) and convex-combination
    weights over the onward value sets such that:

    1. The Bellman equation holds for every agent.
    2. No agent can improve their expected payoff by a unilateral
       deviation (Nash IC constraints).

    The ``onward_weights`` variables are indexed by
    ``StepOutcome(action=A, action_played=B, state=s')``.  On-path
    weights use ``onward_weights[A, A, s']``; deviation weights use
    ``onward_weights[A, dev(A, i, b), s']``.

    Attributes:
        game: The bound Markov-game problem instance.
    """

    game: MarkovGame

    @property
    def num_states(self) -> int:
        return self.game.num_states

    @property
    def reward_dim(self) -> int:
        return self.game.num_agents

    # ------------------------------------------------------------------
    # LP construction
    # ------------------------------------------------------------------

    def _build_model(
        self,
        s0: State,
        onward_value_sets,
        *,
        params: LPSolverParams,
        solver_env: gp.Env | None = None,
    ) -> ValueSetLPModel[StepOutcome] | None:
        """Build the Markov-game Bellman + Nash-IC LP at *s0*."""
        game = self.game
        rewards = game.rewards          # (N, S, A)
        trans = game.transitions        # (S, A, S)
        gamma = game.gamma

        num_agents = game.num_agents
        num_states = rewards.shape[1]
        num_joint_actions = rewards.shape[2]

        # Each outcome's weights sum to its action probability. An empty
        # onward set would force every action probability to zero.
        if any(len(points) == 0 for points in onward_value_sets):
            return None

        states = range(num_states)
        joint_actions = range(num_joint_actions)
        agents = range(num_agents)
        agent_actions = [range(game.num_actions_per_agent[i]) for i in agents]

        step_outcomes = [
            StepOutcome(action=A, state=s, action_played=B)
            for A in range(num_joint_actions)
            for B in range(num_joint_actions)
            for s in range(num_states)
        ]

        lp = gp.Model(env=solver_env)
        try:
            lp.setParam("OutputFlag", 0)
            lp.setParam("OptimalityTol", params.opt_tol)
            lp.setParam("FeasibilityTol", params.fsb_tol)

            # --- Variables ---
            v = lp.addMVar(shape=(num_agents,), lb=-GRB.INFINITY)
            pi = lp.addMVar(shape=(num_joint_actions,), lb=0, ub=1)
            onward_weights = {
                stp_out: lp.addMVar(
                    shape=(len(onward_value_sets[stp_out.state]),), lb=0,
                )
                for stp_out in step_outcomes
            }

            def get_onward_weights(A: int, B: int, s: int) -> gp.MVar | None:
                return onward_weights.get(
                    StepOutcome(action=A, state=s, action_played=B)
                )

            # --- Constraints ---

            # 1. Bellman equality for every agent i
            lp.addConstrs(
                v[i]
                == rewards[i, s0, :] @ pi
                + gamma
                * sum(
                    get_onward_weights(A, A, s)
                    @ onward_value_sets[s][:, i]
                    * trans[s0, A, s]
                    for A in joint_actions
                    for s in states
                )
                for i in agents
            )

            # 2. Onward-value coupling
            lp.addConstrs(
                pi[A] == get_onward_weights(A, B, s).sum()
                for A in joint_actions
                for B in joint_actions
                for s in states
            )

            # 3. pi is a distribution
            lp.addConstr(pi.sum() == 1)

            # 4. Incentive-compatibility constraints
            def deviate(A: int, i: int, b: int) -> int:
                return game.joint_idx_after_deviation(A, i, b)

            lp.addConstrs(
                0 <=
                gp.quicksum(
                    pi[A]
                    * (rewards[i, s0, A] - rewards[i, s0, deviate(A, i, b)])
                    + gamma * gp.quicksum(
                        get_onward_weights(A, A, s)
                        @ onward_value_sets[s][:, i]
                        * trans[s0, A, s]
                        - get_onward_weights(A, deviate(A, i, b), s)
                        @ onward_value_sets[s][:, i]
                        * trans[s0, deviate(A, i, b), s]
                        for s in states
                    )
                    for A in joint_actions if game.get_joint_vec(A)[i] == a
                )
                for i in agents
                for a in agent_actions[i]
                for b in agent_actions[i]
            )

            lp.update()

            return ValueSetLPModel(
                solver=lp,
                value_var=v,
                act_probs_var=pi,
                onward_weights_var=onward_weights,
            )
        except Exception:
            lp.dispose()
            raise
