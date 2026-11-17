"""
SCMDP value-set LP factory for VSI.

Defines :class:`SCMDPLPFactory`, the problem-bound factory that
creates the per-state LP encoding the SCMDP Bellman equations along
with the per-state cost constraints.
"""
from __future__ import annotations

from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB

from scmdp.scmdp import SCMDP
from vsi.value_set_engine import State, StepOutcome
from vsi.vertex_rep.value_set_lp import (
    LPSolverParams, ValueSetLPModel, ValueSetLPFactory,
)


@dataclass(frozen=True)
class SCMDPLPFactory(ValueSetLPFactory[StepOutcome]):
    """
    Factory for the per-state value-set LP of a subprocess-constrained
    MDP.

    Binds one :class:`~scmdp.scmdp.SCMDP` instance and, on each
    :meth:`_build_model` call, encodes the SCMDP Bellman equation in the
    reward dimensions plus the per-state linear cost constraints
    ``cons_coeffs[s0] @ v <= cons_bounds[s0]``.

    Attributes:
        scmdp: The bound SCMDP problem instance.
    """

    scmdp: SCMDP

    @property
    def num_states(self) -> int:
        return self.scmdp.num_states

    @property
    def reward_dim(self) -> int:
        return self.scmdp.reward_dim

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
        """Build the SCMDP Bellman + cost-constraint LP at *s0*."""
        scmdp = self.scmdp
        rewards = scmdp.rewards
        trans = scmdp.transitions
        cons_coeffs = scmdp.cons_coeffs
        cons_bounds = scmdp.cons_bounds
        gamma = scmdp.gamma

        d = scmdp.reward_dim
        num_states = rewards.shape[1]
        num_actions = rewards.shape[2]

        states = range(num_states)
        actions = range(num_actions)

        # Trivial infeasibility: no action has all reachable
        # next-states with non-empty value sets.
        if not any(
                all(
                len(onward_value_sets[s]) > 0
                for s in states
                if trans[s0, a, s] > 0
            )
                for a in actions
        ):
            return None

        step_outcomes = [
            StepOutcome(action=a, state=s)
            for a in range(num_actions)
            for s in range(num_states)
        ]

        lp = gp.Model(env=solver_env)
        try:
            lp.setParam("OutputFlag", 0)
            lp.setParam("OptimalityTol", params.opt_tol)
            lp.setParam("FeasibilityTol", params.fsb_tol)

            # --- Variables ---
            v = lp.addMVar(shape=(d,), lb=-GRB.INFINITY, ub=GRB.INFINITY)
            pi = lp.addMVar(shape=(num_actions,), lb=0, ub=1)

            onward_weights = {}
            for stp_out in step_outcomes:
                s = stp_out.state
                if len(onward_value_sets[s]) > 0:
                    onward_weights[stp_out] = lp.addMVar(
                        shape=(len(onward_value_sets[s]),), lb=0,
                    )

            # --- Constraints ---

            # 1. Bellman equality
            lp.addConstrs(
                v[i]
                == rewards[i, s0, :] @ pi
                + gamma
                * sum(
                    onward_weights[stp_out]
                    @ onward_value_sets[stp_out.state][:, i]
                    * trans[s0, stp_out.action, stp_out.state]
                    for stp_out in step_outcomes
                    if stp_out in onward_weights
                )
                for i in range(d)
            )

            # 2. Onward-value coupling
            for stp_out in step_outcomes:
                if stp_out in onward_weights:
                    lp.addConstr(
                        pi[stp_out.action] == onward_weights[stp_out].sum()
                    )

            # 3. Force pi[a] = 0 for actions with empty next-states
            lp.addConstrs(
                pi[a] == 0
                for a in actions
                for s in states
                if len(onward_value_sets[s]) == 0 and trans[s0, a, s] > 0
            )

            # 4. pi is a distribution
            lp.addConstr(pi.sum() == 1)

            # 5. Cost constraints
            lp.addConstr(cons_coeffs[s0] @ v <= cons_bounds[s0])

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
