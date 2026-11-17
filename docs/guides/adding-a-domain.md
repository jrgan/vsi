# Adding a Domain

This guide walks through plugging a new problem domain into the
**`vsi/vertex_rep/`** engines. It assumes you have read
[vsi-design.md](vsi-design.md) for the overall architecture; for a runnable
minimal instance to read alongside this walkthrough, see
[examples/scmdp/two_state.py](../../examples/scmdp/two_state.py), whose structure
it mirrors.

A domain supplies a factory, usually backed by a model object, and a grid helper
for the grid-based engines. `SingletonEngine` needs no grid.

| piece | role | example (`scmdp/`) |
| --- | --- | --- |
| a model object *(optional)* | holds your problem data | `SCMDP` |
| a `ValueSetLPFactory` subclass | `_build_model(...)` — the per-state LP (the real hook) | `SCMDPLPFactory` |
| a grid helper | turns a target accuracy into a `Grid` | `scmdp_grid` |

The same factory works with all three vertex-rep engines and both runners.
The domain need not be an MDP: discount factors, transition tensors, and Bellman
equations belong to the supplied domains, not the general engine contract.

---

## 1. State count and value dimension

The VSI engine and grid only need two quantities from your domain:

| Quantity | Where it is surfaced | Meaning |
| --- | --- | --- |
| `num_states` | `ValueSetLPFactory.num_states` | how many states the engine iterates over |
| `reward_dim` (`d`) | `ValueSetLPFactory.reward_dim` + `Grid.dim` | the dimension of the value space the grid lives in |

`reward_dim` is the number of components in each value vector; those components
need not all be rewards.

### The (optional) model object

A model holds the data used by the factory and grid helper. The engine accesses
it only through the factory.

For the MDP-based domains we currently ship (`SCMDP`, `MarkovGame`), the `_build_model`
implementation is a Bellman backup, so its model exposes the usual MDP data:

- `gamma` — discount factor (used in the Bellman equality and in the
  grid-spacing formula);
- reward and transition tensors — the per-state Bellman data;
- domain constraint data (e.g. SCMDP's `cons_coeffs` / `cons_bounds`).

A convenient pattern (used by both shipped domains) is a `from_functions`
constructor that builds the tensors from callables, plus cached `num_states` /
`reward_dim` properties:

```python
problem = SCMDP.from_functions(
    num_states=...,
    num_actions=...,
    reward_dim=...,
    reward_fn=...,
    trans_fn=...,
    ...,
)
```

---

## 2. The `ValueSetLPFactory` subclass

This is where the real domain logic lives. Subclass
`vsi.vertex_rep.value_set_lp.ValueSetLPFactory`, bind your model, and implement
three members:

```python
from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB

from vsi.value_set_engine import State, StepOutcome
from vsi.vertex_rep.value_set_lp import (
    LPSolverParams, ValueSetLPModel, ValueSetLPFactory,
)


@dataclass(frozen=True)
class MyValueSetLPFactory(ValueSetLPFactory[StepOutcome]):
    model: MyModel

    @property
    def num_states(self) -> int:
        return self.model.num_states

    @property
    def reward_dim(self) -> int:
        return self.model.reward_dim          # the value-space dimension d

    def _build_model(
        self, s0, onward_value_sets, *,
        params: LPSolverParams, solver_env: gp.Env | None = None,
    ) -> ValueSetLPModel[StepOutcome] | None:
        ...
```

The base class provides `open_lp(...)` and `open_milp(...)`. Both call your
`_build_model` hook; the latter also adds the integer grid constraints. You only
implement the model-building hook, not separate LP and MILP opening methods.
Solver settings are inherited keyword-only fields. `MyValueSetLPFactory(model)`
uses the defaults; pass `lp_params` or `milp_params` to customize them (see
[solver settings](vsi-design.md#solver-settings)).

### The `_build_model` contract

`_build_model(s0, onward_value_sets, *, params, solver_env=None)` builds the
Gurobi LP for the **exact** feasible set of values at state `s0`, given the onward
value sets from the previous iterate. It returns a `ValueSetLPModel`, or `None` when
the state is trivially infeasible. Create the model with
`gp.Model(env=solver_env)` so parallel workers use their own Gurobi
environments. The opening methods supply either the shared quiet serial
environment or a worker environment. Direct calls to `_build_model` with
`None` use Gurobi's default environment.

A `ValueSetLPModel` is a passive container with four fields:

```python
return ValueSetLPModel(
    solver=lp,                      # the gp.Model
    value_var=v,                    # MVar, shape (d,) — the achievable value vector
    act_probs_var=pi,               # MVar — the action distribution
    onward_weights_var=onward_weights,  # dict[StepOutcome, MVar] — convex weights
)
```

Key rules:

- **Return `None` on trivial infeasibility** (e.g. no action has all of its
  onward sets non-empty), before creating a model. Do not try to build a
  slack-padded LP to represent emptiness — the engine handles empty states for
  you.
- **Dispose the model if construction raises.** The factory passes a successfully
  returned model to a session, which disposes it when it closes. The execution
  helper owns worker environments and keeps them alive until the workers stop.
- **Do not add slack.** The LP must encode the *exact* feasible set. The engine's
  `LPSession` adds the slacked companion variable itself when needed.
- `value_var` must have shape `(reward_dim,)`.
- `onward_weights_var` is keyed by the `StepOutcome`s whose onward set is
  non-empty; each weight vector has one entry per onward vertex, and is coupled
  to `act_probs_var` so the branch weights sum to the action probability.

### The MDP-based recipe (what SCMDP/MarkovGame do)

For an MDP-style domain, `_build_model` encodes a Bellman backup. The SCMDP factory,
for reference, builds:

1. a value variable `v` of shape `(d,)` and an action distribution `pi`;
2. onward-weight variables, one block per non-empty `StepOutcome`;
3. the **Bellman equality** linking `v` to immediate reward plus the discounted
   convex combination of onward vertices;
4. **coupling** constraints tying each onward-weight block's sum to `pi[a]`;
5. the distribution constraint `pi.sum() == 1`;
6. domain-specific constraints (SCMDP adds `cons_coeffs[s0] @ v <= cons_bounds[s0]`).

See [scmdp/vslp_factory.py](../../scmdp/vslp_factory.py) for the full, commented
implementation, and [markovgame/vslp_factory.py](../../markovgame/vslp_factory.py)
for a richer variant whose branches carry joint actions.

### Custom `StepOutcome`

The default `StepOutcome(action, state)` (frozen, hashable) covers domains where a
branch is identified by "the action taken and the next state reached." If your
branches carry more — e.g. a *joint* action across players — subclass
`StepOutcome` (as `markovgame/` does) and parameterise the factory with it:
`MyValueSetLPFactory(ValueSetLPFactory[MyStepOutcome])`.

---

## 3. The grid helper

The grid helper converts a target accuracy into a `Grid` over value space. It is
itself a **domain-supplied** piece: the framework only needs the `Grid` it returns
to carry a `dim` and a `grid_spacing` (and, for bounded regimes, integer index
bounds `k_min` / `k_max`).

The `scmdp_grid` helper is a compact reference:

```python
def scmdp_grid(scmdp, *, eps, delta) -> Grid:
    gamma = scmdp.gamma
    grid_spacing = (1 - gamma) * min(eps, delta / 2)

    raw_lbs = scmdp.rewards.min(axis=(1, 2)) / (1 - gamma)
    raw_ubs = scmdp.rewards.max(axis=(1, 2)) / (1 - gamma)

    k_min = np.ceil(raw_lbs / grid_spacing).astype(int) - 1
    k_max = np.floor(raw_ubs / grid_spacing).astype(int) + 1

    return Grid(dim=raw_lbs.shape[0], grid_spacing=grid_spacing,
                k_min=k_min, k_max=k_max)
```

The `grid_spacing = (1 - gamma) * min(eps, delta/2)` formula and the
reward-derived value box are the MDP-based recipe our domains use — a non-MDP
domain is free to compute `grid_spacing` and the bounds any way it likes, or to
hand-build a `Grid` directly. See the [associated paper](../../README.md#associated-paper)
for why that spacing achieves an `(eps, delta)` guarantee for MDP domains.

---

## 4. Wiring it together

With the three pieces in place, every engine in the family is one constructor
call away:

```python
from vsi import (
    VertexRepEngine, SingletonEngine, IterationRunner,
    box_corners, uniform_value_sets,
)
from vsi.vertex_rep.baselines import GridEnumEngine

model = MyModel.from_functions(...)
grid = my_grid(model, eps=0.2, delta=0.2)
lp_factory = MyValueSetLPFactory(model)

engine = VertexRepEngine(lp_factory, grid)  # or GridEnumEngine(lp_factory, grid)
runner = IterationRunner(engine)
runner.run(
    init_value_sets=uniform_value_sets(
        box_corners(grid.lbs, grid.ubs), model.num_states,
    ),
    max_iters=200,
)
final_sets = runner.result
```

Swapping the engine class changes nothing else:

```python
engine = SingletonEngine(lp_factory)  # LP/QP solve, one point per state
```

Finally, export the public pieces from your domain package's `__init__.py` so
callers can `from mydomain import MyModel, MyValueSetLPFactory, my_grid`.

---

## 5. Checklist

**Framework contract (always required):**

- [ ] `ValueSetLPFactory` subclass exposing `num_states` and `reward_dim` (the
      value-space dimension `d`).
- [ ] `_build_model(s0, onward_value_sets, *, params, solver_env=None)` returning a
      `ValueSetLPModel` (or `None` when trivially infeasible), using the supplied
      environment and disposing its model if construction fails.
- [ ] A `Grid` (via a helper or hand-built) with the right `dim` / `grid_spacing`
      (not needed for `SingletonEngine`).
- [ ] A `StepOutcome` subclass if branches carry more than `(action, state)`.
- [ ] Export the pieces from your domain package `__init__.py`.
- [ ] A correctness test (cross-check `VertexRepEngine` against `GridEnumEngine`
      on a coarse grid, and/or against a known fixed point).

**Application-specific (MDP-based domains like ours):**

- [ ] Model object holding `gamma`, reward/transition tensors, and any constraint
      data your `_build_model()` reads, ideally with a `from_functions` constructor.
- [ ] Grid helper using the `(1 - gamma) * min(eps, delta/2)` spacing recipe.
