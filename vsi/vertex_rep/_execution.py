"""Per-state execution, progress reporting, and Gurobi worker ownership."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from queue import SimpleQueue
from threading import local

import gurobipy as gp

from vsi.value_set_engine import Points
from ._progress import RichPerStateProgress


def map_states(
    compute_fn,
    value_sets,
    *,
    lp_factory,
    num_states: int,
    parallel: bool = False,
    max_workers: int | None = None,
    show_progress: bool = True,
) -> list[Points]:
    """Apply *compute_fn* once per state and return results in state order.

    Each call receives ``(state, value_sets, sink, lp_factory=...)``.
    Parallel workers exclusively own separate Gurobi environments; sequential
    calls use the shared quiet environment. Completion is reported after the
    computation returns, using the number of points actually returned.
    """
    worker_local = local()
    with RichPerStateProgress(num_states, enabled=show_progress) as progress:
        def run_state(state):
            sink = progress.sink_for(state)
            opener = worker_local.lp_factory if parallel else lp_factory
            points = compute_fn(state, value_sets, sink, lp_factory=opener)
            sink.finish(len(points))
            return points

        if not parallel:
            return [run_state(state) for state in range(num_states)]

        num_workers = min(num_states, max_workers or os.cpu_count() or 4)
        # Gurobi environment initialization is not thread-safe, so create
        # worker environments on the calling thread before starting workers.
        with ExitStack() as env_stack:
            unassigned_factories = SimpleQueue()
            for _ in range(num_workers):
                env = env_stack.enter_context(gp.Env(params={
                    "OutputFlag": 0,
                    "Threads": 1,
                }))
                unassigned_factories.put(lp_factory.bind_environment(env))

            def initialise_worker():
                worker_local.lp_factory = unassigned_factories.get()

            # Join the workers before disposing any of their environments.
            with ThreadPoolExecutor(
                max_workers=num_workers, initializer=initialise_worker,
            ) as executor:
                futures = [
                    executor.submit(run_state, state)
                    for state in range(num_states)
                ]
                return [future.result() for future in futures]
