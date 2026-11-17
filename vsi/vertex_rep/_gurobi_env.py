"""Quiet Gurobi environment shared by serial sessions and geometry queries.

Created on first use and retained for the process lifetime, including any
license or remote resources it holds. Calls using this environment must not
overlap across application threads. Built-in parallel workers use their own
environments, managed separately by ``_execution.py``.
"""
from __future__ import annotations

import gurobipy as gp


_shared_env: gp.Env | None = None


def shared_env() -> gp.Env:
    """Return the serial environment, suppressing output before startup."""
    global _shared_env
    if _shared_env is None:
        env = gp.Env(empty=True)
        try:
            env.setParam("OutputFlag", 0)
            env.start()
        except Exception:
            env.dispose()
            raise
        _shared_env = env
    return _shared_env
