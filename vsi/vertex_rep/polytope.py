"""
Convex-polytope helpers for the vertex representation.

Geometry utilities operating on value sets stored as point sets
(V-representation): containment slack, symmetric polytope distance,
convex-hull vertex reduction, and support-function evaluation.
"""
import math

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import scipy.spatial as sp

from ._gurobi_env import shared_env

# Type aliases
Points = np.ndarray  # Shape (n, d): n points in d-dimensional space


def _is_empty_point_set(points: np.ndarray) -> bool:
    """Whether *points* has zero rows, even if its dimension is unknown."""
    return points.ndim > 0 and points.shape[0] == 0


# --------------------------------------------------
# Containment Slack
# --------------------------------------------------

def containment_slack(
        P1: Points,
        P2: Points,
        *,
        solver_tol: float = 1e-9,
) -> float:
    """
    Compute the minimum L∞ slack by which conv(P1) must be expanded to contain all points of P2.

    ``solver_tol`` sets the internal LP's feasibility and optimality tolerances;
    it is not a guaranteed bound on distance error.

    Returns:
        0.0 if P2 ⊆ conv(P1) already; otherwise the minimum L∞ amount by which
        each point of P2 must be relaxed to lie in conv(P1).
        Returns inf when P1 is empty but P2 is not.

    Raises:
        ValueError: If non-empty inputs are not 2D or have mismatched dimensions.
        RuntimeError: If LP encounters unexpected status.
    """
    P1 = np.asarray(P1, dtype=float)
    P2 = np.asarray(P2, dtype=float)

    # An empty point set may be represented by an unshaped empty tuple/list.
    # Its ambient dimension is immaterial to these set-theoretic outcomes.
    if _is_empty_point_set(P2):
        return 0.0  # No points need to be contained.
    if _is_empty_point_set(P1):
        return math.inf  # A non-empty set cannot be contained in an empty one.

    # Input validation
    if P1.ndim != 2 or P2.ndim != 2:
        raise ValueError("Each point set must be a 2D array of shape (n, d)")

    d = P1.shape[1]
    if P2.shape[1] != d:
        raise ValueError(
            f"Dimension mismatch: P1 is ({P1.shape[0]}, {d}) but P2 is ({P2.shape[0]}, {P2.shape[1]})."
        )

    # Extract vertices to reduce problem size.
    P1 = convex_hull_vertices(P1)
    P2 = convex_hull_vertices(P2)

    n1, n2 = P1.shape[0], P2.shape[0]

    # --- Set up and solve LP ---
    solver = gp.Model(env=shared_env())
    try:
        solver.setParam('OutputFlag', 0)
        solver.setParam("OptimalityTol", solver_tol)
        solver.setParam("FeasibilityTol", solver_tol)

        # Compute the minimum L-infinity distance by which points in P2 must
        # be relaxed to be contained in conv(P1):
        # ----------------------------------------------------------
        #     min  epsilon
        #     s.t. |P2[j] - P1^T w_j|_inf <= epsilon   for all j
        #          w_j >= 0, sum(w_j) = 1              for all j
        #          epsilon >= 0
        # ----------------------------------------------------------

        # Variables:
        #   W[i, j] = weight of P1[i] in the convex combination for P2[j]
        #   eps = maximum L-infinity violation across all points in P2
        W = solver.addMVar(shape=(n1, n2), lb=0.0, ub=1.0)
        eps = solver.addVar(lb=0.0, name="eps")

        # Constraint: residual must be between -eps and +eps
        residual = P2.T - P1.T @ W  # d x n2
        solver.addConstr(residual <= eps)
        solver.addConstr(residual >= -eps)

        # Constraint: weights sum to 1 for each vertex of P2
        solver.addConstr(W.sum(axis=0) == 1.0)

        # Minimize epsilon
        solver.setObjective(eps, GRB.MINIMIZE)
        solver.optimize()

        if solver.status == GRB.OPTIMAL:
            return float(eps.X)
        raise RuntimeError(
            f"containment_slack: LP returned unexpected status {solver.status}"
        )
    finally:
        solver.dispose()


# --------------------------------------------------
# Polytope Distance
# --------------------------------------------------

def polytope_distance(
        P1: Points,
        P2: Points,
        *,
        solver_tol: float = 1e-9,
) -> float:
    """
    Compute the L∞ Hausdorff distance between conv(P1) and conv(P2).

    Returns max(containment_slack(P1, P2), containment_slack(P2, P1)).
    Mathematically, this is zero iff the convex hulls are equal.
    ``solver_tol`` sets the internal LPs' feasibility and optimality tolerances;
    it is not a guaranteed bound on distance error.
    """

    # Compute distance by which P2 must be relaxed to be contained in conv(P1), 
    # and vice versa.
    P1_P2 = containment_slack(P1, P2, solver_tol=solver_tol)
    P2_P1 = containment_slack(P2, P1, solver_tol=solver_tol)

    return max(P1_P2, P2_P1) 


# --------------------------------------------------
# Convex Hull Vertices
# --------------------------------------------------

def convex_hull_vertices(P: Points) -> Points:
    """
    Reduce a point set to its convex hull vertices.

    Empty point sets are accepted without a known ambient dimension. Their
    existing shape is retained, including ``(0, d)`` when it is available.

    Raises:
        ValueError: If a non-empty *P* is not a 2D array of shape ``(n, d)``.
    """
    P = np.asarray(P, dtype=float)
    if _is_empty_point_set(P):
        return P.copy()
    if P.ndim != 2:
        raise ValueError("Point set must be a 2D array of shape (n, d)")

    d = P.shape[1]
    trimmed = P.copy()

    if d == 1:
        # Qhull needs at least 2D; the interval's vertices are its ends.
        lo, hi = trimmed[:, 0].min(), trimmed[:, 0].max()
        return np.array([[lo]]) if lo == hi else np.array([[lo], [hi]])

    # Extract vertices only if there are enough points
    if len(trimmed) >= d + 1:
        try:
            hull_indices = sp.ConvexHull(trimmed).vertices
            trimmed = trimmed[hull_indices]
        except sp.QhullError:
            # Lower-dimensional point sets are valid; keep the original
            # points when this optional reduction cannot be performed.
            pass

    return trimmed


# --------------------------------------------------
# Support function of a polytope
# --------------------------------------------------

def support_function(
        P: Points,
        obj: np.ndarray,
) -> tuple[float, np.ndarray] | None:
    """
    Evaluate the support function of ``conv(P)`` at direction ``obj``.

    Args:
        P:        ``(n, d)`` array of points whose convex hull is the
                  feasible region.
        obj:      ``(d,)`` direction vector.

    Returns:
        ``(h, v_max)`` where ``h = obj @ v_max`` is the support-function
        value and ``v_max`` is a maximizing point from ``P``; or ``None``
        if ``P`` is empty. Ties select the first maximizing point.

    Raises:
        ValueError: If a non-empty *P* is not a 2D array of shape ``(n, d)``.
    """
    P = np.asarray(P, dtype=float)
    if _is_empty_point_set(P):
        return None
    if P.ndim != 2:
        raise ValueError("Point set must be a 2D array of shape (n, d)")

    # A linear objective attains its maximum over conv(P) at a point of P.
    scores = P @ obj
    i = int(np.argmax(scores))
    return float(scores[i]), P[i].copy()
