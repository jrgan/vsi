"""Small scalar-validation helpers used across the VSI package."""
from __future__ import annotations

import math
import operator


def as_positive_int(value, name: str) -> int:
    """Return *value* as an integer, requiring it to be positive."""
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _as_finite_float(value, name: str, requirement: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be {requirement}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be {requirement}")
    return value


def as_nonnegative_float(value, name: str) -> float:
    """Return *value* as a finite, nonnegative float."""
    requirement = "finite and nonnegative"
    value = _as_finite_float(value, name, requirement)
    if value < 0:
        raise ValueError(f"{name} must be {requirement}")
    return value


def as_positive_float(value, name: str) -> float:
    """Return *value* as a finite, positive float."""
    requirement = "finite and positive"
    value = _as_finite_float(value, name, requirement)
    if value <= 0:
        raise ValueError(f"{name} must be {requirement}")
    return value
