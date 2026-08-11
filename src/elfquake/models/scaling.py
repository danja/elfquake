"""Shared numeric standardization guard."""

from __future__ import annotations


RELATIVE_SCALE_TOLERANCE = 1e-9


def resolve_scale(scale: float, mean: float) -> float:
    """Return a usable standardization scale, collapsing near-constant columns to 1.0.

    A column holding one repeated value can still produce a scale that is tiny
    but not exactly zero, because the mean carries floating-point accumulation
    error. Dividing residuals by that scale turns rounding noise into O(1)
    standardized values, so a constant feature silently becomes a second
    intercept -- or, when the held-out partition holds a different constant, an
    astronomically large input that saturates the model. Treat any column whose
    spread is negligible relative to its own magnitude as constant.
    """
    return scale if scale > RELATIVE_SCALE_TOLERANCE * max(1.0, abs(mean)) else 1.0
