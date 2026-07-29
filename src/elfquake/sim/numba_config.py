"""Shared Numba configuration for normal runs and protected services."""

from __future__ import annotations

import os

try:
    from numba import njit as _numba_njit
except ImportError as error:  # pragma: no cover - depends on optional runtime setup.
    raise RuntimeError("numba is required for sandpile simulation; activate the project venv") from error


CACHE_ENABLED = os.environ.get("ELFQUAKE_NUMBA_CACHE", "1").lower() not in {"0", "false", "no"}


def njit(*args, **kwargs):
    """Apply njit while allowing protected services to disable disk caching."""
    if not CACHE_ENABLED:
        kwargs["cache"] = False
    return _numba_njit(*args, **kwargs)
