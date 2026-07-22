#!/usr/bin/env bash
set -euo pipefail

# The service imports the simulation command tree, but never runs simulation.
# Disabling JIT avoids Numba cache initialization under systemd.
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"

MANIFEST="${MANIFEST:-data/raw/vlf/japan/manifest.csv}"
WINDOWS="${WINDOWS:-}"

MANIFEST="$MANIFEST" ./scripts/discover-japan-vlf-manifest.sh
MANIFEST="$MANIFEST" WINDOWS="$WINDOWS" ./scripts/process-japan-vlf-manifest.sh
