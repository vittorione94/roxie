#!/usr/bin/env bash
#
# The mjx_cpu cell only: CPU physics (MJX) + CPU learner — the same JAX
# program, no card. Out of the default grid in run_release_benchmark.sh; this
# script is the escape hatch made into a dedicated entry point. See that
# script's header for the full grid, usage, resumability and manifest
# documentation. All flags forward; --cells is fixed to mjx_cpu regardless of
# what's passed.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_release_benchmark.sh" "$@" --cells mjx_cpu
