#!/usr/bin/env bash
#
# The mjx_gpu cell only: GPU physics (MJX) + GPU learner — Warp vs MJX on the
# same card. Out of the default grid in run_release_benchmark.sh; this script
# is the escape hatch made into a dedicated entry point. See that script's
# header for the full grid, usage, resumability and manifest documentation.
# All flags forward; --cells is fixed to mjx_gpu regardless of what's passed.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_release_benchmark.sh" "$@" --cells mjx_gpu
