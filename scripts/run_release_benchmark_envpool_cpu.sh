#!/usr/bin/env bash
#
# The envpool_cpu cell only: CPU physics (native MuJoCo pool) + CPU learner,
# fully GPU-free. Thin wrapper around run_release_benchmark.sh — see its
# header for the full grid, usage, resumability and manifest documentation.
# All flags forward; --cells is fixed to envpool_cpu regardless of what's
# passed.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_release_benchmark.sh" "$@" --cells envpool_cpu
