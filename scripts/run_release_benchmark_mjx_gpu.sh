#!/usr/bin/env bash
#
# The mjx_gpu cell only: GPU physics (MJX) + GPU learner — Warp vs MJX on the
# same card. Out of the default grid in run_release_benchmark.sh; this script
# is the escape hatch made into a dedicated entry point. See that script's
# header for the full grid, usage and resumability.
# Selects the cell through the runner's `CELLS` variable; every other knob
# (TASKS, AGENTS, FORCE, ...) is an environment variable too.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CELLS=mjx_gpu exec "$SCRIPT_DIR/run_release_benchmark.sh" "$@"
