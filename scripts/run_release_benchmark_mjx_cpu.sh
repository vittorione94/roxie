#!/usr/bin/env bash
#
# The mjx_cpu cell only: CPU physics (MJX) + CPU learner — the same JAX
# program, no card. Out of the default grid in run_release_benchmark.sh; this
# script is the escape hatch made into a dedicated entry point. See that
# script's header for the full grid, usage and resumability. Selects the cell
# through the runner's `CELLS` variable; every other knob (TASKS, AGENTS,
# FORCE, ...) is an environment variable too.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CELLS=mjx_cpu exec "$SCRIPT_DIR/run_release_benchmark.sh" "$@"
