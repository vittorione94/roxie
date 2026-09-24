#!/usr/bin/env bash
#
# The warp_gpu cell only: GPU physics (mujoco_warp) + GPU learner.
# Thin wrapper around run_release_benchmark.sh — see its header for the full
# grid, usage and resumability. Selects the cell through the
# runner's `CELLS` variable; every other knob (TASKS, AGENTS, FORCE, ...) is an
# environment variable too, so prefix them here as usual.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CELLS=warp_gpu exec "$SCRIPT_DIR/run_release_benchmark.sh" "$@"
