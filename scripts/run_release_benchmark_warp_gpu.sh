#!/usr/bin/env bash
#
# The warp_gpu cell only: GPU physics (mujoco_warp) + GPU learner.
# Thin wrapper around run_release_benchmark.sh — see its header for the full
# grid, usage, resumability and manifest documentation. All flags forward;
# --cells is fixed to warp_gpu regardless of what's passed.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_release_benchmark.sh" "$@" --cells warp_gpu
