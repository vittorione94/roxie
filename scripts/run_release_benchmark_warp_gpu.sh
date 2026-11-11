#!/usr/bin/env bash
# The warp_gpu cell only: GPU physics (mujoco_warp) + GPU learner. Other knobs (TASKS, AGENTS, FORCE, ...)
# are env vars of run_release_benchmark.sh; see docs/benchmark.md.
#   TASKS=HumanoidWalk AGENTS=sac scripts/run_release_benchmark_warp_gpu.sh
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CELLS=warp_gpu exec "$SCRIPT_DIR/run_release_benchmark.sh" "$@"
