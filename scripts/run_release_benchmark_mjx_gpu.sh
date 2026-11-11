#!/usr/bin/env bash
# The mjx_gpu cell only: GPU physics (MJX) + GPU learner. Other knobs (TASKS, AGENTS, FORCE, ...)
# are env vars of run_release_benchmark.sh; see docs/benchmark.md.
#   TASKS=HumanoidWalk AGENTS=sac scripts/run_release_benchmark_mjx_gpu.sh
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CELLS=mjx_gpu exec "$SCRIPT_DIR/run_release_benchmark.sh" "$@"
