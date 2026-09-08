#!/usr/bin/env bash
#
# The envpool_cpu cell only: CPU physics (native MuJoCo pool) + CPU learner,
# fully GPU-free. Thin wrapper around run_release_benchmark.sh — see its header
# for the full grid, usage and resumability.
#
# This cell RUNS SEVERAL PINNED RUNS AT ONCE. One CPU run peaks at ~6 logical
# cores and gets slower with more of them, so the box is sliced into 6-core jobs
# rather than handed whole to one run, and the job count is capped by memory as
# well as by cores. The runner derives that count (and, when no GPU cell shares
# the box as here, spends the reserve on CPU jobs too); override it with
# MAX_CPU_JOBS, CORES_PER_RUN or GB_PER_RUN.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CELLS=envpool_cpu exec "$SCRIPT_DIR/run_release_benchmark.sh" "$@"
