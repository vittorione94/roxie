#!/usr/bin/env bash
#
# Run the roxie v1 release benchmark grid.
#
# Every knob below is an environment variable with a default, so a sweep is
# launched either by editing section 1 or by prefixing the call:
#
#     CELLS=envpool_cpu ./scripts/run_release_benchmark.sh
#     TASKS=HumanoidRun AGENTS=sac ./scripts/run_release_benchmark.sh
#
# RESOURCES — why this script is not just two nested loops. The box is 24 cores,
# 61 GB of RAM and one 16 GB card, and the grid does not fit on it naively:
#
#   * A CPU-cell run peaks at ~6 logical cores and gets SLOWER when given more
#     (XLA's CPU pool spends the extra on barriers), so runs are PINNED to
#     disjoint 6-core slices instead of all being handed the whole box.
#   * Every run holds its replay buffer resident — 2.4 GB (ddpg) to 7.2 GB (ppo)
#     — so the job count is capped by MemAvailable as well as by cores. Exceeding
#     it does not thrash, it gets a run killed by the OOM killer hours in.
#   * The GPU cell runs in the foreground alongside the CPU jobs (the two cells
#     are 40+ hours each; serializing them would double the grid), so a slice of
#     cores and a slab of RAM stay reserved for it.
#
# A cell that finishes leaves a marker in logs/.done, and a marked cell is
# skipped on the next launch — so an interrupted grid is relaunched with the
# same command and picks up where it stopped. FORCE=1 ignores the markers.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

# --- 1. Configuration ---
TASKS="${TASKS:-HumanoidStand HumanoidWalk HumanoidRun}"
AGENTS="${AGENTS:-ddpg ppo sac}"
CELLS="${CELLS:-warp_gpu envpool_cpu}"
FORCE="${FORCE:-0}"

# 6 logical cores per CPU run is the measured knee, not a round number: below it
# the gradient burst is starved, above it XLA's CPU pool spends more on barriers
# than on arithmetic.
CORES_PER_RUN="${CORES_PER_RUN:-6}"
# Resident set of one CPU run, measured: 2.4 GB (ddpg) .. 7.2 GB (ppo).
GB_PER_RUN="${GB_PER_RUN:-8}"
# Held back for the GPU cell's host process (dispatch, wandb) and the desktop,
# and only when a GPU cell is actually in the grid.
GPU_RESERVE_CORES="${GPU_RESERVE_CORES:-2}"
GPU_RESERVE_GB="${GPU_RESERVE_GB:-8}"

LOG_DIR="logs"
DONE_DIR="$LOG_DIR/.done"
mkdir -p "$DONE_DIR"

# --- 2. How many CPU runs fit at once ---
has_gpu_cell() { [[ " $CELLS " == *gpu* ]]; }

reserve_cores=0; reserve_gb=0
if has_gpu_cell; then
    reserve_cores=$GPU_RESERVE_CORES
    reserve_gb=$GPU_RESERVE_GB
fi

ncores=$(nproc)
avail_gb=$(awk '/^MemAvailable:/ {print int($2/1048576)}' /proc/meminfo 2>/dev/null)
# No /proc (not Linux): let the core count decide alone rather than guess.
[[ -z "$avail_gb" ]] && avail_gb=$(( ncores * GB_PER_RUN ))

by_core=$(( (ncores - reserve_cores) / CORES_PER_RUN ))
by_ram=$(( (avail_gb - reserve_gb) / GB_PER_RUN ))
MAX_CPU_JOBS="${MAX_CPU_JOBS:-$(( by_core < by_ram ? by_core : by_ram ))}"
(( MAX_CPU_JOBS < 1 )) && MAX_CPU_JOBS=1

# Pinning needs taskset; without it every run sees all 24 cores and they fight.
PIN=1
command -v taskset >/dev/null || { PIN=0; echo "! taskset missing — CPU runs will not be pinned"; }

echo "== grid: [$TASKS] x [$AGENTS] x [$CELLS]"
echo "== box:  ${ncores} cores, ${avail_gb} GB available"
echo "== plan: $MAX_CPU_JOBS concurrent CPU runs x ${CORES_PER_RUN} cores (cap: ${by_core} by cores, ${by_ram} by RAM)"
has_gpu_cell && echo "==       + 1 GPU run, holding back ${reserve_cores} cores and ${reserve_gb} GB"

# --- 3. Strict Garbage Collection (The Ctrl-C handler) ---
cleanup() {
    echo -e "\n[!] Caught Ctrl-C! Nuking all child processes to free VRAM & CPU..."
    # 1. Ask nicely: Send SIGTERM to all children of this script ($$)
    pkill -TERM -P $$ 2>/dev/null
    sleep 3 # Give Python and wandb a moment to release memory

    # 2. No mercy: Send SIGKILL to anything that refused to die
    pkill -KILL -P $$ 2>/dev/null
    echo "[!] Cleanup complete. Exiting."
    exit 130
}
trap cleanup INT TERM HUP

# --- 4. The Execution Command ---
# `cores` is empty for the GPU cell, which is not pinned: it is GPU-bound and its
# host thread should float across whatever the CPU jobs are not using.
run_experiment() {
    local task=$1 agent=$2 cell=$3 cores="${4:-}"
    local log="$LOG_DIR/${task}_${agent}_${cell}.log"
    local marker="$DONE_DIR/${task}_${agent}_${cell}"
    local pinned=()

    if [[ -n "$cores" && $PIN -eq 1 ]]; then
        # OMP_NUM_THREADS as well as the affinity mask: MuJoCo's own OpenMP
        # regions size themselves from the machine, not from the mask.
        pinned=(taskset -c "$cores" env "OMP_NUM_THREADS=$CORES_PER_RUN")
        echo "==> Starting: $task | $agent | $cell (cores $cores, log $log)"
    else
        echo "==> Starting: $task | $agent | $cell (log $log)"
    fi

    "${pinned[@]}" uv run python roxie/train.py \
        --config-name "dmc/bench_$agent" \
        "release.task=$task" \
        "dmc/backend@backend=$cell" \
        > "$log" 2>&1
    local rc=$?

    # An OOM-killed run exits 137 and the old script still printed a tick, which
    # is how a grid could "finish" having lost half its cells. Report the code,
    # and mark done ONLY on success so a relaunch retries exactly the failures.
    if (( rc == 0 )); then
        : > "$marker"
        echo "    [OK] $task | $agent | $cell"
    elif (( rc == 137 || rc == 9 )); then
        echo "    [KILLED] $task | $agent | $cell — exit $rc, almost certainly the OOM killer."
        echo "             Check: journalctl -k | grep -i 'killed process'"
    else
        echo "    [FAIL] $task | $agent | $cell — exit $rc. Last lines of $log:"
        tail -n 5 "$log" | sed 's/^/             /'
    fi
    return $rc
}

# --- 5. CPU slot bookkeeping ---
# One slot per concurrent CPU run, each owning a fixed core slice above the
# reserve. A slot is reused only once its run has exited, so two runs never
# share cores.
declare -a SLOT_PID

claim_slot() {
    local slot=-1
    while (( slot < 0 )); do
        for (( i = 0; i < MAX_CPU_JOBS; i++ )); do
            if [[ -z "${SLOT_PID[i]:-}" ]] || ! kill -0 "${SLOT_PID[i]}" 2>/dev/null; then
                slot=$i; break
            fi
        done
        # Every slot busy: block until one of them exits, then look again.
        (( slot < 0 )) && wait -n
    done
    echo "$slot"
}

# --- 6. The Loop ---
for task in $TASKS; do
    for agent in $AGENTS; do
        for cell in $CELLS; do

            if [[ $FORCE -ne 1 && -f "$DONE_DIR/${task}_${agent}_${cell}" ]]; then
                echo "--- skipping $task | $agent | $cell (already done)"
                continue
            fi

            if [[ "$cell" == *"gpu"* ]]; then
                # GPU: one at a time, in the foreground. There is one card.
                run_experiment "$task" "$agent" "$cell"
            else
                # CPU: parallel, pinned, capped by cores AND by RAM.
                slot=$(claim_slot)
                lo=$(( reserve_cores + slot * CORES_PER_RUN ))
                hi=$(( lo + CORES_PER_RUN - 1 ))
                run_experiment "$task" "$agent" "$cell" "$lo-$hi" &
                SLOT_PID[slot]=$!
            fi

        done
    done
done

echo "All jobs dispatched. Waiting for remaining background jobs to finish..."
wait
echo "All experiments complete! ($(ls -1 "$DONE_DIR" | wc -l) cells marked done in $DONE_DIR)"
