#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

# --- 1. Configuration (Idiomatic defaults) ---
: "${TASKS:=HumanoidWalk HumanoidRun HumanoidStand}"
: "${AGENTS:=ppo}"
: "${CELLS:=envpool_cpu}"
: "${FORCE:=1}"
# Sized so the CPU budget divides: this is a per-run count of LOGICAL cores and
# the slot count is (ncores - GPU reserve) / it, so 12 on a 24-thread box floors
# to a single run and idles 10 threads.
: "${CORES_PER_RUN:=10}"
: "${GB_PER_RUN:=18}"
: "${GPU_RESERVE_CORES:=2}"
: "${GPU_RESERVE_GB:=8}"

DONE_DIR="logs/.done"
mkdir -p "$DONE_DIR"

# --- 2. Resources & Constraints ---
reserve_cores=0; reserve_gb=0
if [[ " $CELLS " == *gpu* ]]; then
    reserve_cores=$GPU_RESERVE_CORES
    reserve_gb=$GPU_RESERVE_GB
fi

detect_cores() {
    if command -v nproc >/dev/null; then
        nproc
    elif [[ "$(uname -s)" == "Darwin" ]]; then
        sysctl -n hw.ncpu
    else
        getconf _NPROCESSORS_ONLN
    fi
}

# Rough equivalent of Linux MemAvailable: memory we can take without swapping.
detect_avail_gb() {
    if [[ -r /proc/meminfo ]]; then
        awk '/^MemAvailable:/ {print int($2/1048576)}' /proc/meminfo
    elif [[ "$(uname -s)" == "Darwin" ]]; then
        local pagesize
        pagesize=$(sysctl -n hw.pagesize)
        vm_stat | awk -F: -v ps="$pagesize" '
            /Pages free|Pages inactive|Pages speculative/ { gsub(/[ .]/, "", $2); pages += $2 }
            END { print int(pages * ps / 1073741824) }'
    else
        return 1
    fi
}

physical_core_groups() {
    local n seen=" " cpu group c
    n=$(detect_cores) || n=1
    for (( cpu = 0; cpu < n; cpu++ )); do
        [[ "$seen" == *" $cpu "* ]] && continue
        if [[ -r "/sys/devices/system/cpu/cpu$cpu/topology/thread_siblings_list" ]]; then
            group=$(cat "/sys/devices/system/cpu/cpu$cpu/topology/thread_siblings_list")
        else
            group=$cpu
        fi
        echo "$group"
        for c in ${group//,/ }; do seen="$seen $c "; done
    done
}

mapfile -t PHYS_GROUPS < <(physical_core_groups)
GROUP_SIZE=$(( $(grep -o ',' <<< "${PHYS_GROUPS[0]:-0}" | wc -l) + 1 ))

# The core string for job slot $2, given $1 physical-core groups per job.
job_core_string() {
    local groups_per_run=$1 slot=$2 start i ids=()
    start=$(( reserve_cores / GROUP_SIZE + slot * groups_per_run ))
    for (( i = start; i < start + groups_per_run; i++ )); do
        ids+=("${PHYS_GROUPS[i]}")
    done
    local IFS=,
    echo "${ids[*]}"
}

ncores=$(detect_cores 2>/dev/null) || ncores=""
avail_gb=$(detect_avail_gb 2>/dev/null) || avail_gb=""
[[ -n "$ncores"  ]] || { ncores=1;  echo "! could not detect core count — assuming ${ncores}c"; }
[[ -n "$avail_gb" ]] || { avail_gb=$(( ncores * GB_PER_RUN )); echo "! could not detect free memory — assuming ${avail_gb}GB"; }

by_core=$(( (ncores - reserve_cores) / CORES_PER_RUN ))
by_ram=$(( (avail_gb - reserve_gb) / GB_PER_RUN ))
: "${MAX_CPU_JOBS:=$(( by_core < by_ram ? by_core : by_ram ))}"
(( MAX_CPU_JOBS < 1 )) && MAX_CPU_JOBS=1

groups_per_run=$(( CORES_PER_RUN / GROUP_SIZE ))
(( groups_per_run < 1 )) && groups_per_run=1

PIN=1
command -v taskset >/dev/null || { PIN=0; echo "! taskset missing — CPU runs will not be pinned"; }

echo "== grid: [$TASKS] x [$AGENTS] x [$CELLS]"
echo "== plan: $MAX_CPU_JOBS concurrent CPU runs (${ncores}c, ${avail_gb}GB avail; caps: ${by_core} by core, ${by_ram} by ram)"

# --- 3. Signal Handling ---
# The trainers are not our direct children: each job is
#   script -> subshell -> uv -> python (-> envpool/vector workers)
# so `pkill -P $$` only reaps the subshell and orphans the rest onto init,
# where they keep holding cores and RAM. Walk the whole tree instead.
descendants() {
    local pid=$1 child
    for child in $(pgrep -P "$pid" 2>/dev/null); do
        # Skip the subshell this walk is running in, else we kill ourselves.
        [[ "$child" == "$BASHPID" ]] && continue
        descendants "$child"
        echo "$child"   # deepest first, so parents can't outlive their workers
    done
}

still_alive() {
    local pid
    for pid in "$@"; do kill -0 "$pid" 2>/dev/null && echo "$pid"; done
}

cleanup() {
    trap '' INT TERM HUP   # a second Ctrl-C must not re-enter mid-teardown
    echo -e "\n[!] Caught Ctrl-C! Stopping child processes..."

    # Snapshot the tree ONCE and track those PIDs by liveness from here on.
    # Re-walking it would silently lose anything that reparents to init as its
    # intermediate parent dies — exactly the processes we most need to escalate
    # against, since they are the ones that ignored SIGTERM.
    local pids
    pids=$(descendants $$)
    [[ -z "$pids" ]] && { echo "[!] Nothing left to stop."; exit 130; }

    kill -TERM $pids 2>/dev/null

    # Let trainers flush logs and checkpoints, but stop waiting once they're gone.
    local i
    for (( i = 0; i < 15; i++ )); do
        sleep 1
        [[ -z "$(still_alive $pids)" ]] && break
    done

    # Union with a fresh walk to catch workers spawned after the snapshot.
    local stubborn
    stubborn=$( { still_alive $pids; descendants $$; } | sort -u )
    if [[ -n "$stubborn" ]]; then
        echo "[!] $(wc -w <<< "$stubborn" | tr -d ' ') process(es) ignored SIGTERM — sending SIGKILL"
        kill -KILL $stubborn 2>/dev/null
    fi
    echo "[!] Teardown complete."
    exit 130
}
trap cleanup INT TERM HUP

# --- 4. Execution ---
run_job() {
    local t=$1 a=$2 c=$3 cores=${4:-}
    local log="logs/${t}_${a}_${c}.log"
    local marker="$DONE_DIR/${t}_${a}_${c}"
    
    # Build command array to avoid duplicate if/else logic
    local cmd=(uv run python roxie/train.py --config-name "dmc/bench_$a" "release.task=$t" "dmc/backend@backend=$c")
    local where="GPU"
    if [[ -n "$cores" ]]; then
        # EnvPool owns a second thread pool on top of XLA's — only this cell
        # has the key, so mjx_cpu (build_playground_env, no num_threads
        # argument) must not get it or Hydra's instantiate() errors.
        [[ "$c" == "envpool_cpu" ]] && cmd+=("env.num_threads=$CORES_PER_RUN")
        # Thread cap applies whether or not pinning is available, otherwise each
        # run's BLAS/OMP pools fan out across every core and oversubscribe.
        cmd=(env "OMP_NUM_THREADS=$CORES_PER_RUN" "${cmd[@]}")
        if (( PIN )); then
            cmd=(taskset -c "$cores" "${cmd[@]}")
            where="cores $cores"
        else
            where="${CORES_PER_RUN} threads, unpinned"
        fi
    fi

    echo "==> Starting: $t | $a | $c ($where)"
    if "${cmd[@]}" > "$log" 2>&1; then
        touch "$marker"
        echo "    [OK] $t | $a | $c"
    else
        local rc=$?
        (( rc == 137 || rc == 9 )) \
            && echo "    [KILLED] $t | $a | $c — (exit $rc) Likely OOM." \
            || echo "    [FAIL] $t | $a | $c — (exit $rc) Last 5 lines:$(tail -n 5 "$log" | sed 's/^/\n             /')"
    fi
}

# --- 5. Pre-sort Queue (Fixes GPU blocking bug) ---
cpu_jobs=()
gpu_jobs=()

for t in $TASKS; do
    for a in $AGENTS; do
        for c in $CELLS; do
            [[ $FORCE -eq 0 && -f "$DONE_DIR/${t}_${a}_${c}" ]] && { echo "--- skipping $t | $a | $c"; continue; }
            [[ "$c" == *gpu* ]] && gpu_jobs+=("$t $a $c") || cpu_jobs+=("$t $a $c")
        done
    done
done

# --- 6. Dispatch Loops ---
declare -a SLOT_PID

# 6a. The GPU queue is sequential in itself — one device — so the whole of it
# goes to the background as a single job and overlaps the CPU queue. In the
# foreground it would instead wait for 6b to DISPATCH its last job, which at
# MAX_CPU_JOBS slots means idling the GPU until all but that many CPU runs have
# finished.
gpu_pid=""
if (( ${#gpu_jobs[@]} )); then
    (
        for job in "${gpu_jobs[@]}"; do
            read -r t a c <<< "$job"
            run_job "$t" "$a" "$c" ""
        done
    ) &
    gpu_pid=$!
    echo "== dispatched ${#gpu_jobs[@]} GPU job(s) to the background (pid $gpu_pid)"
fi

# 6b. Dispatch CPU jobs asynchronously into their pinned slots.
for job in "${cpu_jobs[@]}"; do
    read -r t a c <<< "$job"

    while true; do
        for (( i=0; i<MAX_CPU_JOBS; i++ )); do
            if ! kill -0 "${SLOT_PID[i]:-}" 2>/dev/null; then
                run_job "$t" "$a" "$c" "$(job_core_string "$groups_per_run" "$i")" &
                SLOT_PID[i]=$!
                break 2 # Break out of both the 'for' and 'while' loops
            fi
        done
        # The slot PIDs by name, not a bare `wait -n`: the GPU queue is a
        # background job too, and reaping THAT here would wake this loop
        # without having freed a slot.
        wait -n "${SLOT_PID[@]}" 2>/dev/null
    done
done

wait
echo "All experiments complete!"