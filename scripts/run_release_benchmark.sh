#!/usr/bin/env bash
# Run the roxie v1 release benchmark grid.
# Usage: ./run_release_benchmark.sh [OPTIONS]
# Options: --smoke, --dry-run, --force, --offline, --allow-busy-gpu,
#          --tasks <list>, --agents <list>, --cells <list>, --steps <N>
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

# --- Configuration ---
ALL_AGENTS="ddpg td3 td4 d4pg sac mpo ppo"
ALL_CELLS="warp_gpu envpool_cpu mjx_gpu mjx_cpu"
DEFAULT_CELLS="${DEFAULT_CELLS:-warp_gpu envpool_cpu}"
ALL_TASKS="${ALL_TASKS:-$(uv run python -c 'from roxie.environment.suites import DMC_TASKS; print(" ".join(DMC_TASKS))' 2>/dev/null)}"
[[ -z "$ALL_TASKS" ]] && { echo "Failed to load tasks. Ensure uv sync is run." >&2; exit 1; }

STEPS="${STEPS:-500000000}"
SMOKE_STEPS=100000
SMOKE_EPOCH=25000
OUT_ROOT="outputs/release_v1"
MANIFEST="$OUT_ROOT/manifest.tsv"
LOG_DIR="$OUT_ROOT/logs"

# --- CLI Args ---
SMOKE=0; DRY_RUN=0; FORCE=0; OFFLINE=0; ALLOW_BUSY_GPU=0
TASK_FILTER=""; AGENT_FILTER=""; CELL_FILTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)          SMOKE=1 ;;
        --dry-run|-n)     DRY_RUN=1 ;;
        --force)          FORCE=1 ;;
        --offline)        OFFLINE=1; export WANDB_MODE=offline ;;
        --allow-busy-gpu) ALLOW_BUSY_GPU=1 ;;
        --tasks)          TASK_FILTER="${2//,/ }"; shift ;;
        --agents)         AGENT_FILTER="${2//,/ }"; shift ;;
        --cells)          CELL_FILTER="${2//,/ }"; shift ;;
        --steps)          STEPS="$2"; shift ;;
        -h|--help)        grep '^#' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

# --- Helpers ---
say()  { echo -e "\033[1m$*\033[0m"; }
warn() { echo -e "\033[33m! $*\033[0m" >&2; }
err()  { echo -e "\033[31mx $*\033[0m" >&2; }
in_list() { local n="$1"; shift; for x in "$@"; do [[ "$x" == "$n" ]] && return 0; done; return 1; }

hms() {
    local s=$1
    if (( s >= 86400 )); then printf '%dd%02dh' $((s/86400)) $(((s%86400)/3600))
    elif (( s >= 3600 )); then printf '%dh%02dm' $((s/3600)) $(((s%3600)/60))
    elif (( s >= 60 )); then printf '%dm%02ds' $((s/60)) $((s%60))
    else printf '%ds' "$s"; fi
}

sps_prior() {
    case "$1" in
        warp_gpu) echo 6600 ;; envpool_cpu) echo 13700 ;;
        mjx_gpu) echo 8500 ;; mjx_cpu) echo 5000 ;; *) echo 0 ;;
    esac
}

already_done() {
    [[ $FORCE -eq 1 ]] && return 1
    [[ -f "$MANIFEST" ]] && grep -q "^ok\t$1\t$2\t$3\t$4\t" "$MANIFEST"
}

get_cells() {
    for c in $ALL_CELLS; do
        if [[ -n "$CELL_FILTER" ]]; then in_list "$c" $CELL_FILTER && echo "$c"
        else in_list "$c" $DEFAULT_CELLS && echo "$c"; fi
    done
}

# --- Preflight ---
preflight() {
    local needs_gpu=0
    [[ "$(get_cells)" == *gpu* ]] && needs_gpu=1

    if [[ $needs_gpu -eq 1 ]]; then
        command -v nvidia-smi >/dev/null || { err "nvidia-smi missing."; return 1; }
        local busy=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
        if (( busy > 2000 )); then
            [[ $ALLOW_BUSY_GPU -eq 1 ]] || { err "GPU busy (${busy}MB). Use --allow-busy-gpu."; return 1; }
        fi
        uv run python scripts/check_warp.py >/dev/null 2>&1 || { err "Warp check failed."; return 1; }
    fi

    if [[ "${WANDB_MODE:-online}" != "offline" && "${WANDB_MODE:-online}" != "disabled" ]]; then
        [[ -n "${WANDB_API_KEY:-}" ]] || grep -q "api.wandb.ai" "${NETRC:-$HOME/.netrc}" 2>/dev/null || \
            { err "Wandb not logged in. Use --offline or set WANDB_API_KEY."; return 1; }
    fi
    return 0
}

# --- Setup Queue ---
mkdir -p "$LOG_DIR"
MANIFEST_HDR=$'status\ttask\tcell\tagent\tsteps\tseconds\tsps\trun_dir'
[[ -f "$MANIFEST" ]] && [[ "$(head -n 1 "$MANIFEST")" != "$MANIFEST_HDR" ]] && \
    { err "Manifest schema mismatch. Move $MANIFEST aside."; exit 2; }
[[ -f "$MANIFEST" ]] || echo "$MANIFEST_HDR" > "$MANIFEST"

steps=$([[ $SMOKE -eq 1 ]] && echo $SMOKE_STEPS || echo $STEPS)
QUEUE=(); total_est=0

for task in $ALL_TASKS; do
    [[ -n "$TASK_FILTER" ]] && ! in_list "$task" $TASK_FILTER && continue
    for agent in $ALL_AGENTS; do
        [[ -n "$AGENT_FILTER" ]] && ! in_list "$agent" $AGENT_FILTER && continue
        for cell in $(get_cells); do
            QUEUE+=("$task|$agent|$cell|$steps")
            sps=$(sps_prior "$cell")
            (( sps > 0 )) && total_est=$(( total_est + steps / sps ))
        done
    done
done

[[ ${#QUEUE[@]} -eq 0 ]] && { err "No runs selected by filters."; exit 2; }

say "\nroxie benchmark: ${#QUEUE[@]} runs | Est compute: $(hms $total_est)\nPreflight checks..."
preflight || [[ $DRY_RUN -eq 1 ]] || exit 1
[[ $DRY_RUN -eq 1 ]] && { say "Dry run complete."; exit 0; }

# --- Execution ---
ok_count=0; fail_count=0; skip_count=0; INTERRUPTED=0
trap 'INTERRUPTED=1' INT TERM

for item in "${QUEUE[@]}"; do
    IFS='|' read -r task agent cell run_steps <<< "$item"
    if already_done "$task" "$cell" "$agent" "$run_steps"; then
        echo "-- skipping $task/$cell/$agent (already in manifest)"
        ((skip_count++))
        continue
    fi

    stamp=$(date +%Y-%m-%d_%H-%M-%S)
    run_dir="$OUT_ROOT/$task/$cell/$agent/$stamp"
    log="$LOG_DIR/$task.$cell.$agent.log"

    cmd=(uv run python roxie/train.py --config-name "dmc/bench_$agent" "release.task=$task" "dmc/backend@backend=$cell" "trainer.steps=$run_steps" "hydra.run.dir=$run_dir")
    [[ $SMOKE -eq 1 ]] && cmd+=("trainer.epoch_steps=$SMOKE_EPOCH" "logging.wandb.enabled=false")

    say "\n== $task / $cell / $agent ($run_steps steps) -> log: $log"
    
    t0=$SECONDS
    "${cmd[@]}" > "$log" 2>&1
    rc=$?
    elapsed=$(( SECONDS - t0 ))

    sps_actual=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="sys/sps") c=i; next} c&&$c!="None"{v=$c} END{if(v!="") printf "%.0f", v}' "$run_dir/log.csv" 2>/dev/null)
    [[ -z "$sps_actual" ]] && sps_actual="-"

    if (( INTERRUPTED == 1 || rc == 130 || rc == 143 )); then
        err "Interrupted! Aborting the grid."
        [[ -d "$run_dir/checkpoints" ]] && warn "Checkpoint found! Resume via:\n  ${cmd[*]/#hydra.run.dir=*/hydra.run.dir=$run_dir.resume} resume=$run_dir"
        exit 130
    elif [[ $rc -eq 0 ]]; then
        ((ok_count++))
        echo -e "\033[32m== ok\033[0m $(hms $elapsed) | ${sps_actual} sps"
        printf "ok\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$task" "$cell" "$agent" "$run_steps" "$elapsed" "$sps_actual" "$run_dir" >> "$MANIFEST"
    else
        ((fail_count++))
        err "Failed (exit $rc)! Last 10 lines of $log:"
        tail -n 10 "$log" >&2
        [[ -d "$run_dir/checkpoints" ]] && warn "Checkpoint found! Resume via:\n  ${cmd[*]/#hydra.run.dir=*/hydra.run.dir=$run_dir.resume} resume=$run_dir"
        printf "fail\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$task" "$cell" "$agent" "$run_steps" "$elapsed" "$sps_actual" "$run_dir" >> "$MANIFEST"
    fi
done

say "\nDone: $ok_count ok, $fail_count failed, $skip_count skipped."
[[ $fail_count -gt 0 ]] && exit 1 || exit 0