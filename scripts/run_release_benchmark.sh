#!/usr/bin/env bash
#
# Run the roxie v1 release benchmark: every agent on every dm_control task, on
# both physics implementations.
#
#   THE GRID
#
#   25 tasks  x  7 agents  x  2 cells  =  350 runs
#
#     tasks   the dm_control suite, as mujoco_playground registers it and as
#             EnvPool wraps it. The list is roxie.environment.suites.DMC_TASKS
#             (one place, with a test on it), not a copy in this file.
#     agents  ddpg td3 td4 d4pg sac mpo ppo, at matched hyperparameters.
#     cells   warp_gpu     GPU physics (mujoco_warp) + GPU learner
#             envpool_cpu  CPU physics (native MuJoCo pool) + CPU learner
#                          <- fully GPU-free, and a genuinely INDEPENDENT
#                             implementation of the same 25 tasks
#
#   Two more cells are configured and out of the default grid, because the
#   GPU-vs-CPU claim is what needs all 25 tasks behind it and the physics-backend
#   comparison does not:
#
#     mjx_gpu   GPU physics (MJX) + GPU learner   — Warp vs MJX on the same card
#     mjx_cpu   CPU physics (MJX) + CPU learner   — the same JAX program, no card
#
#   --cells is an ESCAPE HATCH as well as a filter: naming one runs it even
#   though the default grid omits it.
#
#     scripts/run_release_benchmark.sh --cells mjx_gpu --tasks CheetahRun,WalkerWalk
#
#   WHAT IS HELD FIXED
#
#   Every run gets the SAME step budget and the SAME matched agent
#   hyperparameters — see experiments/README.md. Nothing here is tuned per task:
#   this script varies the launchable (the agent), `release.task` and the backend
#   group, and never passes a tuning override, so each run's condition is fully
#   recorded by its own yaml plus the resolved config Hydra writes to
#   .hydra/config.yaml in the run dir.
#
#   WHERE THE RESULTS GO
#
#   ONE WANDB PROJECT PER TASK — `roxie-<Task>` — with the agent and the cell as
#   the run identity inside it, and `group` naming the grid. A project is the
#   unit wandb gives a workspace and cross-run charts to, so this makes its
#   default view exactly the comparison that means something: same task, same
#   budget, 7 agents x 2 cells.
#
#   USAGE
#
#     scripts/run_release_benchmark.sh --dry-run        # print grid + estimates
#     scripts/run_release_benchmark.sh --smoke          # tiny budgets, validate
#     scripts/run_release_benchmark.sh                  # the real thing
#     scripts/run_release_benchmark.sh --tasks CheetahRun,WalkerWalk
#     scripts/run_release_benchmark.sh --agents td3,ppo --cells warp_gpu
#
#   Runs execute STRICTLY SEQUENTIALLY: every cell wants either the whole card
#   or every core, so overlapping two of them measures contention rather than
#   the backend. AT THE FULL 50M-STEP BUDGET THE WHOLE GRID IS WEEKS, NOT A
#   WEEKEND — start with --dry-run, which prints the projection, and consider
#   running it a task-batch at a time with --tasks.
#
#   The script is resumable at RUN granularity: each completed run is appended
#   to outputs/release_v1/manifest.tsv and skipped on a later invocation unless
#   --force is passed. A failed run is recorded too, and retried next time; on
#   failure the script prints the `resume=` command for the last checkpoint it
#   wrote (checkpoints land every trainer.save_steps env steps).
#
#   AFTERWARDS
#
#     uv run python roxie/report.py                     # W&B report
#     uv run python roxie/plot.py --grid --path outputs/release_v1 --output grid.pdf
#     uv run python scripts/export_release_weights.py   # publishable policies
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

# ---------------------------------------------------------------- grid ------

ALL_AGENTS="ddpg td3 td4 d4pg sac mpo ppo"
ALL_CELLS="warp_gpu envpool_cpu mjx_gpu mjx_cpu"
# Cells the DEFAULT grid runs. A cell above but not here is supported and
# configured, just not worth 175 runs unasked; --cells runs it anyway.
DEFAULT_CELLS="${DEFAULT_CELLS:-warp_gpu envpool_cpu}"

# The task list comes from the package, so the grid and the ${envpool_task:...}
# mapping cannot disagree about what the suite IS.
ALL_TASKS="${ALL_TASKS:-$(uv run python -c \
    'from roxie.environment.suites import DMC_TASKS; print(" ".join(DMC_TASKS))' \
    2>/dev/null)}"
if [[ -z "$ALL_TASKS" ]]; then
    echo "could not import roxie.environment.suites — is the venv set up? (uv sync)" >&2
    exit 1
fi

# Step budget. Identical across every agent, task and cell — that equality is
# what makes score-vs-env-steps a comparison. Anything measured in env steps
# inside the configs is tied to this number (the noise anneal is 40% of it), so
# read experiments/dmc/bench/dmc.yaml before overriding it for a real result.
STEPS="${STEPS:-50000000}"
# --smoke budgets: enough for two epochs and a first eval, i.e. enough to prove
# the config composes, the env builds, the agent compiles and the trainer logs.
SMOKE_STEPS=100000
SMOKE_EPOCH=25000

# Throughput priors used ONLY for the --dry-run projection (steps/s, measured on
# a 12-core 7900X + RTX 5080 WITH A LEARNER ATTACHED — these are not physics-only
# ceilings). warp_gpu and envpool_cpu are measured on CheetahRun/TD3 at 256 envs
# (6.6k and 13.7k); the two MJX cells are carried over from the walker grid. A
# single number for 25 tasks is a projection aid and nothing more — the manifest
# records what each run actually achieved.
#
# Note the ordering, which is the benchmark's least intuitive fact: the GPU cell
# is the SLOWER one on this suite. dm_control bodies are tiny, so at 256 envs the
# card is nowhere near saturated and Warp's per-step dispatch dominates, while
# EnvPool's native MuJoCo runs 256 cheap envs across 24 threads very happily. The
# GPU cell wins on env count, not on env size — which is why parallel_envs is one
# of the things held fixed.
sps_prior() {
    case "$1" in
        warp_gpu)    echo  6600 ;;
        envpool_cpu) echo 13700 ;;
        mjx_gpu)     echo  8500 ;;
        mjx_cpu)     echo  5000 ;;
        *)           echo     0 ;;
    esac
}

OUT_ROOT="outputs/release_v1"
MANIFEST="$OUT_ROOT/manifest.tsv"
LOG_DIR="$OUT_ROOT/logs"

# ------------------------------------------------------------ arguments -----

SMOKE=0; DRY_RUN=0; FORCE=0; OFFLINE=0; ALLOW_BUSY_GPU=0
TASK_FILTER=""; AGENT_FILTER=""; CELL_FILTER=""

# The whole header block, whatever length it is. The fixed line range this used
# to carry silently truncated --help every time the header grew.
usage() {
    awk 'NR == 1 { next }
         /^#/    { sub(/^#+ ?/, ""); print; next }
                 { exit }' "$0"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)          SMOKE=1 ;;
        --dry-run|-n)     DRY_RUN=1 ;;
        --force)          FORCE=1 ;;
        --offline)        OFFLINE=1 ;;
        --allow-busy-gpu) ALLOW_BUSY_GPU=1 ;;
        --tasks)          TASK_FILTER="${2//,/ }"; shift ;;
        --agents)         AGENT_FILTER="${2//,/ }"; shift ;;
        --cells)          CELL_FILTER="${2//,/ }"; shift ;;
        --steps)          STEPS="$2"; shift ;;
        -h|--help)        usage ;;
        *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
    esac
    shift
done

if [[ $OFFLINE -eq 1 ]]; then export WANDB_MODE=offline; fi

# ------------------------------------------------------------- helpers ------

c_bold=$'\033[1m'; c_red=$'\033[31m'; c_yellow=$'\033[33m'
c_green=$'\033[32m'; c_off=$'\033[0m'
say()  { printf '%s%s%s\n' "$c_bold" "$*" "$c_off"; }
warn() { printf '%s! %s%s\n' "$c_yellow" "$*" "$c_off" >&2; }
err()  { printf '%sx %s%s\n' "$c_red" "$*" "$c_off" >&2; }

in_list() { local needle="$1"; shift; for x in $*; do [[ "$x" == "$needle" ]] && return 0; done; return 1; }

hms() {  # seconds -> 1h23m, or 4d05h once it stops fitting in hours
    local s=$1
    if   (( s >= 86400 )); then printf '%dd%02dh' $((s/86400)) $(((s%86400)/3600))
    elif (( s >= 3600  )); then printf '%dh%02dm' $((s/3600)) $(((s%3600)/60))
    elif (( s >= 60    )); then printf '%dm%02ds' $((s/60)) $((s%60))
    else                        printf '%ds' "$s"; fi
}

# A config group living in a subdirectory of the search path needs its FULL
# path and its package — `dmc/backend@backend=envpool_cpu`. The bare
# `backend=envpool_cpu` that reads naturally is rejected ("Key 'backend' is not
# in struct"), and `+backend=` appends a second entry instead of replacing.
backend_override() { echo "dmc/backend@backend=$1"; }

selected_cells() {
    local cells=""
    for cell in $ALL_CELLS; do
        if [[ -n "$CELL_FILTER" ]]; then
            in_list "$cell" "$CELL_FILTER" || continue
        else
            in_list "$cell" "$DEFAULT_CELLS" || continue
        fi
        cells="$cells $cell"
    done
    echo "$cells"
}

# A run counts as done only if it finished AT THE SAME STEP BUDGET. Matching on
# (task, cell, agent) alone would let a --smoke run — which is recorded `ok`
# with a tiny budget — suppress the real one, quietly leaving a truncated arm in
# the middle of the release grid.
already_done() {
    [[ $FORCE -eq 1 ]] && return 1
    [[ -f "$MANIFEST" ]] || return 1
    grep -qP "^ok\t$1\t$2\t$3\t$4\t" "$MANIFEST"
}

# ------------------------------------------------------------ preflight -----

preflight() {
    local need_gpu=0 need_warp=0 status=0
    for cell in $(selected_cells); do
        [[ "$cell" == *gpu* ]] && need_gpu=1
        [[ "$cell" == warp_gpu ]] && need_warp=1
    done

    # ORDER MATTERS: the card-occupancy check runs BEFORE the warp check,
    # because a busy card makes `check_warp.py` fail with a Warp CUDA OOM
    # inside graph creation — which reads like a broken warp install and is
    # not one. Diagnose the real cause first, and don't bother running the
    # warp probe at all when the card is known to be full.
    local gpu_busy=0
    if [[ $need_gpu -eq 1 ]]; then
        if ! command -v nvidia-smi >/dev/null 2>&1; then
            err "no nvidia-smi — the GPU cells will fail. Run with --cells envpool_cpu."
            status=1
        else
            local busy
            busy=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null \
                   | awk '{s+=$1} END {print s+0}')
            if (( busy > 2000 )); then
                gpu_busy=1
                if [[ $ALLOW_BUSY_GPU -eq 1 ]]; then
                    warn "GPU already has ${busy} MiB in use by another process — continuing anyway."
                else
                    err "GPU already has ${busy} MiB in use by another process:"
                    nvidia-smi --query-compute-apps=pid,used_memory,process_name \
                               --format=csv,noheader >&2
                    err "  A warp run started now will OOM in CUDA graph creation."
                    err "  Wait for it to finish, or pass --allow-busy-gpu."
                    status=1
                fi
            else
                say "  gpu         ok (${busy} MiB in use)"
            fi
        fi
    fi

    if [[ $need_warp -eq 1 ]]; then
        if [[ $gpu_busy -eq 1 ]]; then
            warn "warp        not probed (the card is busy, so the probe would OOM and tell us nothing)"
        elif uv run python scripts/check_warp.py >/dev/null 2>&1; then
            say "  warp        ok"
        else
            err "warp is not usable — the warp_gpu cell will fail."
            err "  uv sync --group cuda   (and see the warp-lang pin in pyproject.toml)"
            err "  reproduce with: uv run python scripts/check_warp.py"
            status=1
        fi
    fi

    # wandb: the benchmark configs set relogin:false (an interactive re-login
    # prompt would deadlock an unattended grid), so credentials have to already
    # be in place or the very first run dies after building its env.
    if [[ "${WANDB_MODE:-online}" != "offline" && "${WANDB_MODE:-online}" != "disabled" ]]; then
        if [[ -n "${WANDB_API_KEY:-}" ]] || grep -q "api.wandb.ai" "${NETRC:-$HOME/.netrc}" 2>/dev/null; then
            # Resolve and PRINT the account, don't just assert a credential
            # exists. With relogin:false the cached ~/.netrc key is used
            # silently, so on a machine with more than one wandb account the
            # grid can create 25 projects under the wrong org without a single
            # prompt. Naming the entity up front is the only warning you get.
            local who
            who=$(uv run python -c "
import wandb
try:
    api = wandb.Api()
    print(f'{api.default_entity}')
except Exception as e:
    print(f'UNRESOLVED ({type(e).__name__})')
" 2>/dev/null | tail -1)
            local src="~/.netrc"
            [[ -n "${WANDB_API_KEY:-}" ]] && src="WANDB_API_KEY"
            if [[ "$who" == UNRESOLVED* || -z "$who" ]]; then
                err "wandb credential ($src) did not resolve to an account: $who"
                err "  uv run wandb login --relogin     # or export WANDB_API_KEY"
                status=1
            else
                say "  wandb       ok — creating roxie-<Task> projects under '${who}' (from $src)"
                warn "not the account you want? 'uv run wandb login --relogin', or"
                warn "  export WANDB_API_KEY=<key>, or pass --offline to log locally."
            fi
        else
            err "wandb is enabled in the benchmark configs but no credential was found."
            err "  run 'uv run wandb login' once, or export WANDB_API_KEY, or pass --offline."
            status=1
        fi
    else
        say "  wandb       ${WANDB_MODE} (no credential needed)"
    fi
    return $status
}

# ------------------------------------------------------------- the grid -----

mkdir -p "$LOG_DIR"
MANIFEST_HEADER=$'status\ttask\tcell\tagent\tsteps\tseconds\tsps\trun_dir'

# The ledger is append-only and read back by BOTH `already_done` (which greps
# positionally) and scripts/export_release_weights.py (which reads it as a TSV
# with named columns). A manifest written under an older schema — the v1 grid's
# second column was `suite`, holding `walker_walk` / `mocap_cmu_006_13` — would
# take new rows silently and then be half-parsed by both readers. Refuse rather
# than append, and say exactly what to do about it.
if [[ -f "$MANIFEST" ]]; then
    existing=$(head -n 1 "$MANIFEST")
    if [[ "$existing" != "$MANIFEST_HEADER" ]]; then
        err "$MANIFEST was written under a different schema:"
        err "    found:    $existing"
        err "    expected: $MANIFEST_HEADER"
        err "  Move it aside — the runs it lists are from a superseded grid:"
        err "    mv $MANIFEST $MANIFEST.superseded"
        exit 2
    fi
else
    printf '%s\n' "$MANIFEST_HEADER" > "$MANIFEST"
fi

steps=$STEPS
[[ $SMOKE -eq 1 ]] && steps=$SMOKE_STEPS

declare -a QUEUE
total_estimate=0
declare -A cell_runs
for task in $ALL_TASKS; do
    [[ -n "$TASK_FILTER" ]] && ! in_list "$task" "$TASK_FILTER" && continue
    for agent in $ALL_AGENTS; do
        [[ -n "$AGENT_FILTER" ]] && ! in_list "$agent" "$AGENT_FILTER" && continue
        for cell in $(selected_cells); do
            QUEUE+=("$task|$agent|$cell|$steps")
            cell_runs[$cell]=$(( ${cell_runs[$cell]:-0} + 1 ))
            sps=$(sps_prior "$cell")
            (( sps > 0 )) && total_estimate=$(( total_estimate + steps / sps ))
        done
    done
done

if [[ ${#QUEUE[@]} -eq 0 ]]; then err "the filters selected no runs."; exit 2; fi

say ""
say "roxie release benchmark — ${#QUEUE[@]} runs$( [[ $SMOKE -eq 1 ]] && echo ' (SMOKE: tiny budgets)' )"
say "preflight"
preflight || { [[ $DRY_RUN -eq 1 ]] || exit 1; }
say ""

# The full grid is 350 rows; printing them all buries the number that matters.
# Per-cell totals, then the rows only when the selection is small enough to read.
if [[ ${#QUEUE[@]} -le 40 ]]; then
    printf '%-22s %-13s %-6s %14s %12s\n' TASK CELL AGENT STEPS "EST. TIME"
    for item in "${QUEUE[@]}"; do
        IFS='|' read -r task agent cell run_steps <<< "$item"
        sps=$(sps_prior "$cell")
        est=$( (( sps > 0 )) && hms $(( run_steps / sps )) || echo "?" )
        mark=""
        already_done "$task" "$cell" "$agent" "$run_steps" && mark=" (done, skipping)"
        printf '%-22s %-13s %-6s %14s %12s%s\n' "$task" "$cell" "$agent" "$run_steps" "$est" "$mark"
    done
else
    printf '%-13s %8s %14s %14s\n' CELL RUNS "STEPS/RUN" "EST. TOTAL"
    for cell in $(selected_cells); do
        n=${cell_runs[$cell]:-0}; sps=$(sps_prior "$cell")
        printf '%-13s %8s %14s %14s\n' "$cell" "$n" "$steps" \
               "$( (( sps > 0 )) && hms $(( n * steps / sps )) || echo '?' )"
    done
    say ""
    say "  tasks:  $(echo $ALL_TASKS | wc -w) known$( [[ -n "$TASK_FILTER" ]] && echo ", filtered to: $TASK_FILTER" )"
    say "  agents: $ALL_AGENTS"
fi
say ""
say "estimated compute: $(hms $total_estimate) (+ ~1-2 min compile per run)"
if (( total_estimate > 7 * 86400 )) && [[ $SMOKE -eq 0 ]]; then
    warn "that is over a week of sequential compute. --tasks runs it in batches,"
    warn "and the estimate above is from a prior table — re-run --dry-run after"
    warn "the first batch and it will still use the prior, but manifest.tsv will"
    warn "have the real per-cell sps to check it against."
fi

if [[ $DRY_RUN -eq 1 ]]; then say "dry run — nothing launched."; exit 0; fi

# ------------------------------------------------------------ execution -----

ok_count=0; fail_count=0; skip_count=0
grid_t0=$SECONDS

for item in "${QUEUE[@]}"; do
    IFS='|' read -r task agent cell run_steps <<< "$item"

    if already_done "$task" "$cell" "$agent" "$run_steps"; then
        skip_count=$((skip_count + 1))
        echo "-- skip $task/$cell/$agent (already in the manifest; --force to redo)"
        continue
    fi

    stamp=$(date +%Y-%m-%d_%H-%M-%S)
    run_dir="$OUT_ROOT/$task/$cell/$agent/$stamp"
    log="$LOG_DIR/$task.$cell.$agent.log"

    cmd=(uv run python roxie/train.py
         --config-name "dmc/bench_$agent"
         "release.task=$task"
         "$(backend_override "$cell")"
         "trainer.steps=$run_steps"
         "hydra.run.dir=$run_dir")
    if [[ $SMOKE -eq 1 ]]; then
        cmd+=("trainer.epoch_steps=$SMOKE_EPOCH")
        # Keep smoke runs out of the per-task projects' history.
        cmd+=("logging.wandb.enabled=false")
    fi

    say ""
    say "== $task / $cell / $agent   ($run_steps steps)"
    echo "   ${cmd[*]}"
    echo "   log: $log"

    t0=$SECONDS
    "${cmd[@]}" > "$log" 2>&1
    rc=$?
    elapsed=$(( SECONDS - t0 ))

    # Actual throughput, read back from the run's own CSV rather than computed
    # from wall-clock, so compile time is excluded the same way the trainer
    # excludes it.
    sps_actual=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="sys/sps") c=i; next} c&&$c!="None"{v=$c} END{if(v!="") printf "%.0f", v}' \
                 "$run_dir/log.csv" 2>/dev/null)
    [[ -z "$sps_actual" ]] && sps_actual="-"

    if [[ $rc -eq 0 ]]; then
        ok_count=$((ok_count + 1))
        printf '%s%s ok%s  %s  %s sps\n' "$c_green" "==" "$c_off" "$(hms $elapsed)" "$sps_actual"
        printf 'ok\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
               "$task" "$cell" "$agent" "$run_steps" "$elapsed" "$sps_actual" "$run_dir" >> "$MANIFEST"
    else
        fail_count=$((fail_count + 1))
        err "FAILED (exit $rc) after $(hms $elapsed) — last lines of $log:"
        tail -n 15 "$log" >&2
        # If it got far enough to checkpoint, say so, so the retry is a resume
        # rather than a restart from zero. `resume=` takes the run dir and picks
        # the highest step_<N> under it (see roxie/utils/checkpoint.py).
        #
        # It must NOT resume into the dead run's own dir: the CSV backend opens
        # log.csv with "w" on its first row, so a second run pointed at that dir
        # truncates the curve the first leg wrote. `.resume` keeps both halves
        # on disk; the resumed leg logs TOTAL env steps (the trainer seeds its
        # counters from the checkpoint metadata), so the two concatenate into
        # one curve.
        if compgen -G "$run_dir/checkpoints/step_*" > /dev/null; then
            resume_cmd=("${cmd[@]/#hydra.run.dir=*/hydra.run.dir=$run_dir.resume}")
            warn "it checkpointed before dying — resume instead of restarting:"
            warn "  ${resume_cmd[*]} resume=$run_dir"
            warn "(the manifest records this as 'fail', so a plain re-invocation"
            warn " of this script would start the arm over from zero.)"
        fi
        printf 'fail\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
               "$task" "$cell" "$agent" "$run_steps" "$elapsed" "$sps_actual" "$run_dir" >> "$MANIFEST"
        # Deliberately NOT fatal: one agent failing one task should not cost the
        # other 349 runs of the grid. The exit code below still reports it.
    fi
done

# -------------------------------------------------------------- summary -----

say ""
say "================================================================"
say "release benchmark finished in $(hms $(( SECONDS - grid_t0 )) )"
say "  $ok_count ok, $fail_count failed, $skip_count skipped"
say "  manifest: $MANIFEST"
say ""
if [[ $fail_count -gt 0 ]]; then
    err "failed runs:"
    awk -F'\t' '$1=="fail" {printf "    %s / %s / %s   (%s)\n", $2,$3,$4,$8}' "$MANIFEST" >&2
fi
if [[ $SMOKE -eq 1 ]]; then
    say "smoke run: budgets were tiny and wandb was disabled — nothing here is a result."
    say "measured throughput (use it to sanity-check the projection above):"
    awk -F'\t' 'NR>1 && $1=="ok" {printf "    %-22s %-13s %-6s %8s sps\n", $2,$3,$4,$7}' "$MANIFEST"
else
    say "next:"
    say "  uv run python roxie/report.py                 # per-task W&B reports"
    say "  uv run python roxie/plot.py --grid --path $OUT_ROOT --output grid.pdf"
fi

exit $(( fail_count > 0 ))
