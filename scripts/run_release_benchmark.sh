#!/usr/bin/env bash
#
# Run the roxie v1 release benchmark: every agent on a simple task and on a
# complex one, across the CPU/GPU cells that each task can express.
#
#   THE GRID
#
#   walker_walk (simple, mujoco_playground WalkerWalk)      7 agents x 3 cells
#     warp_gpu    GPU physics (mujoco_warp) + GPU learner
#     mjx_gpu     GPU physics (MJX)         + GPU learner
#     mjx_cpu     CPU physics (MJX)         + CPU learner   <- fully GPU-free
#
#   mocap_cmu_006_13 (complex, CMU humanoid tracking)       7 agents x 1 cell
#                                                         + subset x 1 cell
#     warp_gpu      GPU physics + GPU learner
#     envpool_gpu   CPU physics (native MuJoCo pool) + GPU learner (async on)
#     envpool_cpu   CPU physics + CPU learner  <- GPU-free, NOT IN THE DEFAULT
#                   GRID: at ~13.6k sps the 1B budget is ~20 h per run, so the
#                   fully GPU-free claim is carried by the walker suite's
#                   mjx_cpu cell and discussed rather than re-measured here.
#                   Still runnable on demand: --cells envpool_cpu (and give it
#                   a shorter --steps-mocap unless you mean to wait).
#
#   Between the two suites that covers all four physics x learner placements.
#   The mocap envpool_gpu cell runs a subset of agents (MOCAP_HYBRID_AGENTS)
#   because each mocap run is ~10-16 h; the walker suite carries the full-width
#   cross.
#
#   WHAT IS HELD FIXED
#
#   Every run of a suite gets the SAME env, the SAME step budget and the SAME
#   matched agent hyperparameters — see experiments/README.md. The only thing
#   this script varies is the launchable (the agent) and the backend group (the
#   cell). It never passes a tuning override, so each run's condition is fully
#   recorded by its own yaml plus the resolved config Hydra writes to
#   .hydra/config.yaml in the run dir.
#
#   USAGE
#
#     scripts/run_release_benchmark.sh --dry-run        # print grid + estimates
#     scripts/run_release_benchmark.sh --smoke          # tiny budgets, validate
#     scripts/run_release_benchmark.sh                  # the real thing
#     scripts/run_release_benchmark.sh --suite walker
#     scripts/run_release_benchmark.sh --agents td3,ppo --cells warp_gpu
#
#   Runs execute STRICTLY SEQUENTIALLY: every cell wants either the whole card
#   or every core, so overlapping two of them measures contention rather than
#   the backend. At the 1B-step mocap budget the full grid is ~4-5 DAYS, not an
#   overnight sweep — start with --dry-run, which prints the projection.
#
#   The script is resumable at RUN granularity: each completed run is appended
#   to outputs/release_v1/manifest.tsv and skipped on a later invocation unless
#   --force is passed. A failed run is recorded too, and retried next time.
#   A 10-hour run that dies is expensive to retry from zero, so on failure the
#   script prints the `resume=` command for the last checkpoint it wrote
#   (checkpoints land every trainer.save_steps env steps).
#
#   AFTERWARDS
#
#     uv run python roxie/report.py                     # W&B report
#     uv run python roxie/plot.py --path outputs/release_v1 --output release.pdf
#     uv run python scripts/export_release_weights.py   # publishable policies
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

# ---------------------------------------------------------------- grid ------

ALL_AGENTS="ddpg td3 td4 d4pg sac mpo ppo"
WALKER_CELLS="warp_gpu mjx_gpu mjx_cpu"
MOCAP_CELLS="warp_gpu envpool_gpu envpool_cpu"

# Cells the DEFAULT grid runs. A cell listed above but not here is supported and
# configured, just too expensive to include unasked; naming it with --cells runs
# it anyway. mocap/envpool_cpu is out for that reason — ~20 h per run at the 1B
# budget, for a GPU-free claim the walker suite's mjx_cpu cell already carries.
WALKER_DEFAULT_CELLS="${WALKER_DEFAULT_CELLS:-$WALKER_CELLS}"
MOCAP_DEFAULT_CELLS="${MOCAP_DEFAULT_CELLS:-warp_gpu envpool_gpu}"

# Which agents get the second (non-headline) mocap cell. Both are deliberate:
# PPO is the arm the envpool path was originally validated on, TD3 is the
# reference off-policy arm and exercises the replay buffer that the envpool
# path moves into system RAM.
MOCAP_HYBRID_AGENTS="${MOCAP_HYBRID_AGENTS:-td3 ppo}"

# Step budgets. Identical across every agent and every cell of a suite — that
# equality is what makes score-vs-env-steps a comparison. If you shorten one,
# shorten it for the whole suite and re-run all of its cells. Anything measured
# in env steps inside the configs is tied to these numbers too (the mocap noise
# anneal is 40% of MOCAP_STEPS) — see experiments/mocap/bench/cmu_006_13.yaml
# before overriding either of them for a real result.
WALKER_STEPS="${WALKER_STEPS:-5000000}"
MOCAP_STEPS="${MOCAP_STEPS:-1000000000}"
# --smoke budgets: enough for two epochs and a first eval, i.e. enough to prove
# the config composes, the env builds, the agent compiles and the trainer logs.
SMOKE_WALKER_STEPS=100000
SMOKE_WALKER_EPOCH=25000
SMOKE_MOCAP_STEPS=500000
SMOKE_MOCAP_EPOCH=250000

# Throughput priors used ONLY for the --dry-run projection (steps/s, measured on
# a 12-core 7900X + RTX 5080). They are estimates; the manifest records what
# each run actually achieved. The walker numbers and mocap:warp_gpu are now read
# back off the 2026-08-20 grid in manifest.tsv rather than guessed, which is why
# they are lower than the original priors — those were physics-only ceilings and
# did not carry a learner.
sps_prior() {
    case "$1:$2" in
        walker_walk:warp_gpu)        echo  6500 ;;   # measured, 7-agent grid
        walker_walk:mjx_gpu)         echo  8500 ;;   # measured, 7-agent grid
        walker_walk:mjx_cpu)         echo  5000 ;;   # measured, 7-agent grid
        mocap_cmu_006_13:warp_gpu)   echo 28000 ;;   # measured, ddpg + td3
        mocap_cmu_006_13:envpool_gpu) echo 17200 ;;
        mocap_cmu_006_13:envpool_cpu) echo 13600 ;;
        *)                           echo     0 ;;
    esac
}

OUT_ROOT="outputs/release_v1"
MANIFEST="$OUT_ROOT/manifest.tsv"
LOG_DIR="$OUT_ROOT/logs"

# ------------------------------------------------------------ arguments -----

SMOKE=0; DRY_RUN=0; FORCE=0; OFFLINE=0; ALLOW_BUSY_GPU=0
SUITES="walker mocap"
AGENT_FILTER=""; CELL_FILTER=""

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
        --suite)          SUITES="${2//,/ }"; shift ;;
        --agents)         AGENT_FILTER="${2//,/ }"; shift ;;
        --cells)          CELL_FILTER="${2//,/ }"; shift ;;
        --steps-walker)   WALKER_STEPS="$2"; shift ;;
        --steps-mocap)    MOCAP_STEPS="$2"; shift ;;
        -h|--help)        usage ;;
        *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
    esac
    shift
done

[[ "$SUITES" == "all" ]] && SUITES="walker mocap"
if [[ $OFFLINE -eq 1 ]]; then export WANDB_MODE=offline; fi

# ------------------------------------------------------------- helpers ------

c_bold=$'\033[1m'; c_red=$'\033[31m'; c_yellow=$'\033[33m'
c_green=$'\033[32m'; c_off=$'\033[0m'
say()  { printf '%s%s%s\n' "$c_bold" "$*" "$c_off"; }
warn() { printf '%s! %s%s\n' "$c_yellow" "$*" "$c_off" >&2; }
err()  { printf '%sx %s%s\n' "$c_red" "$*" "$c_off" >&2; }

in_list() { local needle="$1"; shift; for x in $*; do [[ "$x" == "$needle" ]] && return 0; done; return 1; }

hms() {  # seconds -> 1h23m
    local s=$1
    if   (( s >= 3600 )); then printf '%dh%02dm' $((s/3600)) $(((s%3600)/60))
    elif (( s >= 60   )); then printf '%dm%02ds' $((s/60)) $((s%60))
    else                       printf '%ds' "$s"; fi
}

# The Hydra override that selects a backend cell. A config group living in a
# subdirectory of the search path needs its FULL path and its package —
# `mocap/backend@backend=envpool_cpu`. The bare `backend=envpool_cpu` that
# reads naturally is rejected ("Key 'backend' is not in struct"), and a
# `+backend=` appends a second entry instead of replacing the default.
backend_override() {
    case "$1" in
        walker_walk)      echo "walker/backend@backend=$2" ;;
        mocap_cmu_006_13) echo "mocap/backend@backend=$2" ;;
    esac
}

launchable() {
    case "$1" in
        walker_walk)      echo "walker/bench_$2" ;;
        mocap_cmu_006_13) echo "mocap/bench_$2" ;;
    esac
}

# Cells for a suite, honouring --cells and (for mocap) the subset rule.
#
# --cells is an ESCAPE HATCH, not just a filter: naming a cell explicitly runs
# it for every selected agent, including the cells the default grid leaves out
# (mocap/envpool_cpu) and the agents the non-headline cells otherwise skip. That
# is what makes an excluded cell reachable without editing this file.
cells_for() {
    local suite="$1" agent="$2" all default cells=""
    case "$suite" in
        walker_walk)      all="$WALKER_CELLS"; default="$WALKER_DEFAULT_CELLS" ;;
        mocap_cmu_006_13) all="$MOCAP_CELLS";  default="$MOCAP_DEFAULT_CELLS" ;;
    esac
    for cell in $all; do
        if [[ -n "$CELL_FILTER" ]]; then
            in_list "$cell" "$CELL_FILTER" || continue
        else
            in_list "$cell" "$default" || continue
            # The non-headline mocap cell is 16 h a run; only the subset gets it.
            if [[ "$suite" == "mocap_cmu_006_13" && "$cell" != "warp_gpu" ]] \
               && ! in_list "$agent" "$MOCAP_HYBRID_AGENTS"; then
                continue
            fi
        fi
        cells="$cells $cell"
    done
    echo "$cells"
}

# A run counts as done only if it finished AT THE SAME STEP BUDGET. Matching on
# (suite, cell, agent) alone would let a --smoke run — which is recorded `ok`
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
    for suite in $SUITES; do
        for cell in $(cells_for "$( [[ $suite == walker ]] && echo walker_walk || echo mocap_cmu_006_13 )" td3); do
            [[ "$cell" == *gpu* ]] && need_gpu=1
            [[ "$cell" == warp_gpu ]] && need_warp=1
        done
    done

    # ORDER MATTERS: the card-occupancy check runs BEFORE the warp check,
    # because a busy card makes `check_warp.py` fail with a Warp CUDA OOM
    # inside graph creation — which reads like a broken warp install and is
    # not one. Diagnose the real cause first, and don't bother running the
    # warp probe at all when the card is known to be full.
    local gpu_busy=0
    if [[ $need_gpu -eq 1 ]]; then
        if ! command -v nvidia-smi >/dev/null 2>&1; then
            err "no nvidia-smi — the GPU cells will fail. Run with --cells mjx_cpu / envpool_cpu."
            status=1
        else
            # A GPU cell that starts while something else holds the card does not
            # fail cleanly: Warp OOMs inside CUDA graph creation, which reads as
            # a code bug. Check for it up front instead.
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
            err "warp is not usable — the warp_gpu cells will fail."
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
            # grid can publish 32 runs to the wrong org without a single
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
                say "  wandb       ok — publishing as '${who}' (from $src)"
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
[[ -f "$MANIFEST" ]] || printf 'status\tsuite\tcell\tagent\tsteps\tseconds\tsps\trun_dir\n' > "$MANIFEST"

declare -a QUEUE
total_estimate=0
for suite_short in $SUITES; do
    case "$suite_short" in
        walker) suite=walker_walk ;;
        mocap)  suite=mocap_cmu_006_13 ;;
        *) err "unknown suite '$suite_short' (walker|mocap)"; exit 2 ;;
    esac
    for agent in $ALL_AGENTS; do
        [[ -n "$AGENT_FILTER" ]] && ! in_list "$agent" "$AGENT_FILTER" && continue
        for cell in $(cells_for "$suite" "$agent"); do
            if [[ "$suite" == walker_walk ]]; then
                steps=$WALKER_STEPS
                [[ $SMOKE -eq 1 ]] && steps=$SMOKE_WALKER_STEPS
            else
                steps=$MOCAP_STEPS
                [[ $SMOKE -eq 1 ]] && steps=$SMOKE_MOCAP_STEPS
            fi
            QUEUE+=("$suite|$agent|$cell|$steps")
            sps=$(sps_prior "$suite" "$cell")
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
printf '%-18s %-13s %-6s %14s %12s\n' SUITE CELL AGENT STEPS "EST. TIME"
for item in "${QUEUE[@]}"; do
    IFS='|' read -r suite agent cell steps <<< "$item"
    sps=$(sps_prior "$suite" "$cell")
    est=$( (( sps > 0 )) && hms $(( steps / sps )) || echo "?" )
    mark=""
    already_done "$suite" "$cell" "$agent" "$steps" && { mark=" (done, skipping)"; }
    printf '%-18s %-13s %-6s %14s %12s%s\n' "$suite" "$cell" "$agent" "$steps" "$est" "$mark"
done
say ""
say "estimated compute: $(hms $total_estimate) (+ ~1-2 min compile per run)"

if [[ $DRY_RUN -eq 1 ]]; then say "dry run — nothing launched."; exit 0; fi

# ------------------------------------------------------------ execution -----

ok_count=0; fail_count=0; skip_count=0
grid_t0=$SECONDS

for item in "${QUEUE[@]}"; do
    IFS='|' read -r suite agent cell steps <<< "$item"

    if already_done "$suite" "$cell" "$agent" "$steps"; then
        skip_count=$((skip_count + 1))
        echo "-- skip $suite/$cell/$agent (already in the manifest; --force to redo)"
        continue
    fi

    stamp=$(date +%Y-%m-%d_%H-%M-%S)
    run_dir="$OUT_ROOT/$suite/$cell/$agent/$stamp"
    log="$LOG_DIR/$suite.$cell.$agent.log"

    cmd=(uv run python roxie/train.py
         --config-name "$(launchable "$suite" "$agent")"
         "$(backend_override "$suite" "$cell")"
         "trainer.steps=$steps"
         "hydra.run.dir=$run_dir")
    if [[ $SMOKE -eq 1 ]]; then
        if [[ "$suite" == walker_walk ]]; then
            cmd+=("trainer.epoch_steps=$SMOKE_WALKER_EPOCH")
        else
            cmd+=("trainer.epoch_steps=$SMOKE_MOCAP_EPOCH")
        fi
        # Keep smoke runs out of the release project's history.
        cmd+=("logging.wandb.enabled=false")
    fi

    say ""
    say "== $suite / $cell / $agent   ($steps steps)"
    echo "   ${cmd[*]}"
    echo "   log: $log"

    t0=$SECONDS
    "${cmd[@]}" > "$log" 2>&1
    rc=$?
    elapsed=$(( SECONDS - t0 ))

    # Actual throughput, read back from the run's own CSV rather than computed
    # from wall-clock, so compile time is excluded the same way the trainer
    # excludes it. Accepts the bare `sps` of pre-namespace runs as well as the
    # current `sys/sps`, so re-reading an older manifest's run dirs still works.
    sps_actual=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="sys/sps" || $i=="sps") c=i; next} c&&$c!="None"{v=$c} END{if(v!="") printf "%.0f", v}' \
                 "$run_dir/log.csv" 2>/dev/null)
    [[ -z "$sps_actual" ]] && sps_actual="-"

    if [[ $rc -eq 0 ]]; then
        ok_count=$((ok_count + 1))
        printf '%s%s ok%s  %s  %s sps\n' "$c_green" "==" "$c_off" "$(hms $elapsed)" "$sps_actual"
        printf 'ok\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
               "$suite" "$cell" "$agent" "$steps" "$elapsed" "$sps_actual" "$run_dir" >> "$MANIFEST"
    else
        fail_count=$((fail_count + 1))
        err "FAILED (exit $rc) after $(hms $elapsed) — last lines of $log:"
        tail -n 15 "$log" >&2
        # A mocap arm is 10-16 h; if it got far enough to checkpoint, say so, so
        # the retry is a resume rather than a restart from zero. `resume=` takes
        # the run dir and picks the highest step_<N> under it (see
        # roxie/utils/checkpoint.py), so the bare path is all that is needed.
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
               "$suite" "$cell" "$agent" "$steps" "$elapsed" "$sps_actual" "$run_dir" >> "$MANIFEST"
        # Deliberately NOT fatal: one agent failing a cell should not cost the
        # other twenty runs of an overnight grid. The exit code below still
        # reports it.
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
    say "measured throughput per cell (use it to sanity-check the projection above):"
    awk -F'\t' 'NR>1 && $1=="ok" {printf "    %-18s %-13s %-6s %8s sps\n", $2,$3,$4,$7}' "$MANIFEST"
else
    say "next:"
    say "  uv run python roxie/report.py                 # assemble the W&B report"
    say "  uv run python roxie/plot.py --path $OUT_ROOT --output release.pdf"
fi

exit $(( fail_count > 0 ))
