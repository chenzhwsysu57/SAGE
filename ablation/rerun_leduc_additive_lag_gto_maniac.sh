#!/bin/bash
# Rerun 5 Leduc additive-component runs with a different seed, all in parallel:
#   - LLM+Nash (llm_nash): LAG, GTO
#   - LLM+Opponent model (llm_om): LAG, GTO, Maniac
# Original runs used seed 1637 (results/*/additive_components/20260730_*);
# this rerun defaults to seed 42 to get an independent draw.
#
# Usage:
#   bash ablation/rerun_leduc_additive_lag_gto_maniac.sh --dry-run
#   bash ablation/rerun_leduc_additive_lag_gto_maniac.sh --approved-real-measurement
#   bash ablation/rerun_leduc_additive_lag_gto_maniac.sh --approved-real-measurement --games 30 --seed 42

set -e

SEED=42
GAMES=30
PHASE="real_measurement"
APPROVED=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --approved-real-measurement)
            APPROVED=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --games)
            [[ $# -ge 2 ]] || { echo "Missing value for --games" >&2; exit 2; }
            GAMES="$2"
            shift 2
            ;;
        --seed)
            [[ $# -ge 2 ]] || { echo "Missing value for --seed" >&2; exit 2; }
            SEED="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ "$DRY_RUN" -ne 1 && "$APPROVED" -ne 1 ]]; then
    echo "This launcher starts 5 live-API real_measurement tasks." >&2
    echo "Re-run with --approved-real-measurement after explicit approval." >&2
    exit 2
fi

# (condition, archetype) pairs to rerun.
TASKS=(
    "llm_nash|LAG"
    "llm_nash|GTO"
    "llm_om|LAG"
    "llm_om|GTO"
    "llm_om|Maniac"
)

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="results/leduc/ablation_logs/rerun_additive_5runs_seed${SEED}"
mkdir -p "$LOG_DIR"

echo "Leduc additive rerun launcher"
echo "  seed: $SEED (original 1637)"
echo "  games/position: $GAMES"
echo "  phase: $PHASE"
echo "  tasks: ${#TASKS[@]}"
echo "  log dir: $LOG_DIR"
echo ""

if [[ "$DRY_RUN" -eq 1 ]]; then
    for task in "${TASKS[@]}"; do
        condition="${task%%|*}"
        archetype="${task##*|}"
        echo "DRY-RUN: python runs/additive.py --game leduc --conditions $condition --run-phase $PHASE --approved-real-measurement --games $GAMES --start-seed $SEED --leduc-archetypes $archetype"
        echo "         log -> $LOG_DIR/${RUN_TS}_leduc_${archetype}_${condition}.log"
    done
    exit 0
fi

PIDS=()
NAMES=()
LOGS=()

launch_task() {
    local name="$1"
    local logfile="$2"
    shift 2

    echo "Launching $name -> $logfile"
    "$@" > "$logfile" 2>&1 &
    PIDS+=($!)
    NAMES+=("$name")
    LOGS+=("$logfile")
    sleep 1
}

for task in "${TASKS[@]}"; do
    condition="${task%%|*}"
    archetype="${task##*|}"
    launch_task \
        "leduc_${archetype}_${condition}" \
        "$LOG_DIR/${RUN_TS}_leduc_${archetype}_${condition}.log" \
        python runs/additive.py \
            --game leduc \
            --conditions "$condition" \
            --run-phase "$PHASE" \
            --approved-real-measurement \
            --games "$GAMES" \
            --start-seed "$SEED" \
            --leduc-archetypes "$archetype"
done

echo ""
echo "Launched ${#PIDS[@]} tasks in parallel. Waiting for completion..."
echo ""

FAILED=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    name="${NAMES[$i]}"
    logfile="${LOGS[$i]}"
    if wait "$pid"; then
        echo "Result: $name OK (log: $logfile)"
    else
        echo "Result: $name FAILED pid=$pid (log: $logfile)"
        ((FAILED++))
    fi
done

echo ""
echo "Finished: failed=$FAILED succeeded=$(( ${#PIDS[@]} - FAILED ))"
echo "Logs: $LOG_DIR"

if [[ "$FAILED" -gt 0 ]]; then
    exit 1
fi
