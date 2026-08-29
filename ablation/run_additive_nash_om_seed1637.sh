#!/bin/bash
# Run additive LLM+Nash and LLM+Opponent-Modeling experiments in parallel.
# This launches 3 games x 6 styles x 2 conditions = 36 tasks.
#
# Usage:
#   bash ablation/run_additive_nash_om_seed1637.sh --approved-real-measurement
#
# Optional:
#   bash ablation/run_additive_nash_om_seed1637.sh --approved-real-measurement --games 30

set -e

SEED=1637
GAMES=30
PHASE="real_measurement"
APPROVED=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --approved-real-measurement)
            APPROVED=1
            shift
            ;;
        --games)
            [[ $# -ge 2 ]] || { echo "Missing value for --games" >&2; exit 2; }
            GAMES="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ "$APPROVED" -ne 1 ]]; then
    echo "This launcher starts 36 live-API real_measurement tasks." >&2
    echo "Re-run with --approved-real-measurement after explicit approval." >&2
    exit 2
fi

CONDITIONS=("llm_nash" "llm_om")
LEDUC_ARCHETYPES=("Maniac" "LAG" "GTO" "CallingStation" "Nit" "TAG")
LIARS_ARCHETYPES=("Bluffer" "Honest" "AggressiveChallenger" "ConservativeChallenger" "Random" "GTO")
GOOFSPIEL_PERSONAS=("Greedy" "Conservative" "Sacrificial" "Random" "GTO" "Mirror")

LOG_DIR="results/additive_component_logs/seed${SEED}"
mkdir -p "$LOG_DIR"

PIDS=()
NAMES=()

launch_task() {
    local name="$1"
    local logfile="$2"
    shift 2

    echo "Launching $name -> $logfile"
    "$@" > "$logfile" 2>&1 &
    PIDS+=($!)
    NAMES+=("$name")
    sleep 1
}

echo "Additive component launcher"
echo "  seed: $SEED"
echo "  games/position: $GAMES"
echo "  phase: $PHASE"
echo "  conditions: ${CONDITIONS[*]}"
echo "  log dir: $LOG_DIR"
echo ""

for condition in "${CONDITIONS[@]}"; do
    for archetype in "${LEDUC_ARCHETYPES[@]}"; do
        launch_task \
            "leduc_${archetype}_${condition}" \
            "$LOG_DIR/leduc_${archetype}_${condition}.log" \
            python runs/additive.py \
                --game leduc \
                --conditions "$condition" \
                --run-phase "$PHASE" \
                --approved-real-measurement \
                --games "$GAMES" \
                --start-seed "$SEED" \
                --leduc-archetypes "$archetype"
    done

    for archetype in "${LIARS_ARCHETYPES[@]}"; do
        launch_task \
            "liars_dice_${archetype}_${condition}" \
            "$LOG_DIR/liars_dice_${archetype}_${condition}.log" \
            python runs/additive.py \
                --game liars_dice \
                --conditions "$condition" \
                --run-phase "$PHASE" \
                --approved-real-measurement \
                --games "$GAMES" \
                --start-seed "$SEED" \
                --liars-archetypes "$archetype"
    done

    for persona in "${GOOFSPIEL_PERSONAS[@]}"; do
        launch_task \
            "goofspiel_${persona}_${condition}" \
            "$LOG_DIR/goofspiel_${persona}_${condition}.log" \
            python runs/additive.py \
                --game goofspiel \
                --conditions "$condition" \
                --run-phase "$PHASE" \
                --approved-real-measurement \
                --games "$GAMES" \
                --start-seed "$SEED" \
                --goofspiel-personas "$persona"
    done
done

echo ""
echo "Launched ${#PIDS[@]} tasks. Waiting for completion..."
echo ""

FAILED=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    name="${NAMES[$i]}"
    if wait "$pid"; then
        echo "done: $name"
    else
        echo "failed: $name pid=$pid"
        ((FAILED++))
    fi
done

echo ""
echo "Finished additive component batch: failed=$FAILED succeeded=$(( ${#PIDS[@]} - FAILED ))"
echo "Logs: $LOG_DIR"

if [[ "$FAILED" -gt 0 ]]; then
    exit 1
fi
