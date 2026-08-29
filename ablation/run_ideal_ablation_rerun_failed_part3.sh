#!/bin/bash
# Rerun of the failed ideal_ablation runs — part 3 of 3 (9 runs, parallel).
# Run part1/part2/part3 simultaneously (8 + 8 + 9 = 25 failed runs).
#
# This part covers: liars dice full sheet (3), liars dice full swapped (6).
#
# Usage (run from repo root):
#   bash ablation/run_ideal_ablation_rerun_failed_part3.sh --dry-run
#   bash ablation/run_ideal_ablation_rerun_failed_part3.sh --approved-real-measurement [--games 30]
#
# Without --approved-real-measurement the script refuses to launch.

set -e

cd "$(dirname "$0")/.."

GAMES=30
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
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

NUM_DICE=2
NUM_FACES=3
CFR_MODEL="results/liars_dice/cfr/cfr_2x3_evalevery/model"

# label|game|opponent|seed|refresh(1=full, 0=memory)
ENTRIES=(
    "ld_full_ConservativeChallenger|liars_dice|ConservativeChallenger|42|1"
    "ld_full_Random|liars_dice|Random|42|1"
    "ld_full_GTO|liars_dice|GTO|42|1"
    "ld_full_swapped_Bluffer|liars_dice|Bluffer|1627|1"
    "ld_full_swapped_Honest|liars_dice|Honest|1627|1"
    "ld_full_swapped_AggressiveChallenger|liars_dice|AggressiveChallenger|1627|1"
    "ld_full_swapped_ConservativeChallenger|liars_dice|ConservativeChallenger|1627|1"
    "ld_full_swapped_Random|liars_dice|Random|1627|1"
    "ld_full_swapped_GTO|liars_dice|GTO|1627|1"
)

LOG_DIR="results/ablation_logs/ideal_ablation_rerun_failed_part3"

if [[ "$DRY_RUN" -eq 0 && "$APPROVED" -ne 1 ]]; then
    echo "This launcher starts 9 live-API runs (failed ideal_ablation rerun, part 3) of ${GAMES} games/position each." >&2
    echo "It calls deepseek-v4-flash for game decisions (plus memory-refresh calls when enabled) and writes artifacts under results/." >&2
    echo "Preview with --dry-run, or re-run with --approved-real-measurement after explicit approval." >&2
    exit 2
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
    mkdir -p "$LOG_DIR"
fi

echo "Ideal_ablation failed rerun — part 3 (9 runs, parallel)"
PIDS=()
NAMES=()
for entry in "${ENTRIES[@]}"; do
    IFS='|' read -r label game opp seed refresh <<< "$entry"
    extra=""
    [[ "$refresh" == "0" ]] && extra="--no-refresh-memory-bank"
    logfile="$LOG_DIR/${label}.log"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "  [dry-run] ${label} game=${game} opp=${opp} seed=${seed} ${extra} -> ${logfile}"
        continue
    fi
    echo "  Launching ${label} game=${game} opp=${opp} seed=${seed} -> ${logfile}"
    python runs/liars_dice/ld_tom_ablation.py \
        --condition tom_cfr \
        --games "$GAMES" \
        --start-seed "$seed" \
        --num-dice "$NUM_DICE" \
        --num-faces "$NUM_FACES" \
        --cfr-model-path "$CFR_MODEL" \
        --archetype "$opp" \
        $extra \
        > "$logfile" 2>&1 &
    PIDS+=($!)
    NAMES+=("$label")
    sleep 2
done

if [[ "$DRY_RUN" -eq 1 ]]; then
    exit 0
fi

echo ""
echo "All 9 launched (2s staggered). Waiting for completion..."
echo ""

FAILED=0
for idx in "${!PIDS[@]}"; do
    pid="${PIDS[$idx]}"
    name="${NAMES[$idx]}"
    wait "$pid" && echo "  ✓ $name done" || { echo "  ✗ $name failed (PID $pid)"; ((FAILED++)); }
done

echo ""
echo "═══════════════════════════════════════════════"
echo "  All done. $FAILED failed, $(( ${#PIDS[@]} - FAILED )) succeeded."
echo "═══════════════════════════════════════════════"
echo ""
for label in "${NAMES[@]}"; do
    LOGFILE="$LOG_DIR/${label}.log"
    if [[ -f "$LOGFILE" ]]; then
        echo "── $label ──"
        grep -E "Result:|wr=|chips=" "$LOGFILE" | tail -3
        echo ""
    fi
done
echo "Full logs: $LOG_DIR/"

if [[ "$FAILED" -gt 0 ]]; then
    exit 1
fi
