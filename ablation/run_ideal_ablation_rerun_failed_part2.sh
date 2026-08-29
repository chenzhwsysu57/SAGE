#!/bin/bash
# Rerun of the failed ideal_ablation runs — part 2 of 3 (8 runs, parallel).
# Run part1/part2/part3 simultaneously (8 + 8 + 9 = 25 failed runs).
#
# This part covers: goofspiel full swapped (5), liars dice full sheet (3).
#
# Usage (run from repo root):
#   bash ablation/run_ideal_ablation_rerun_failed_part2.sh --dry-run
#   bash ablation/run_ideal_ablation_rerun_failed_part2.sh --approved-real-measurement [--games 30]
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

N=5
NUM_DICE=2
NUM_FACES=3
CFR_MODEL="results/liars_dice/cfr/cfr_2x3_evalevery/model"

# label|game|opponent|seed|refresh(1=full, 0=memory)
ENTRIES=(
    "goof_full_swapped_Conservative|goofspiel|Conservative|42|1"
    "goof_full_swapped_Sacrificial|goofspiel|Sacrificial|42|1"
    "goof_full_swapped_Random|goofspiel|Random|1558|1"
    "goof_full_swapped_GTO|goofspiel|GTO|42|1"
    "goof_full_swapped_Mirror|goofspiel|Mirror|1627|1"
    "ld_full_Bluffer|liars_dice|Bluffer|42|1"
    "ld_full_Honest|liars_dice|Honest|42|1"
    "ld_full_AggressiveChallenger|liars_dice|AggressiveChallenger|42|1"
)

LOG_DIR="results/ablation_logs/ideal_ablation_rerun_failed_part2"

if [[ "$DRY_RUN" -eq 0 && "$APPROVED" -ne 1 ]]; then
    echo "This launcher starts 8 live-API runs (failed ideal_ablation rerun, part 2) of ${GAMES} games/position each." >&2
    echo "It calls deepseek-v4-flash for game decisions (plus memory-refresh calls when enabled) and writes artifacts under results/." >&2
    echo "Preview with --dry-run, or re-run with --approved-real-measurement after explicit approval." >&2
    exit 2
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
    mkdir -p "$LOG_DIR"
fi

echo "Ideal_ablation failed rerun — part 2 (8 runs, parallel)"
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
    if [[ "$game" == "goofspiel" ]]; then
        python runs/goofspiel/goofspiel_bench.py \
            --condition tom_bayes \
            --games "$GAMES" \
            --start-seed "$seed" \
            --n "$N" \
            --personas "$opp" \
            $extra \
            > "$logfile" 2>&1 &
    else
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
    fi
    PIDS+=($!)
    NAMES+=("$label")
    sleep 2
done

if [[ "$DRY_RUN" -eq 1 ]]; then
    exit 0
fi

echo ""
echo "All 8 launched (2s staggered). Waiting for completion..."
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
