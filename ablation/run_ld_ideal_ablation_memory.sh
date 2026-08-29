#!/bin/bash
# Liar's Dice 2x3 — ideal_ablation rerun: PLSE-ablate-memory (memory OFF via
# --no-refresh-memory-bank), the current replacement of the sheet's
# "failed_trajectory" condition.
# Seeds from liarsdice工作簿.xlsx sheet "ideal_ablation"
# (PLSE-ablate-failed_trajectory rows: all 1627).
#
# Usage (run from repo root):
#   bash ablation/run_ld_ideal_ablation_memory.sh --dry-run
#   bash ablation/run_ld_ideal_ablation_memory.sh --approved-real-measurement [--games 30]
#
# This launches 6 live-API runs (6 archetypes, 30 games/position each).
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

ARCHETYPES=("Bluffer" "Honest" "AggressiveChallenger" "ConservativeChallenger" "Random" "GTO")
SEEDS=(1627 1627 1627 1627 1627 1627)
NUM_DICE=2
NUM_FACES=3
CFR_MODEL="results/liars_dice/cfr/cfr_2x3_evalevery/model"

LOG_DIR="results/liars_dice/ablation_logs/ideal_ablation_memory"

if [[ "$DRY_RUN" -eq 0 && "$APPROVED" -ne 1 ]]; then
    echo "This launcher starts 6 live-API runs (Liar's Dice PLSE-ablate-memory x 6 archetypes, ${GAMES} games/position each)." >&2
    echo "It calls deepseek-v4-flash for game decisions (no memory-refresh calls) and writes artifacts under results/." >&2
    echo "Preview with --dry-run, or re-run with --approved-real-measurement after explicit approval." >&2
    exit 2
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
    mkdir -p "$LOG_DIR"
fi

echo "Liar's Dice ideal_ablation rerun — PLSE-ablate-memory (memory OFF), ${NUM_DICE}x${NUM_FACES}"
for i in "${!ARCHETYPES[@]}"; do
    printf "  %-24s seed=%s\n" "${ARCHETYPES[$i]}" "${SEEDS[$i]}"
done
echo ""

PIDS=()
for i in "${!ARCHETYPES[@]}"; do
    arch="${ARCHETYPES[$i]}"
    seed="${SEEDS[$i]}"
    logfile="$LOG_DIR/${arch}.log"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "  [dry-run] PLSE-ablate-memory ${arch} seed=${seed} --no-refresh-memory-bank -> ${logfile}"
        continue
    fi
    echo "  Launching PLSE-ablate-memory ${arch} seed=${seed} -> ${logfile}"
    python runs/liars_dice/ld_tom_ablation.py \
        --condition tom_cfr \
        --games "$GAMES" \
        --start-seed "$seed" \
        --num-dice "$NUM_DICE" \
        --num-faces "$NUM_FACES" \
        --cfr-model-path "$CFR_MODEL" \
        --archetype "$arch" \
        --no-refresh-memory-bank \
        > "$logfile" 2>&1 &
    PIDS+=($!)
    sleep 2
done

if [[ "$DRY_RUN" -eq 1 ]]; then
    exit 0
fi

echo ""
echo "All 6 launched (2s staggered). Waiting for completion..."
echo ""

FAILED=0
for i in "${!ARCHETYPES[@]}"; do
    arch="${ARCHETYPES[$i]}"
    pid="${PIDS[$i]}"
    wait "$pid" && echo "  ✓ $arch done" || { echo "  ✗ $arch failed (PID $pid)"; ((FAILED++)); }
done

echo ""
echo "═══════════════════════════════════════════════"
echo "  All done. $FAILED failed, $(( ${#ARCHETYPES[@]} - FAILED )) succeeded."
echo "═══════════════════════════════════════════════"
echo ""
echo "Results per archetype:"
for arch in "${ARCHETYPES[@]}"; do
    LOGFILE="$LOG_DIR/${arch}.log"
    if [[ -f "$LOGFILE" ]]; then
        echo "── $arch ──"
        grep -E "Result:|wr=|chips=" "$LOGFILE" | tail -3
        echo ""
    fi
done
echo "Full logs: $LOG_DIR/"

if [[ "$FAILED" -gt 0 ]]; then
    exit 1
fi
