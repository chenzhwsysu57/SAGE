#!/bin/bash
# Liar's Dice 2x3 — ToM+CFR
# Usage: bash tom-ld.sh

set -e

ARCHETYPES=("Bluffer" "Honest" "AggressiveChallenger" "ConservativeChallenger" "Random" "GTO")
GAMES=30
SEED=1637
NUM_DICE=2
NUM_FACES=3
CFR_MODEL="results/liars_dice/cfr/cfr_2x3_evalevery/model"
LOG_DIR="results/liars_dice/seed_1440_logs/tom"
mkdir -p "$LOG_DIR"

echo "Liar's Dice 2x3 — ToM+CFR — seed $SEED, $GAMES games each for ${#ARCHETYPES[@]} archetypes"
echo ""

PIDS=()
for arch in "${ARCHETYPES[@]}"; do
    LOGFILE="$LOG_DIR/${arch}.log"
    echo "  Launching $arch → $LOGFILE"
    python runs/liars_dice/ld_tom_ablation.py \
        --condition tom_cfr \
        --games "$GAMES" \
        --start-seed "$SEED" \
        --num-dice "$NUM_DICE" \
        --num-faces "$NUM_FACES" \
        --cfr-model-path "$CFR_MODEL" \
        --archetype "$arch" \
        > "$LOGFILE" 2>&1 &
    PIDS+=($!)
    sleep 2
done

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
