#!/bin/bash
# Leduc — CFR baseline
# Usage: bash cfr-leduc.sh

set -e

ARCHETYPES=("Maniac" "LAG" "GTO" "CallingStation" "Nit" "TAG")
GAMES=30
SEED=1637
LOG_DIR="results/leduc/seed_1440_logs/cfr"
mkdir -p "$LOG_DIR"

echo "Leduc — CFR baseline — seed $SEED, $GAMES games each for ${#ARCHETYPES[@]} archetypes"
echo ""

PIDS=()
for arch in "${ARCHETYPES[@]}"; do
    LOGFILE="$LOG_DIR/${arch}.log"
    echo "  Launching $arch → $LOGFILE"
    python runs/leduc/leduc_tom_ablation.py \
        --condition cfr \
        --games "$GAMES" \
        --start-seed "$SEED" \
        --archetype "$arch" \
        --method llm \
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
