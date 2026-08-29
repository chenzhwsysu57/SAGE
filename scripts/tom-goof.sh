#!/bin/bash
# Goofspiel n=5 — ToM-Bayes
# Usage: bash tom-goof.sh

set -e

PERSONAS=("Greedy" "Conservative" "Sacrificial" "Random" "GTO" "Mirror")
GAMES=30
SEED=1637
N=5
LOG_DIR="results/goofspiel/seed_1440_logs/tom"
mkdir -p "$LOG_DIR"

echo "Goofspiel n=$N — ToM-Bayes — seed $SEED, $GAMES games each for ${#PERSONAS[@]} personas"
echo ""

PIDS=()
for persona in "${PERSONAS[@]}"; do
    LOGFILE="$LOG_DIR/${persona}.log"
    echo "  Launching $persona → $LOGFILE"
    python runs/goofspiel/goofspiel_bench.py \
        --condition tom_bayes \
        --games "$GAMES" \
        --start-seed "$SEED" \
        --n "$N" \
        --personas "$persona" \
        > "$LOGFILE" 2>&1 &
    PIDS+=($!)
    sleep 2
done

echo ""
echo "All 6 launched (2s staggered). Waiting for completion..."
echo ""

FAILED=0
for i in "${!PERSONAS[@]}"; do
    persona="${PERSONAS[$i]}"
    pid="${PIDS[$i]}"
    wait "$pid" && echo "  ✓ $persona done" || { echo "  ✗ $persona failed (PID $pid)"; ((FAILED++)); }
done

echo ""
echo "═══════════════════════════════════════════════"
echo "  All done. $FAILED failed, $(( ${#PERSONAS[@]} - FAILED )) succeeded."
echo "═══════════════════════════════════════════════"
echo ""
echo "Results per persona:"
for persona in "${PERSONAS[@]}"; do
    LOGFILE="$LOG_DIR/${persona}.log"
    if [[ -f "$LOGFILE" ]]; then
        echo "── $persona ──"
        grep -E "Result:|wr=|chips=" "$LOGFILE" | tail -3
        echo ""
    fi
done
echo "Full logs: $LOG_DIR/"
