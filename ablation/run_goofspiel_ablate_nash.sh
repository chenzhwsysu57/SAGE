#!/bin/bash
# Goofspiel n=5 — ablate Nash (remove GTO BASELINE section)
# Usage: bash ablation/run_goofspiel_ablate_nash.sh [--seed SEED]

set -e

PERSONAS=("Greedy" "Conservative" "Sacrificial" "Random" "GTO" "Mirror")
GAMES=30
SEED=42
N=5
while [[ $# -gt 0 ]]; do
    case "$1" in
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
LOG_DIR="results/goofspiel/ablation_logs/ablate_nash"
mkdir -p "$LOG_DIR"
RUN_TS="$(date +%Y%m%d_%H%M%S)"

echo "Goofspiel n=$N — ablate nash — $GAMES games each for ${#PERSONAS[@]} personas"
echo ""

PIDS=()
for persona in "${PERSONAS[@]}"; do
    LOGFILE="$LOG_DIR/${RUN_TS}_${persona}.log"
    echo "  Launching $persona → $LOGFILE"
    python runs/goofspiel/goofspiel_bench.py \
        --condition tom_bayes \
        --games "$GAMES" \
        --start-seed "$SEED" \
        --n "$N" \
        --personas "$persona" \
        --ablate nash \
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
    LOGFILE="$LOG_DIR/${RUN_TS}_${persona}.log"
    if [[ -f "$LOGFILE" ]]; then
        echo "── $persona ──"
        grep -E "Result:|wr=|chips=" "$LOGFILE" | tail -3
        echo ""
    fi
done
echo "Full logs: $LOG_DIR/"
