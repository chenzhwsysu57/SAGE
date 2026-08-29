#!/bin/bash
# Leduc — ablate belief (remove OPPONENT MODEL section)
# Usage: bash ablation/run_leduc_ablate_belief.sh [--seed SEED]

set -e

ARCHETYPES=("Maniac" "LAG" "GTO" "CallingStation" "Nit" "TAG")
GAMES=30
SEED=42
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
LOG_DIR="results/leduc/ablation_logs/ablate_belief"
mkdir -p "$LOG_DIR"
RUN_TS="$(date +%Y%m%d_%H%M%S)"

echo "Leduc — ablate belief — $GAMES games each for ${#ARCHETYPES[@]} archetypes"
echo ""

PIDS=()
for arch in "${ARCHETYPES[@]}"; do
    LOGFILE="$LOG_DIR/${RUN_TS}_${arch}.log"
    echo "  Launching $arch → $LOGFILE"
    python runs/leduc/leduc_tom_ablation.py \
        --condition tom_cfr \
        --games "$GAMES" \
        --start-seed "$SEED" \
        --archetype "$arch" \
        --method statistical \
        --ablate belief \
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
    LOGFILE="$LOG_DIR/${RUN_TS}_${arch}.log"
    if [[ -f "$LOGFILE" ]]; then
        echo "── $arch ──"
        grep -E "Result:|wr=|chips=" "$LOGFILE" | tail -3
        echo ""
    fi
done
echo "Full logs: $LOG_DIR/"
