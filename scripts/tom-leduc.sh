#!/bin/bash
# Leduc — ToM-CFR (statistical)
# Usage: bash tom-leduc.sh [--seed SEED]

set -e

ARCHETYPES=("Maniac" "LAG" "GTO" "CallingStation" "Nit" "TAG")
GAMES=30
SEED="$(date +%s)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)
            if [[ $# -lt 2 ]]; then
                echo "Error: --seed requires a value" >&2
                exit 1
            fi
            SEED="$2"
            shift 2
            ;;
        *)
            echo "Error: unknown argument '$1'. Usage: bash scripts/tom-leduc.sh [--seed SEED]" >&2
            exit 1
            ;;
    esac
done

LOG_DIR="results/leduc/seed_${SEED}_logs/tom"
mkdir -p "$LOG_DIR"

echo "Leduc — ToM-CFR — seed $SEED, $GAMES games each for ${#ARCHETYPES[@]} archetypes"
echo ""

PIDS=()
for arch in "${ARCHETYPES[@]}"; do
    LOGFILE="$LOG_DIR/${arch}.log"
    echo "  Launching $arch → $LOGFILE"
    python runs/leduc/leduc_tom_ablation.py \
        --condition tom_cfr \
        --games "$GAMES" \
        --start-seed "$SEED" \
        --archetype "$arch" \
        --method statistical \
        --stat-mode q_window_log \
        --q-tau 1.5 \
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
