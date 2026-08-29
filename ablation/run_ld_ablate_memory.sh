#!/bin/bash
# Liar's Dice 2x3 — ablate memory (remove case-based self-evolve memory)
# Usage: bash ablation/run_ld_ablate_memory.sh [--seed SEED]

set -e

ARCHETYPES=("Bluffer" "Honest" "AggressiveChallenger" "ConservativeChallenger" "Random" "GTO")
GAMES=30
SEED=42
NUM_DICE=2
NUM_FACES=3
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
CFR_MODEL="results/liars_dice/cfr/cfr_2x3_evalevery/model"
LOG_DIR="results/liars_dice/ablation_logs/ablate_memory"
mkdir -p "$LOG_DIR"
RUN_TS="$(date +%Y%m%d_%H%M%S)"

echo "Liar's Dice 2x3 — ablate memory (no self-evolve memory) — $GAMES games each for ${#ARCHETYPES[@]} archetypes"
echo ""

PIDS=()
for arch in "${ARCHETYPES[@]}"; do
    LOGFILE="$LOG_DIR/${RUN_TS}_${arch}.log"
    echo "  Launching $arch → $LOGFILE"
    python runs/liars_dice/ld_tom_ablation.py \
        --condition tom_cfr \
        --games "$GAMES" \
        --start-seed "$SEED" \
        --num-dice "$NUM_DICE" \
        --num-faces "$NUM_FACES" \
        --cfr-model-path "$CFR_MODEL" \
        --archetype "$arch" \
        --no-refresh-memory-bank \
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
