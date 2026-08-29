#!/bin/bash
# Liar's Dice 2x3 — ideal_ablation rerun, complementary swapped seeds.
# For every archetype the sheet seeds are swapped:
#   full   uses the sheet's memory seed (PLSE-ablate-failed_trajectory rows: 1627)
#   memory uses the sheet's full seed (PLSE-full rows: 42)
#
# Full and memory groups run in parallel (12 live-API runs total).
#
# Usage (run from repo root):
#   bash ablation/run_ld_ideal_ablation_swapped.sh --dry-run
#   bash ablation/run_ld_ideal_ablation_swapped.sh --approved-real-measurement [--games 30]
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

ARCHETYPES=("Bluffer" "Honest" "AggressiveChallenger" "ConservativeChallenger" "Random" "GTO")
FULL_SEEDS=(1627 1627 1627 1627 1627 1627)
MEMORY_SEEDS=(42 42 42 42 42 42)
NUM_DICE=2
NUM_FACES=3
CFR_MODEL="results/liars_dice/cfr/cfr_2x3_evalevery/model"

LOG_DIR="results/liars_dice/ablation_logs/ideal_ablation_swapped"

if [[ "$DRY_RUN" -eq 0 && "$APPROVED" -ne 1 ]]; then
    echo "This launcher starts 12 live-API runs (Liar's Dice PLSE-full and PLSE-ablate-memory with swapped seeds x 6 archetypes, ${GAMES} games/position each)." >&2
    echo "It calls deepseek-v4-flash for game decisions (plus memory-refresh calls when enabled) and writes artifacts under results/." >&2
    echo "Preview with --dry-run, or re-run with --approved-real-measurement after explicit approval." >&2
    exit 2
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
    mkdir -p "$LOG_DIR"
fi

echo "Liar's Dice ideal_ablation swapped seeds — full and memory run in parallel, ${NUM_DICE}x${NUM_FACES}"
for i in "${!ARCHETYPES[@]}"; do
    printf "  %-24s full=%-5s memory=%s\n" "${ARCHETYPES[$i]}" "${FULL_SEEDS[$i]}" "${MEMORY_SEEDS[$i]}"
done
echo ""

PIDS=()
NAMES=()
for i in "${!ARCHETYPES[@]}"; do
    arch="${ARCHETYPES[$i]}"
    full_seed="${FULL_SEEDS[$i]}"
    mem_seed="${MEMORY_SEEDS[$i]}"
    full_log="$LOG_DIR/full_${arch}.log"
    mem_log="$LOG_DIR/memory_${arch}.log"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "  [dry-run] PLSE-full    ${arch} seed=${full_seed} -> ${full_log}"
        echo "  [dry-run] PLSE-memory  ${arch} seed=${mem_seed} --no-refresh-memory-bank -> ${mem_log}"
        continue
    fi
    echo "  Launching PLSE-full    ${arch} seed=${full_seed} -> ${full_log}"
    python runs/liars_dice/ld_tom_ablation.py \
        --condition tom_cfr \
        --games "$GAMES" \
        --start-seed "$full_seed" \
        --num-dice "$NUM_DICE" \
        --num-faces "$NUM_FACES" \
        --cfr-model-path "$CFR_MODEL" \
        --archetype "$arch" \
        > "$full_log" 2>&1 &
    PIDS+=($!)
    NAMES+=("full_${arch}")
    sleep 2
    echo "  Launching PLSE-memory  ${arch} seed=${mem_seed} -> ${mem_log}"
    python runs/liars_dice/ld_tom_ablation.py \
        --condition tom_cfr \
        --games "$GAMES" \
        --start-seed "$mem_seed" \
        --num-dice "$NUM_DICE" \
        --num-faces "$NUM_FACES" \
        --cfr-model-path "$CFR_MODEL" \
        --archetype "$arch" \
        --no-refresh-memory-bank \
        > "$mem_log" 2>&1 &
    PIDS+=($!)
    NAMES+=("memory_${arch}")
    sleep 2
done

if [[ "$DRY_RUN" -eq 1 ]]; then
    exit 0
fi

echo ""
echo "All 12 launched (2s staggered). Waiting for completion..."
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
echo "Results per archetype:"
for arch in "${ARCHETYPES[@]}"; do
    FULL_LOG="$LOG_DIR/full_${arch}.log"
    MEM_LOG="$LOG_DIR/memory_${arch}.log"
    echo "── $arch ──"
    if [[ -f "$FULL_LOG" ]]; then
        echo "  full:"
        grep -E "Result:|wr=|chips=" "$FULL_LOG" | tail -3
    fi
    if [[ -f "$MEM_LOG" ]]; then
        echo "  memory:"
        grep -E "Result:|wr=|chips=" "$MEM_LOG" | tail -3
    fi
    echo ""
done
echo "Full logs: $LOG_DIR/"

if [[ "$FAILED" -gt 0 ]]; then
    exit 1
fi
