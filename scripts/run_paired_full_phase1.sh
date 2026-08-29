#!/bin/bash
# Paired Full Phase 1: run full ToM conditions for LD and Goofspiel
# under the same seed schedules used by the added ablation runs.
# Usage: bash scripts/run_paired_full_phase1.sh

set -e

SEEDS=(1623 1627)
GAMES=30

LD_ARCHETYPES=(
    Bluffer
    Honest
    AggressiveChallenger
    ConservativeChallenger
    Random
    GTO
)
LD_NUM_DICE=2
LD_NUM_FACES=3
LD_CFR_MODEL="results/liars_dice/cfr/cfr_2x3_evalevery/model"

GOOF_PERSONAS=(
    Greedy
    Conservative
    Sacrificial
    Random
    GTO
    Mirror
)
GOOF_N=5

LOG_DIR="results/paired_full_phase1_logs"
mkdir -p "$LOG_DIR"

for seed in "${SEEDS[@]}"; do
    echo "===== launching paired full phase1 seed=${seed} ====="

    PIDS=()
    NAMES=()

    for arch in "${LD_ARCHETYPES[@]}"; do
        name="ld_full_${arch}_seed${seed}"
        logfile="$LOG_DIR/${name}.log"
        echo "  Launching LD full $arch --seed $seed -> $logfile"
        python runs/liars_dice/ld_tom_ablation.py \
            --condition tom_cfr \
            --games "$GAMES" \
            --start-seed "$seed" \
            --num-dice "$LD_NUM_DICE" \
            --num-faces "$LD_NUM_FACES" \
            --cfr-model-path "$LD_CFR_MODEL" \
            --archetype "$arch" \
            > "$logfile" 2>&1 &
        PIDS+=($!)
        NAMES+=("$name")
    done

    for persona in "${GOOF_PERSONAS[@]}"; do
        name="goof_full_${persona}_seed${seed}"
        logfile="$LOG_DIR/${name}.log"
        echo "  Launching Goofspiel full $persona --seed $seed -> $logfile"
        python runs/goofspiel/goofspiel_bench.py \
            --condition tom_bayes \
            --games "$GAMES" \
            --start-seed "$seed" \
            --n "$GOOF_N" \
            --personas "$persona" \
            > "$logfile" 2>&1 &
        PIDS+=($!)
        NAMES+=("$name")
    done

    failed=0
    for i in "${!PIDS[@]}"; do
        pid="${PIDS[$i]}"
        name="${NAMES[$i]}"
        if wait "$pid"; then
            echo "  done: $name"
        else
            echo "  failed: $name pid=$pid"
            ((failed++))
        fi
    done

    echo "===== finished paired full phase1 seed=${seed}; failed=${failed} ====="
    if [[ "$failed" -gt 0 ]]; then
        exit 1
    fi
done

echo "Paired Full Phase 1 completed. Logs: $LOG_DIR/"
