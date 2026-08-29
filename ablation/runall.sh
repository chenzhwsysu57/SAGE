#!/bin/bash
# Run all ablation scripts for the two additional seed schedules.
# Usage: bash ablation/runall.sh

set -e

SEEDS=(985 1085 2085 3085 4085)
SCRIPTS=(
    ablation/run_leduc_ablate_none.sh
    ablation/run_leduc_ablate_nash.sh
    ablation/run_leduc_ablate_belief.sh
    ablation/run_leduc_ablate_memory.sh
    ablation/run_ld_ablate_none.sh
    ablation/run_ld_ablate_nash.sh
    ablation/run_ld_ablate_belief.sh
    ablation/run_ld_ablate_memory.sh
    ablation/run_goofspiel_ablate_none.sh
    ablation/run_goofspiel_ablate_nash.sh
    ablation/run_goofspiel_ablate_belief.sh
    ablation/run_goofspiel_ablate_memory.sh
)

LOG_DIR="results/ablation_batch_logs"
mkdir -p "$LOG_DIR"
RUN_TS="$(date +%Y%m%d_%H%M%S)"

for seed in "${SEEDS[@]}"; do
    echo "===== launching ablation batch seed=${seed} ====="

    PIDS=()
    NAMES=()
    for script in "${SCRIPTS[@]}"; do
        name="$(basename "$script" .sh)"
        logfile="$LOG_DIR/${RUN_TS}_${name}_seed${seed}.log"
        echo "  Launching $script --seed $seed -> $logfile"
        bash "$script" --seed "$seed" > "$logfile" 2>&1 &
        PIDS+=($!)
        NAMES+=("$name")
    done

    failed=0
    for i in "${!PIDS[@]}"; do
        pid="${PIDS[$i]}"
        name="${NAMES[$i]}"
        if wait "$pid"; then
            echo "  done: $name seed=$seed"
        else
            echo "  failed: $name seed=$seed pid=$pid"
            ((failed++))
        fi
    done

    echo "===== finished ablation batch seed=${seed}; failed=${failed} ====="
    if [[ "$failed" -gt 0 ]]; then
        exit 1
    fi
done

echo "All ablation batches completed. Logs: $LOG_DIR/"
