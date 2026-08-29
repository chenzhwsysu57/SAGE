#!/bin/bash
# Memory-only ablation across the three games and the five runall seeds.
# Seeds are processed one at a time; within a seed the 6 scripts (3 games x
# full/memory) run in parallel. "full" means memory ON (run entry default),
# "memory" means memory OFF (--no-refresh-memory-bank).
#
# Each per-game script internally runs its 6 opponents concurrently
# (30 games/position each, hardcoded in those scripts).
#
# Usage (run from repo root):
#   bash ablation/ablate_all_mem.sh --dry-run
#   bash ablation/ablate_all_mem.sh --approved-real-measurement
#
# Without --approved-real-measurement the script refuses to launch.

set -e

cd "$(dirname "$0")/.."

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
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

SEEDS=(985 1085 2085 3085 4085)
SCRIPTS=(
    ablation/run_leduc_ablate_none.sh
    ablation/run_leduc_ablate_memory.sh
    ablation/run_ld_ablate_none.sh
    ablation/run_ld_ablate_memory.sh
    ablation/run_goofspiel_ablate_none.sh
    ablation/run_goofspiel_ablate_memory.sh
)

LOG_DIR="results/ablation_logs/ablate_all_mem"

if [[ "$DRY_RUN" -eq 0 && "$APPROVED" -ne 1 ]]; then
    echo "This launcher starts 6 per-game scripts per seed (3 games x full/memory, 5 seeds, seeds sequential) with 30 games/position each." >&2
    echo "It calls deepseek-v4-flash for game decisions (plus memory-refresh calls for the full runs) and writes artifacts under results/." >&2
    echo "Preview with --dry-run, or re-run with --approved-real-measurement after explicit approval." >&2
    exit 2
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
    mkdir -p "$LOG_DIR"
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"

echo "Memory-only ablation — 3 games x 5 seeds, full(memory ON) vs memory(OFF)"
echo "Seeds: ${SEEDS[*]}"
echo ""

for seed in "${SEEDS[@]}"; do
    echo "===== memory ablation seed=${seed} ====="
    PIDS=()
    NAMES=()
    for script in "${SCRIPTS[@]}"; do
        name="$(basename "$script" .sh)"
        logfile="$LOG_DIR/${RUN_TS}_${name}_seed${seed}.log"
        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "  [dry-run] $script --seed $seed -> $logfile"
            continue
        fi
        echo "  Launching $script --seed $seed -> $logfile"
        bash "$script" --seed "$seed" > "$logfile" 2>&1 &
        PIDS+=($!)
        NAMES+=("$name")
    done

    if [[ "$DRY_RUN" -eq 1 ]]; then
        continue
    fi

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

    echo "===== finished seed=${seed}; failed=${failed} ====="
    if [[ "$failed" -gt 0 ]]; then
        exit 1
    fi
done

if [[ "$DRY_RUN" -eq 1 ]]; then
    exit 0
fi

echo ""
echo "══════════════════════════════════════════════════"
echo "  All 5 seeds done."
echo "══════════════════════════════════════════════════"

echo ""
echo "══════════════════════════════════════════════════"
echo "  Comparison summary: with memory (none) vs without memory"
echo "══════════════════════════════════════════════════"
for seed in "${SEEDS[@]}"; do
    echo ""
    echo "── seed ${seed} ──"
    for game in leduc ld goofspiel; do
        none_log="$LOG_DIR/${RUN_TS}_run_${game}_ablate_none_seed${seed}.log"
        mem_log="$LOG_DIR/${RUN_TS}_run_${game}_ablate_memory_seed${seed}.log"
        echo "  [${game}] with memory:"
        if [[ -f "$none_log" ]]; then
            grep -E "Result:" "$none_log" | sed 's/^/    /'
        fi
        echo "  [${game}] without memory:"
        if [[ -f "$mem_log" ]]; then
            grep -E "Result:" "$mem_log" | sed 's/^/    /'
        fi
    done
done

echo ""
echo "Full logs: $LOG_DIR/"
