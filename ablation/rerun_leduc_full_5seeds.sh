#!/bin/bash
# Rerun the Leduc "full" (memory ON) condition of the 5-seed memory ablation.
#
# Why: the Leduc memory-generation prompt previously carried only the
# opponent action distribution (opp_1). After the style-posterior fix,
# Leduc method_data now includes the full opponent style distribution, so the
# memory prompts render the correct opponent estimate. The memory-ON Leduc
# runs of the five ablate_all_mem seeds must therefore be redone.
# The memory-OFF runs never generate memory and are unaffected.
#
# Seeds are processed one at a time; inside each seed the 6 archetypes run
# concurrently via ablation/run_leduc_ablate_none.sh (memory ON by default).
#
# Usage (run from repo root):
#   bash ablation/rerun_leduc_full_5seeds.sh --dry-run
#   bash ablation/rerun_leduc_full_5seeds.sh --approved-real-measurement
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
SCRIPT="ablation/run_leduc_ablate_none.sh"
LOG_DIR="results/leduc/ablation_logs/rerun_leduc_full_5seeds"
RUN_TS="$(date +%Y%m%d_%H%M%S)"

if [[ "$DRY_RUN" -eq 0 && "$APPROVED" -ne 1 ]]; then
    echo "This launcher re-runs 5 seeds x 6 archetypes = 30 live-API runs (Leduc full, memory ON, 30 games/position each)." >&2
    echo "It calls deepseek-v4-flash for game decisions plus memory-refresh calls, and writes artifacts under results/." >&2
    echo "Preview with --dry-run, or re-run with --approved-real-measurement after explicit approval." >&2
    exit 2
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
    mkdir -p "$LOG_DIR"
fi

echo "Leduc full (memory ON) rerun — 5 seeds, one seed at a time"
echo "Seeds: ${SEEDS[*]}"
echo ""

for seed in "${SEEDS[@]}"; do
    logfile="$LOG_DIR/${RUN_TS}_run_leduc_ablate_none_seed${seed}.log"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "  [dry-run] $SCRIPT --seed $seed -> $logfile"
        continue
    fi
    echo "===== leduc full (memory ON) seed=${seed} ====="
    bash "$SCRIPT" --seed "$seed" > "$logfile" 2>&1
    echo "  done: leduc full seed=${seed} (log: $logfile)"
done

if [[ "$DRY_RUN" -eq 1 ]]; then
    exit 0
fi

echo ""
echo "══════════════════════════════════════════════════"
echo "  Results per seed (Leduc full, memory ON)"
echo "══════════════════════════════════════════════════"
for seed in "${SEEDS[@]}"; do
    logfile="$LOG_DIR/${RUN_TS}_run_leduc_ablate_none_seed${seed}.log"
    echo ""
    echo "── seed ${seed} ──"
    if [[ -f "$logfile" ]]; then
        grep -E "Result:" "$logfile" | sed 's/^/    /'
    fi
done

echo ""
echo "Full logs: $LOG_DIR/"
