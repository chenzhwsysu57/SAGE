#!/bin/bash
# Goofspiel n=5 — ideal_ablation rerun: PLSE-ablate-memory (memory OFF via
# --no-refresh-memory-bank), the current replacement of the sheet's
# "failed_trajectory" condition.
# Seeds from goofspiel工作簿.xlsx sheet "ideal_ablation"
# (PLSE-ablate-failed_trajectory rows, per persona).
#
# Usage (run from repo root):
#   bash ablation/run_goofspiel_ideal_ablation_memory.sh --dry-run
#   bash ablation/run_goofspiel_ideal_ablation_memory.sh --approved-real-measurement [--games 30]
#
# This launches 6 live-API runs (6 personas, 30 games/position each).
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

PERSONAS=("Greedy" "Conservative" "Sacrificial" "Random" "GTO" "Mirror")
SEEDS=(1627 42 42 1558 42 1627)
N=5

LOG_DIR="results/goofspiel/ablation_logs/ideal_ablation_memory"

if [[ "$DRY_RUN" -eq 0 && "$APPROVED" -ne 1 ]]; then
    echo "This launcher starts 6 live-API runs (Goofspiel PLSE-ablate-memory x 6 personas, ${GAMES} games/position each)." >&2
    echo "It calls deepseek-v4-flash for game decisions (no memory-refresh calls) and writes artifacts under results/." >&2
    echo "Preview with --dry-run, or re-run with --approved-real-measurement after explicit approval." >&2
    exit 2
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
    mkdir -p "$LOG_DIR"
fi

echo "Goofspiel ideal_ablation rerun — PLSE-ablate-memory (memory OFF), n=$N"
for i in "${!PERSONAS[@]}"; do
    printf "  %-15s seed=%s\n" "${PERSONAS[$i]}" "${SEEDS[$i]}"
done
echo ""

PIDS=()
for i in "${!PERSONAS[@]}"; do
    persona="${PERSONAS[$i]}"
    seed="${SEEDS[$i]}"
    logfile="$LOG_DIR/${persona}.log"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "  [dry-run] PLSE-ablate-memory ${persona} seed=${seed} --no-refresh-memory-bank -> ${logfile}"
        continue
    fi
    echo "  Launching PLSE-ablate-memory ${persona} seed=${seed} -> ${logfile}"
    python runs/goofspiel/goofspiel_bench.py \
        --condition tom_bayes \
        --games "$GAMES" \
        --start-seed "$seed" \
        --n "$N" \
        --personas "$persona" \
        --no-refresh-memory-bank \
        > "$logfile" 2>&1 &
    PIDS+=($!)
    sleep 2
done

if [[ "$DRY_RUN" -eq 1 ]]; then
    exit 0
fi

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

if [[ "$FAILED" -gt 0 ]]; then
    exit 1
fi
