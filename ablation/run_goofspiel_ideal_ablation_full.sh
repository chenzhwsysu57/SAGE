#!/bin/bash
# Goofspiel n=5 — ideal_ablation rerun: PLSE-full (case-based self-evolve
# memory ON, which is the run entry's default).
# Seeds from goofspiel工作簿.xlsx sheet "ideal_ablation" (PLSE-full rows: all 1623).
#
# Usage (run from repo root):
#   bash ablation/run_goofspiel_ideal_ablation_full.sh --dry-run
#   bash ablation/run_goofspiel_ideal_ablation_full.sh --approved-real-measurement [--games 30]
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
SEEDS=(1623 1623 1623 1623 1623 1623)
N=5

LOG_DIR="results/goofspiel/ablation_logs/ideal_ablation_full"

if [[ "$DRY_RUN" -eq 0 && "$APPROVED" -ne 1 ]]; then
    echo "This launcher starts 6 live-API runs (Goofspiel PLSE-full x 6 personas, ${GAMES} games/position each)." >&2
    echo "It calls deepseek-v4-flash for game decisions plus memory-refresh calls, and writes artifacts under results/." >&2
    echo "Preview with --dry-run, or re-run with --approved-real-measurement after explicit approval." >&2
    exit 2
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
    mkdir -p "$LOG_DIR"
fi

echo "Goofspiel ideal_ablation rerun — PLSE-full (memory ON), n=$N"
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
        echo "  [dry-run] PLSE-full ${persona} seed=${seed} -> ${logfile}"
        continue
    fi
    echo "  Launching PLSE-full ${persona} seed=${seed} -> ${logfile}"
    python runs/goofspiel/goofspiel_bench.py \
        --condition tom_bayes \
        --games "$GAMES" \
        --start-seed "$seed" \
        --n "$N" \
        --personas "$persona" \
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
