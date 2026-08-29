#!/bin/bash
# Leduc — rerun PLSE-full and PLSE-ablate-memory, complementary swapped seeds
# (script 2 of 2). For every opponent, the seeds of script 1 are swapped:
# full uses script 1's memory seed, memory uses script 1's full seed.
# LAG stays 42/42 because both seeds are 42 in the sheet.
#
# PLSE-full            = --condition tom_cfr --method statistical
#                        (case-based self-evolve memory refresh is ON by default)
# PLSE-ablate-memory   = same + --no-refresh-memory-bank
#                        (current replacement of the sheet's "failed_trajectory" condition)
#
# Seed table (--start-seed per opponent, swapped vs script 1):
#   Maniac         full=1627  memory=42
#   LAG            full=42    memory=42
#   GTO            full=42    memory=1440
#   CallingStation full=1627  memory=1440
#   Nit            full=1627  memory=42
#   TAG            full=42    memory=1440
#
# Usage (run from repo root):
#   bash ablation/run_leduc_ideal_ablation_rerun_swapped.sh --dry-run
#   bash ablation/run_leduc_ideal_ablation_rerun_swapped.sh --approved-real-measurement [--games 30]
#
# This launches 12 live-API runs (2 groups x 6 archetypes, 30 games/position each).
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

ARCHETYPES=("Maniac" "LAG" "GTO" "CallingStation" "Nit" "TAG")
FULL_SEEDS=(1627 42 42 1627 1627 42)
MEMORY_SEEDS=(42 42 1440 1440 42 1440)

LOG_DIR="results/leduc/ablation_logs/ideal_ablation_rerun_swapped"

if [[ "$DRY_RUN" -eq 0 && "$APPROVED" -ne 1 ]]; then
    echo "This launcher starts 12 live-API runs (PLSE-full and PLSE-ablate-memory x 6 archetypes, ${GAMES} games/position each)." >&2
    echo "It calls deepseek-v4-flash for game decisions (plus memory-refresh calls when enabled) and writes artifacts under results/." >&2
    echo "Preview with --dry-run, or re-run with --approved-real-measurement after explicit approval." >&2
    exit 2
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
    mkdir -p "$LOG_DIR"
fi

run_group() {
    local label="$1"      # full | memory
    local extra="$2"      # "" | "--no-refresh-memory-bank"
    shift 2
    local seeds=("$@")
    local pids=()

    echo ""
    echo "===== PLSE-${label} (${GAMES} games/position) ====="
    for i in "${!ARCHETYPES[@]}"; do
        local arch="${ARCHETYPES[$i]}"
        local seed="${seeds[$i]}"
        local logfile="$LOG_DIR/${label}_${arch}.log"
        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "  [dry-run] PLSE-${label} ${arch} seed=${seed}${extra:+ $extra} -> ${logfile}"
            continue
        fi
        echo "  Launching PLSE-${label} ${arch} seed=${seed} -> ${logfile}"
        python runs/leduc/leduc_tom_ablation.py \
            --condition tom_cfr \
            --games "$GAMES" \
            --start-seed "$seed" \
            --archetype "$arch" \
            --method statistical \
            $extra \
            > "$logfile" 2>&1 &
        pids+=($!)
        sleep 2
    done

    if [[ "$DRY_RUN" -eq 1 ]]; then
        return 0
    fi

    local failed=0
    for i in "${!pids[@]}"; do
        local pid="${pids[$i]}"
        local arch="${ARCHETYPES[$i]}"
        if wait "$pid"; then
            echo "  OK  PLSE-${label} ${arch}"
        else
            echo "  FAILED PLSE-${label} ${arch} (pid ${pid})"
            ((failed++))
        fi
    done
    return "$failed"
}

echo "Leduc ideal_ablation rerun — swapped complementary seeds (script 2)"
for i in "${!ARCHETYPES[@]}"; do
    printf "  %-15s full=%-5s memory=%s\n" "${ARCHETYPES[$i]}" "${FULL_SEEDS[$i]}" "${MEMORY_SEEDS[$i]}"
done

full_failed=0
run_group full "" "${FULL_SEEDS[@]}" || full_failed=$?

memory_failed=0
run_group memory "--no-refresh-memory-bank" "${MEMORY_SEEDS[@]}" || memory_failed=$?

echo ""
echo "═══════════════════════════════════════════"
echo "  Done. full_failed=${full_failed} memory_failed=${memory_failed}"
echo "  Logs: ${LOG_DIR}/"
echo "═══════════════════════════════════════════"

if [[ "$full_failed" -gt 0 || "$memory_failed" -gt 0 ]]; then
    exit 1
fi
