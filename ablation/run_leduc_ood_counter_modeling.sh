#!/bin/bash
# Leduc OOD counter-modeling benchmark launcher.
#
# "Cheater" opponent reads the method's style posterior from the previous game
# (harness-provided channel); whenever the argmax confidence stays above
# --threshold for --min-games consecutive games it switches to one of the two
# farthest styles by signature distance (farthest, or second-farthest with
# probability --second-prob).  Switching is continuous.  Compares llm_om /
# full only.
#
# Usage (run from repo root):
#   bash ablation/run_leduc_ood_counter_modeling.sh --dry-run
#   bash ablation/run_leduc_ood_counter_modeling.sh --approved-real-measurement [--games 40]
#
# Live-API experiment (game decisions call deepseek-v4-flash; the `full`
# condition also generates case-based memory).  Without
# --approved-real-measurement the script refuses to launch.

set -e

cd "$(dirname "$0")/.."

GAMES=40
SEED=42
START_STYLE="LAG"
THRESHOLD=0.5
MIN_GAMES=3
SECOND_PROB=0.5
CONDITIONS="llm_om,full"
SAVE_PROMPTS=1
APPROVED=0
DRY_RUN=0
EXTRA_ARGS=()

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
        --seed)
            [[ $# -ge 2 ]] || { echo "Missing value for --seed" >&2; exit 2; }
            SEED="$2"
            shift 2
            ;;
        --start-style)
            [[ $# -ge 2 ]] || { echo "Missing value for --start-style" >&2; exit 2; }
            START_STYLE="$2"
            shift 2
            ;;
        --threshold)
            [[ $# -ge 2 ]] || { echo "Missing value for --threshold" >&2; exit 2; }
            THRESHOLD="$2"
            shift 2
            ;;
        --min-games)
            [[ $# -ge 2 ]] || { echo "Missing value for --min-games" >&2; exit 2; }
            MIN_GAMES="$2"
            shift 2
            ;;
        --second-prob)
            [[ $# -ge 2 ]] || { echo "Missing value for --second-prob" >&2; exit 2; }
            SECOND_PROB="$2"
            shift 2
            ;;
        --conditions)
            [[ $# -ge 2 ]] || { echo "Missing value for --conditions" >&2; exit 2; }
            CONDITIONS="$2"
            shift 2
            ;;
        --no-save-prompts)
            SAVE_PROMPTS=0
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

LOG_DIR="results/leduc/ablation_logs/leduc_ood_counter_modeling"

if [[ "$DRY_RUN" -eq 0 && "$APPROVED" -ne 1 ]]; then
    echo "This launcher starts a live-API Leduc OOD counter-modeling experiment (${GAMES} games/position x 2 positions x conditions: ${CONDITIONS})." >&2
    echo "It calls deepseek-v4-flash for game decisions (plus memory-refresh calls for the full condition) and writes artifacts under results/." >&2
    echo "Preview with --dry-run, or re-run with --approved-real-measurement after explicit approval." >&2
    exit 2
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
    mkdir -p "$LOG_DIR"
fi
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOGFILE="$LOG_DIR/${RUN_TS}_counter_seed${SEED}_g${GAMES}.log"

CMD=(
    python runs/leduc/leduc_ood_bench.py
    --ood-opponent counter_modeling
    --conditions "$CONDITIONS"
    --start-style "$START_STYLE"
    --switch-threshold "$THRESHOLD"
    --switch-min-games "$MIN_GAMES"
    --switch-second-prob "$SECOND_PROB"
    --games "$GAMES"
    --start-seed "$SEED"
)
if [[ "$APPROVED" -eq 1 ]]; then
    CMD+=(--run-phase real_measurement --approved-real-measurement)
fi
if [[ "$SAVE_PROMPTS" -eq 0 ]]; then
    CMD+=(--no-save-prompts)
fi
if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Leduc OOD counter-modeling benchmark"
echo "  start_style=$START_STYLE threshold=$THRESHOLD min_games=$MIN_GAMES second_prob=$SECOND_PROB seed=$SEED conditions=$CONDITIONS"
echo "  logfile=$LOGFILE"
echo
echo "CMD: ${CMD[*]}"

if [[ "$DRY_RUN" -eq 1 ]]; then
    exit 0
fi

"${CMD[@]}" >"$LOGFILE" 2>&1
STATUS=$?
if [[ "$STATUS" -ne 0 ]]; then
    echo "OOD counter-modeling run FAILED (exit=$STATUS). Log: $LOGFILE" >&2
    exit "$STATUS"
fi
echo "Result: OOD counter-modeling OK  -> $(grep -o 'Results saved to: .*' "$LOGFILE" | tail -1)"
echo "Log: $LOGFILE"
