#!/bin/bash
# Leduc OOD soft-drift benchmark launcher.
#
# Parametric action-frequency opponent; theta stays at style A for
# --pre-stable games, interpolates linearly to style B over --ramp games,
# then stays at style B for --post-stable games (soft style switch with stable
# windows).  Compares the additive component ladder: simple / llm_nash /
# llm_om / full.
#
# Usage (run from repo root):
#   bash ablation/run_leduc_ood_drift.sh --dry-run
#   bash ablation/run_leduc_ood_drift.sh --approved-real-measurement [--games 40]
#
# This launches a live-API experiment (game decisions call deepseek-v4-flash;
# the `full` condition also generates case-based memory).  Without
# --approved-real-measurement the script refuses to launch.

set -e

cd "$(dirname "$0")/.."

GAMES=50
SEED=42
PRE_STABLE=10
RAMP=30
POST_STABLE=10
ENDPOINTS="LAG,Nit"
CONDITIONS="simple,llm_nash,llm_om,full"
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
        --pre-stable)
            [[ $# -ge 2 ]] || { echo "Missing value for --pre-stable" >&2; exit 2; }
            PRE_STABLE="$2"
            shift 2
            ;;
        --ramp)
            [[ $# -ge 2 ]] || { echo "Missing value for --ramp" >&2; exit 2; }
            RAMP="$2"
            shift 2
            ;;
        --post-stable)
            [[ $# -ge 2 ]] || { echo "Missing value for --post-stable" >&2; exit 2; }
            POST_STABLE="$2"
            shift 2
            ;;
        --seed)
            [[ $# -ge 2 ]] || { echo "Missing value for --seed" >&2; exit 2; }
            SEED="$2"
            shift 2
            ;;
        --endpoints)
            [[ $# -ge 2 ]] || { echo "Missing value for --endpoints" >&2; exit 2; }
            ENDPOINTS="$2"
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

LOG_DIR="results/leduc/ablation_logs/leduc_ood_drift"

if [[ "$DRY_RUN" -eq 0 && "$APPROVED" -ne 1 ]]; then
    echo "This launcher starts a live-API Leduc OOD drift experiment (${GAMES} games/position x 2 positions x conditions: ${CONDITIONS})." >&2
    echo "It calls deepseek-v4-flash for game decisions (plus memory-refresh calls for the full condition) and writes artifacts under results/." >&2
    echo "Preview with --dry-run, or re-run with --approved-real-measurement after explicit approval." >&2
    exit 2
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
    mkdir -p "$LOG_DIR"
fi
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOGFILE="$LOG_DIR/${RUN_TS}_drift_seed${SEED}_g${GAMES}.log"

CMD=(
    python runs/leduc/leduc_ood_bench.py
    --ood-opponent drift
    --conditions "$CONDITIONS"
    --drift-endpoints "$ENDPOINTS"
    --drift-pre-stable "$PRE_STABLE"
    --drift-ramp "$RAMP"
    --drift-post-stable "$POST_STABLE"
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

echo "Leduc OOD drift benchmark"
echo "  endpoints=$ENDPOINTS pre_stable=$PRE_STABLE ramp=$RAMP post_stable=$POST_STABLE games=$GAMES seed=$SEED conditions=$CONDITIONS"
echo "  logfile=$LOGFILE"
echo
echo "CMD: ${CMD[*]}"

if [[ "$DRY_RUN" -eq 1 ]]; then
    exit 0
fi

"${CMD[@]}" >"$LOGFILE" 2>&1
STATUS=$?
if [[ "$STATUS" -ne 0 ]]; then
    echo "OOD drift run FAILED (exit=$STATUS). Log: $LOGFILE" >&2
    exit "$STATUS"
fi
echo "Result: OOD drift OK  -> $(grep -o 'Results saved to: .*' "$LOGFILE" | tail -1)"
echo "Log: $LOGFILE"
