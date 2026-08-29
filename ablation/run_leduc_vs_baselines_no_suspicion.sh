#!/bin/bash
# Leduc method-vs-method long-run launcher (no suspicion).
#
# Pits the Full (SAGE) agent against three LLM opponent methods only:
#   simple (vanilla), llm_nash (LLM+Nash), llm_om (LLM+OM).
# Designed for longer-run statistics: defaults to 100 games/position and
# supports multiple seeds (--seeds 42,1332,4085) so that each seed draws an
# independent pair of hand streams for P0/P1 (Leduc deals player 0 the
# deck[0] stream and player 1 the deck[1] stream for a given seed).
#
# Usage (run from repo root):
#   bash ablation/run_leduc_vs_baselines_no_suspicion.sh --dry-run
#   bash ablation/run_leduc_vs_baselines_no_suspicion.sh --approved-real-measurement [--games 100] [--seed 42]
#   bash ablation/run_leduc_vs_baselines_no_suspicion.sh --approved-real-measurement --seeds 42,1332,4085 --games 100
#
# Live-API experiment (both players call deepseek-v4-flash; the full agent
# also generates case-based memory).  Without --approved-real-measurement the
# script refuses to launch.

set -e

cd "$(dirname "$0")/.."

GAMES=100
SEED=42
SEEDS=""
AGENT=full
OPPONENTS=(simple llm_nash llm_om)
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
        --seeds)
            [[ $# -ge 2 ]] || { echo "Missing value for --seeds" >&2; exit 2; }
            SEEDS="$2"
            shift 2
            ;;
        --agent)
            [[ $# -ge 2 ]] || { echo "Missing value for --agent" >&2; exit 2; }
            AGENT="$2"
            shift 2
            ;;
        --opponents)
            [[ $# -ge 2 ]] || { echo "Missing value for --opponents" >&2; exit 2; }
            IFS=',' read -r -a OPPONENTS <<< "$2"
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

if [[ -n "$SEEDS" ]]; then
    IFS=',' read -r -a SEED_LIST <<< "$SEEDS"
else
    SEED_LIST=("$SEED")
fi

LOG_DIR="results/leduc/ablation_logs/leduc_vs_baselines_no_suspicion"
TOTAL_TASKS=$(( ${#OPPONENTS[@]} * ${#SEED_LIST[@]} ))

if [[ "$DRY_RUN" -eq 0 && "$APPROVED" -ne 1 ]]; then
    echo "This launcher starts live-API method-vs-method battles (agent=$AGENT vs ${OPPONENTS[*]}, seeds=${SEED_LIST[*]})." >&2
    echo "Both players call deepseek-v4-flash (plus memory-refresh for the full agent); artifacts go under results/." >&2
    echo "Preview with --dry-run, or re-run with --approved-real-measurement after explicit approval." >&2
    exit 2
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
    mkdir -p "$LOG_DIR"
fi
RUN_TS="$(date +%Y%m%d_%H%M%S)"

echo "Leduc method-vs-method long-run launcher (no suspicion)"
echo "  agent=$AGENT  opponents=${OPPONENTS[*]}  seeds=${SEED_LIST[*]}  games/position=$GAMES"
echo "  tasks=$TOTAL_TASKS  log dir: $LOG_DIR"
echo

PIDS=()
for seed in "${SEED_LIST[@]}"; do
    for opp in "${OPPONENTS[@]}"; do
        LOGFILE="$LOG_DIR/${RUN_TS}_${AGENT}_vs_${opp}_g${GAMES}_seed${seed}.log"
        CMD=(
            python runs/leduc/leduc_vs_bench.py
            --agent-condition "$AGENT"
            --opponent-condition "$opp"
            --games "$GAMES"
            --start-seed "$seed"
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
        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "DRY-RUN: ${CMD[*]}"
            echo "         log -> $LOGFILE"
            continue
        fi
        "${CMD[@]}" >"$LOGFILE" 2>&1 &
        PIDS+=($!)
        sleep 1
    done
done

if [[ "$DRY_RUN" -eq 1 ]]; then
    exit 0
fi

FAILED=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAILED=$((FAILED + 1))
done

for seed in "${SEED_LIST[@]}"; do
    for opp in "${OPPONENTS[@]}"; do
        LOGFILE="$LOG_DIR/${RUN_TS}_${AGENT}_vs_${opp}_g${GAMES}_seed${seed}.log"
        if grep -q "Results saved to:" "$LOGFILE"; then
            echo "Result: $AGENT vs $opp (seed $seed) OK  -> $(grep -o 'Results saved to: .*' "$LOGFILE" | tail -1)"
        else
            echo "Result: $AGENT vs $opp (seed $seed) FAILED (log: $LOGFILE)"
        fi
    done
done

if [[ "$FAILED" -gt 0 ]]; then
    echo "$FAILED of $TOTAL_TASKS method-vs-method runs failed" >&2
    exit 1
fi
echo "All $TOTAL_TASKS method-vs-method runs completed."
