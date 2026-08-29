#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash ablation/run_style_shift.sh [--seed 42] [--games 40] [--switch-after 20]
#       [--phase dev_debug] [--model deepseek-v4-flash] [--no-save-prompts]

SEED=42
GAMES=40
SWITCH_AFTER=20
PHASE="dev_debug"
MODEL="deepseek-v4-flash"
SAVE_PROMPTS=1
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed|--start-seed)
            SEED="$2"
            shift 2
            ;;
        --games)
            GAMES="$2"
            shift 2
            ;;
        --switch-after)
            SWITCH_AFTER="$2"
            shift 2
            ;;
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --model|--model-name)
            MODEL="$2"
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

COMMON_ARGS=(
    --games "$GAMES"
    --switch-after "$SWITCH_AFTER"
    --start-seed "$SEED"
    --phase "$PHASE"
    --model-name "$MODEL"
)

if [[ "$SAVE_PROMPTS" -eq 0 ]]; then
    COMMON_ARGS+=(--no-save-prompts)
fi

LOG_DIR="results/style_shift_logs"
mkdir -p "$LOG_DIR"
RUN_TAG="$(date +%Y%m%d_%H%M%S)_seed${SEED}_g${GAMES}_switch${SWITCH_AFTER}"

echo "Style-shift benchmark"
echo "  seed=$SEED games=$GAMES switch_after=$SWITCH_AFTER phase=$PHASE model=$MODEL"
echo "  prompts=$([[ "$SAVE_PROMPTS" -eq 1 ]] && echo yes || echo no)"
echo "  log_dir=$LOG_DIR/$RUN_TAG"
echo

mkdir -p "$LOG_DIR/$RUN_TAG"

PIDS=()

launch() {
    local game="$1"
    local condition="$2"
    local log_file="$LOG_DIR/$RUN_TAG/${game}_${condition}.log"
    local cmd=(
        python runs/style_shift/style_shift_bench.py
        --game "$game"
        --condition "$condition"
        "${COMMON_ARGS[@]}"
    )
    if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
        cmd+=("${EXTRA_ARGS[@]}")
    fi
    echo "launch $game / $condition -> $log_file"
    "${cmd[@]}" >"$log_file" 2>&1 &
    PIDS+=("$!")
}

for game in leduc liars_dice goofspiel; do
    for condition in full cfr simple; do
        launch "$game" "$condition"
    done
done

launch leduc suspicion

FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAIL=1
    fi
done

if [[ "$FAIL" -ne 0 ]]; then
    echo "One or more style-shift jobs failed. Logs: $LOG_DIR/$RUN_TAG" >&2
    exit 1
fi

echo "All style-shift jobs completed. Logs: $LOG_DIR/$RUN_TAG"
