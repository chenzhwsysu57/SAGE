#!/usr/bin/env bash
set -euo pipefail

# Run Goofspiel strategic-reasoning benchmarks in parallel, one persona per job.
# Usage:
#   bash ablation/run_goofspiel_strategic_reasoning_parallel.sh \
#       [--personas all|Greedy,Conservative,...] [--games 10] [--start-seed 9120] \
#       [--n 5] [--phase dev_debug] [--conditions vanilla,llm_eq,llm_tot,llm_pomcts] \
#       [--model-name deepseek-v4-flash] [--base-url https://api.deepseek.com] \
#       [--cfr-model path/to.pkl] [--no-save-prompts] [--no-save-trajectories]

PERSONAS="all"
CONDITIONS="vanilla,llm_eq,llm_tot,llm_pomcts"
GAMES=10
START_SEED=9120
N=5
PHASE="dev_debug"
MODEL_NAME="deepseek-v4-flash"
BASE_URL="https://api.deepseek.com"
CFR_MODEL=""
SAVE_PROMPTS=1
SAVE_TRAJECTORIES=1
VERBOSE=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --personas)
            PERSONAS="$2"
            shift 2
            ;;
        --conditions)
            CONDITIONS="$2"
            shift 2
            ;;
        --games)
            GAMES="$2"
            shift 2
            ;;
        --start-seed|--seed)
            START_SEED="$2"
            shift 2
            ;;
        --n)
            N="$2"
            shift 2
            ;;
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --model-name|--model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --base-url)
            BASE_URL="$2"
            shift 2
            ;;
        --cfr-model)
            CFR_MODEL="$2"
            shift 2
            ;;
        --no-save-prompts)
            SAVE_PROMPTS=0
            shift
            ;;
        --no-save-trajectories)
            SAVE_TRAJECTORIES=0
            shift
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ "$PERSONAS" == "all" ]]; then
    PERSONA_LIST=(Greedy Conservative Sacrificial Random GTO Mirror)
else
    IFS=',' read -r -a PERSONA_LIST <<< "$PERSONAS"
fi

if [[ "${#PERSONA_LIST[@]}" -eq 0 ]]; then
    echo "No personas selected." >&2
    exit 1
fi

RUN_TAG="$(date +%Y%m%d_%H%M%S)_${GAMES}g_parallel_personas"
LOG_DIR="results/goofspiel/strategic_reasoning_logs/$RUN_TAG"
OUTPUT_ROOT="results/goofspiel/strategic_reasoning_parallel/$RUN_TAG"
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_ROOT"

COMMON_ARGS=(
    --conditions "$CONDITIONS"
    --games "$GAMES"
    --start-seed "$START_SEED"
    --n "$N"
    --phase "$PHASE"
    --model-name "$MODEL_NAME"
    --base-url "$BASE_URL"
)

if [[ -n "$CFR_MODEL" ]]; then
    COMMON_ARGS+=(--cfr-model "$CFR_MODEL")
fi
if [[ "$SAVE_PROMPTS" -eq 0 ]]; then
    COMMON_ARGS+=(--no-save-prompts)
fi
if [[ "$SAVE_TRAJECTORIES" -eq 0 ]]; then
    COMMON_ARGS+=(--no-save-trajectories)
fi
if [[ "$VERBOSE" -eq 1 ]]; then
    COMMON_ARGS+=(--verbose)
fi
if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
    COMMON_ARGS+=("${EXTRA_ARGS[@]}")
fi

echo "Goofspiel strategic-reasoning parallel launcher"
echo "  personas=${PERSONA_LIST[*]}"
echo "  conditions=$CONDITIONS"
echo "  games=$GAMES start_seed=$START_SEED n=$N phase=$PHASE"
echo "  model=$MODEL_NAME base_url=$BASE_URL"
echo "  logs=$LOG_DIR"
echo "  output_root=$OUTPUT_ROOT"

PIDS=()
NAMES=()

launch_persona() {
    local persona="$1"
    local log_file="$LOG_DIR/${persona}.log"
    local output_dir="$OUTPUT_ROOT/$persona"
    local cmd=(
        python runs/goofspiel/strategic_reasoning_bench.py
        --personas "$persona"
        --output-dir "$output_dir"
        "${COMMON_ARGS[@]}"
    )
    echo "launch $persona -> $log_file (output: $output_dir)"
    "${cmd[@]}" >"$log_file" 2>&1 &
    PIDS+=("$!")
    NAMES+=("$persona")
}

for persona in "${PERSONA_LIST[@]}"; do
    launch_persona "$persona"
done

FAIL=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    name="${NAMES[$i]}"
    if wait "$pid"; then
        echo "done: $name"
    else
        echo "failed: $name (pid=$pid)" >&2
        FAIL=1
    fi
done

if [[ "$FAIL" -ne 0 ]]; then
    echo "One or more persona jobs failed. Logs: $LOG_DIR" >&2
    exit 1
fi

echo "All persona jobs completed. Logs: $LOG_DIR"
