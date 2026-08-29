#!/usr/bin/env bash
set -euo pipefail

SEED=""
GAMES="${GAMES:-30}"
PHASE="${PHASE:-2}"
MODEL_NAME="${MODEL_NAME:-deepseek-v4-flash}"
BASE_URL="${BASE_URL:-https://api.deepseek.com}"

usage() {
  echo "Usage: bash ablation/run_length_placebo.sh --seed <start_seed> [--games 30] [--phase 2] [--model-name deepseek-v4-flash] [--base-url https://api.deepseek.com]"
}

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
    --phase)
      PHASE="$2"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${SEED}" ]]; then
  echo "Missing required argument: --seed"
  usage
  exit 1
fi

LEDUC_ARCHETYPES=(
  Maniac
  CallingStation
  LAG
  TAG
  Nit
  GTO
)

LD_ARCHETYPES=(
  Bluffer
  Honest
  AggressiveChallenger
  ConservativeChallenger
  Random
  GTO
)

GOOFSPIEL_PERSONAS=(
  Greedy
  Conservative
  Sacrificial
  Random
  GTO
  Mirror
)

echo "Running simple_placebo length-control experiments"
echo "  seed=${SEED}"
echo "  games=${GAMES}"
echo "  phase=${PHASE}"
echo "  model=${MODEL_NAME}"
echo "  base_url=${BASE_URL}"

for archetype in "${LEDUC_ARCHETYPES[@]}"; do
  python runs/leduc/leduc_tom_ablation.py \
    --condition simple_placebo \
    --archetype "${archetype}" \
    --phase "${PHASE}" \
    --games "${GAMES}" \
    --start-seed "${SEED}" \
    --model-name "${MODEL_NAME}" \
    --base-url "${BASE_URL}" \
    &
done

for archetype in "${LD_ARCHETYPES[@]}"; do
  python runs/liars_dice/ld_tom_ablation.py \
    --condition simple_placebo \
    --archetype "${archetype}" \
    --phase "${PHASE}" \
    --games "${GAMES}" \
    --start-seed "${SEED}" \
    --model-name "${MODEL_NAME}" \
    --base-url "${BASE_URL}" \
    &
done

for persona in "${GOOFSPIEL_PERSONAS[@]}"; do
  python runs/goofspiel/goofspiel_bench.py \
    --condition simple_placebo \
    --personas "${persona}" \
    --phase "${PHASE}" \
    --games "${GAMES}" \
    --start-seed "${SEED}" \
    --model-name "${MODEL_NAME}" \
    --base-url "${BASE_URL}" \
    &
done

wait
echo "All simple_placebo length-control experiments finished."
