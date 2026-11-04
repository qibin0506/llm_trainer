#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Default parameters
NPROC=${NPROC:-2}
STAGE=${STAGE:-pretrain}
ZERO=${ZERO:-2}
CHUNKED=${CHUNKED:-""}
RL_MODE=${RL_MODE:-single}
EPOCHS=${EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-16}

while [[ $# -gt 0 ]]; do
  case $1 in
    --nproc)
      NPROC="$2"
      shift 2
      ;;
    --stage)
      STAGE="$2"
      shift 2
      ;;
    --zero)
      ZERO="$2"
      shift 2
      ;;
    --chunked)
      CHUNKED="--chunked"
      shift
      ;;
    --rl-mode)
      RL_MODE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --init-weights)
      INIT_WEIGHTS="$2"
      shift 2
      ;;
    --ref-weights)
      REF_WEIGHTS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

export STAGE="${STAGE}"
export ZERO="${ZERO}"
export CHUNKED="${CHUNKED:-0}"
export RL_MODE="${RL_MODE}"
export EPOCHS="${EPOCHS}"
export BATCH_SIZE="${BATCH_SIZE}"
export PYTHONUNBUFFERED=1
if [ -n "${INIT_WEIGHTS}" ]; then export INIT_WEIGHTS; fi
if [ -n "${REF_WEIGHTS}" ]; then export REF_WEIGHTS; fi

echo "=========================================================="
echo " Launching Training via torchrun"
echo " GPUs (nproc_per_node): ${NPROC}"
echo " Stage:                ${STAGE}"
echo " ZeRO Stage:           ${ZERO}"
echo " Chunked:              ${CHUNKED:-false}"
echo " RL Mode:              ${RL_MODE}"
echo "=========================================================="

torchrun --nproc_per_node="${NPROC}" "${SCRIPT_DIR}/main.py"

