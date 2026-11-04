#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Default parameters
NPROC=${NPROC:-2}
BATCH_SIZE=${BATCH_SIZE:-32}
EPOCHS=${EPOCHS:-1}

while [[ $# -gt 0 ]]; do
  case $1 in
    --nproc)
      NPROC="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: bash $0 [--nproc N] [--batch-size B] [--epochs E]"
      echo "Example: bash $0 --nproc 2 --batch-size 32 --epochs 1"
      exit 0
      ;;
    [0-9]*)
      NPROC="$1"
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

RL_BATCH_SIZE=$((BATCH_SIZE > 16 ? 16 : BATCH_SIZE))

echo "=========================================================="
echo " Running Full 28-Combination Verification Matrix"
echo " Stages:               Pretrain (4) / SFT (4) / DPO (4) / GRPO (8) / PPO (8)"
echo " Dimensions:           ZeRO-2 & ZeRO-3 | Chunked & Non-Chunked | Single-Turn & Multi-Turn RL"
echo " Launcher:             torchrun"
echo " Number of GPUs:       ${NPROC}"
echo " Batch Size:           ${BATCH_SIZE} (RL: ${RL_BATCH_SIZE})"
echo " Epochs per Stage:     ${EPOCHS}"
echo "=========================================================="

if [ ! -f "data/pretrain.jsonl" ] || [ ! -f "checkpoints/base_model.pt" ]; then
  echo ""
  echo ">>> [0/28] Preparing Shakespeare dataset and base model checkpoint..."
  python3 prepare_data.py
fi

# ==================== 1. Pretrain (4 Combinations) ====================
echo ""
echo ">>> [1/28] Test Pretrain (ZeRO-2, Non-Chunked)"
bash run_multi_gpu.sh --nproc "${NPROC}" --stage pretrain --zero 2 --batch-size "${BATCH_SIZE}" --epochs "${EPOCHS}"

echo ""
echo ">>> [2/28] Test Pretrain (ZeRO-2, Chunked)"
bash run_multi_gpu.sh --nproc "${NPROC}" --stage pretrain --zero 2 --chunked --batch-size "${BATCH_SIZE}" --epochs "${EPOCHS}"

echo ""
echo ">>> [3/28] Test Pretrain (ZeRO-3, Non-Chunked)"
bash run_multi_gpu.sh --nproc "${NPROC}" --stage pretrain --zero 3 --batch-size "${BATCH_SIZE}" --epochs "${EPOCHS}"

echo ""
echo ">>> [4/28] Test Pretrain (ZeRO-3, Chunked)"
bash run_multi_gpu.sh --nproc "${NPROC}" --stage pretrain --zero 3 --chunked --batch-size "${BATCH_SIZE}" --epochs "${EPOCHS}"

# ==================== 2. SFT (4 Combinations) ====================
echo ""
echo ">>> [5/28] Test SFT (ZeRO-2, Non-Chunked)"
bash run_multi_gpu.sh --nproc "${NPROC}" --stage sft --zero 2 --batch-size "${BATCH_SIZE}" --epochs "${EPOCHS}"

echo ""
echo ">>> [6/28] Test SFT (ZeRO-2, Chunked)"
bash run_multi_gpu.sh --nproc "${NPROC}" --stage sft --zero 2 --chunked --batch-size "${BATCH_SIZE}" --epochs "${EPOCHS}"

echo ""
echo ">>> [7/28] Test SFT (ZeRO-3, Non-Chunked)"
bash run_multi_gpu.sh --nproc "${NPROC}" --stage sft --zero 3 --batch-size "${BATCH_SIZE}" --epochs "${EPOCHS}"

echo ""
echo ">>> [8/28] Test SFT (ZeRO-3, Chunked)"
bash run_multi_gpu.sh --nproc "${NPROC}" --stage sft --zero 3 --chunked --batch-size "${BATCH_SIZE}" --epochs "${EPOCHS}"

# ==================== 3. DPO (4 Combinations) ====================
echo ""
echo ">>> [9/28] Test DPO (ZeRO-2, Non-Chunked)"
bash run_multi_gpu.sh --nproc "${NPROC}" --stage dpo --zero 2 --batch-size "${BATCH_SIZE}" --epochs "${EPOCHS}"

echo ""
echo ">>> [10/28] Test DPO (ZeRO-2, Chunked)"
bash run_multi_gpu.sh --nproc "${NPROC}" --stage dpo --zero 2 --chunked --batch-size "${BATCH_SIZE}" --epochs "${EPOCHS}"

echo ""
echo ">>> [11/28] Test DPO (ZeRO-3, Non-Chunked)"
bash run_multi_gpu.sh --nproc "${NPROC}" --stage dpo --zero 3 --batch-size "${BATCH_SIZE}" --epochs "${EPOCHS}"

echo ""
echo ">>> [12/28] Test DPO (ZeRO-3, Chunked)"
bash run_multi_gpu.sh --nproc "${NPROC}" --stage dpo --zero 3 --chunked --batch-size "${BATCH_SIZE}" --epochs "${EPOCHS}"

# ==================== 4. GRPO (8 Combinations) ====================
echo ""
echo ">>> [13/28] Test GRPO (ZeRO-2, Single-Turn RL, Non-Chunked)"
bash run_multi_gpu.sh --nproc "${NPROC}" --stage grpo --zero 2 --rl-mode single --batch-size "${RL_BATCH_SIZE}" --epochs "${EPOCHS}"

echo ""
echo ">>> [14/28] Test GRPO (ZeRO-2, Single-Turn RL, Chunked)"
bash run_multi_gpu.sh --nproc "${NPROC}" --stage grpo --zero 2 --rl-mode single --chunked --batch-size "${RL_BATCH_SIZE}" --epochs "${EPOCHS}"

echo ""
echo ">>> [15/28] Test GRPO (ZeRO-2, Multi-Turn RL, Non-Chunked)"
bash run_multi_gpu.sh --nproc "${NPROC}" --stage grpo --zero 2 --rl-mode multi --batch-size "${RL_BATCH_SIZE}" --epochs "${EPOCHS}"

echo ""
echo ">>> [16/28] Test GRPO (ZeRO-2, Multi-Turn RL, Chunked)"
bash run_multi_gpu.sh --nproc "${NPROC}" --stage grpo --zero 2 --rl-mode multi --chunked --batch-size "${RL_BATCH_SIZE}" --epochs "${EPOCHS}"

echo ""
echo ">>> [17/28] Test GRPO (ZeRO-3, Single-Turn RL, Non-Chunked)"
bash run_multi_gpu.sh --nproc "${NPROC}" --stage grpo --zero 3 --rl-mode single --batch-size "${RL_BATCH_SIZE}" --epochs "${EPOCHS}"

echo ""
echo ">>> [18/28] Test GRPO (ZeRO-3, Single-Turn RL, Chunked)"
bash run_multi_gpu.sh --nproc "${NPROC}" --stage grpo --zero 3 --rl-mode single --chunked --batch-size "${RL_BATCH_SIZE}" --epochs "${EPOCHS}"

echo ""
echo ">>> [19/28] Test GRPO (ZeRO-3, Multi-Turn RL, Non-Chunked)"
bash run_multi_gpu.sh --nproc "${NPROC}" --stage grpo --zero 3 --rl-mode multi --batch-size "${RL_BATCH_SIZE}" --epochs "${EPOCHS}"

echo ""
echo ">>> [20/28] Test GRPO (ZeRO-3, Multi-Turn RL, Chunked)"
bash run_multi_gpu.sh --nproc "${NPROC}" --stage grpo --zero 3 --rl-mode multi --chunked --batch-size "${RL_BATCH_SIZE}" --epochs "${EPOCHS}"

# ==================== 5. PPO (8 Combinations) ====================
echo ""
echo ">>> [21/28] Test PPO (ZeRO-2, Single-Turn RL, Non-Chunked)"
bash run_multi_gpu.sh --nproc "${NPROC}" --stage ppo --zero 2 --rl-mode single --batch-size "${RL_BATCH_SIZE}" --epochs "${EPOCHS}"

echo ""
echo ">>> [22/28] Test PPO (ZeRO-2, Single-Turn RL, Chunked)"
bash run_multi_gpu.sh --nproc "${NPROC}" --stage ppo --zero 2 --rl-mode single --chunked --batch-size "${RL_BATCH_SIZE}" --epochs "${EPOCHS}"

echo ""
echo ">>> [23/28] Test PPO (ZeRO-2, Multi-Turn RL, Non-Chunked)"
bash run_multi_gpu.sh --nproc "${NPROC}" --stage ppo --zero 2 --rl-mode multi --batch-size "${RL_BATCH_SIZE}" --epochs "${EPOCHS}"

echo ""
echo ">>> [24/28] Test PPO (ZeRO-2, Multi-Turn RL, Chunked)"
bash run_multi_gpu.sh --nproc "${NPROC}" --stage ppo --zero 2 --rl-mode multi --chunked --batch-size "${RL_BATCH_SIZE}" --epochs "${EPOCHS}"

echo ""
echo ">>> [25/28] Test PPO (ZeRO-3, Single-Turn RL, Non-Chunked)"
bash run_multi_gpu.sh --nproc "${NPROC}" --stage ppo --zero 3 --rl-mode single --batch-size "${RL_BATCH_SIZE}" --epochs "${EPOCHS}"

echo ""
echo ">>> [26/28] Test PPO (ZeRO-3, Single-Turn RL, Chunked)"
bash run_multi_gpu.sh --nproc "${NPROC}" --stage ppo --zero 3 --rl-mode single --chunked --batch-size "${RL_BATCH_SIZE}" --epochs "${EPOCHS}"

echo ""
echo ">>> [27/28] Test PPO (ZeRO-3, Multi-Turn RL, Non-Chunked)"
bash run_multi_gpu.sh --nproc "${NPROC}" --stage ppo --zero 3 --rl-mode multi --batch-size "${RL_BATCH_SIZE}" --epochs "${EPOCHS}"

echo ""
echo ">>> [28/28] Test PPO (ZeRO-3, Multi-Turn RL, Chunked)"
bash run_multi_gpu.sh --nproc "${NPROC}" --stage ppo --zero 3 --rl-mode multi --chunked --batch-size "${RL_BATCH_SIZE}" --epochs "${EPOCHS}"

echo ""
echo ""
echo "=========================================================="
echo " All 28 test configurations succeeded! Running Final Comparison..."
echo "=========================================================="

python3 eval.py --compare

echo ""
echo "=========================================================="
echo " Full Verification and Evaluation Completed Successfully!"
echo "=========================================================="
