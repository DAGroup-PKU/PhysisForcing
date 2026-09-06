#!/bin/bash
# Training launcher for PhysisForcing (pf-wan), full fine-tuning.
#
# Usage (run from the project root):
#   bash tools/launch_1node.sh [config_file] [nproc_per_node] [master_port]
#
# Examples:
#   bash tools/launch_1node.sh                                                    # ours, 8 GPUs, port 29500
#   bash tools/launch_1node.sh configs/train/full_ft/train_full_ft_ours.jsonc 4 29501
#
# Multi-node: run this on every node with a shared MASTER_ADDR/MASTER_PORT and a
# per-node NODE_RANK, e.g.
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=<host0> bash tools/launch_1node.sh
#   NNODES=2 NODE_RANK=1 MASTER_ADDR=<host0> bash tools/launch_1node.sh

set -euo pipefail
set -x

CONFIG_FILE=${1:-configs/train/full_ft/train_full_ft_ours.jsonc}
NPROC_PER_NODE=${2:-8}
MASTER_PORT=${3:-29500}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="${PROJECT_DIR:-$(dirname "${SCRIPT_DIR}")}"
TRAIN_SCRIPT=tools/main.py

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}

echo "============================================"
echo "PhysisForcing (pf-wan) training launcher"
echo "  Project:      ${PROJECT_DIR}"
echo "  Config:       ${CONFIG_FILE}"
echo "  GPUs/node:    ${NPROC_PER_NODE}"
echo "  Nodes:        ${NNODES} (rank ${NODE_RANK})"
echo "  Master:       ${MASTER_ADDR}:${MASTER_PORT}"
echo "============================================"

cd "${PROJECT_DIR}"

if [ ! -d "./weights" ]; then
  echo "[ERROR] './weights' not found under PROJECT_DIR=${PROJECT_DIR}"
  echo "        Expected layout:"
  echo "          weights/Wan2.2-I2V-A14B/        (high_noise_model/, models_t5_*.pth, Wan2.1_VAE.pth, google/umt5-xxl, ...)"
  echo "          weights/vjepa2-vitl-fpc64-256/  (config.json, model.safetensors, ...)"
  echo "          weights/depth_anything_v2/      (depth_anything_v2_vitl.pth)"
  echo "          weights/cotracker/              (hubconf.py, weights/scaled_offline.pth, ...)"
  echo "        Either copy weights into ./weights, or symlink them: ln -s /path/to/weights ./weights"
  exit 1
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
ulimit -n 1024768 || true

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-31}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_API_KEY=${WANDB_API_KEY:-}

# Optional: auto-activate a conda env by setting CONDA_ENV (and optionally
# CONDA_ROOT). Otherwise, activate your Python environment before running.
if [[ -n "${CONDA_ENV:-}" ]]; then
  source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

mkdir -p logs
job_name=$(basename "${CONFIG_FILE}" .jsonc)
now=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/${job_name}_${now}.log"
echo "[INFO] log file: ${LOG_FILE}"

torchrun \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --nnodes="${NNODES}" \
  --node-rank="${NODE_RANK}" \
  --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  "${TRAIN_SCRIPT}" \
  --config-file "${CONFIG_FILE}" \
  2>&1 | tee "${LOG_FILE}"
