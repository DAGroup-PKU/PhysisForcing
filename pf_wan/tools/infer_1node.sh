#!/bin/bash
# Single-node inference launcher for PF_Wan (Wan2.2-A14B I2V) generation.
#
# Usage (run from the project root, or from anywhere -- it will auto-cd):
#   bash tools/infer_1node.sh [config_file] [nproc_per_node] [master_port]
#
# Examples:
#   bash tools/infer_1node.sh                                            # default (PF_Wan), 8 GPUs, port 29510
#   bash tools/infer_1node.sh configs/generate/pf_wan_i2v.jsonc 4 29511  # 4 GPUs, custom port
#
# Notes:
# - FSDP / sequence-parallel inference is configured through the .jsonc config
#   (ulysses_size / ring_size) together with the env vars set below.
# - The launcher cd's into PROJECT_DIR before running, so paths to
#   configs / tools/main.py are still given relative to that root.

set -euo pipefail
set -x

# -------------------------
# Args
# -------------------------
CONFIG_FILE=${1:-configs/generate/pf_wan_i2v.jsonc}
NPROC_PER_NODE=${2:-8}
MASTER_PORT=${3:-29510}

# -------------------------
# Project paths / scripts
# -------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="${PROJECT_DIR:-$(dirname "${SCRIPT_DIR}")}"
INFER_SCRIPT=tools/main.py

MASTER_ADDR=127.0.0.1
NNODES=1
NODE_RANK=0

echo "============================================"
echo "PF_Wan Single-node Inference Launcher"
echo "  Project:        ${PROJECT_DIR}"
echo "  Config:         ${CONFIG_FILE}"
echo "  GPUs:           ${NPROC_PER_NODE}"
echo "  Master Port:    ${MASTER_PORT}"
echo "============================================"

cd "${PROJECT_DIR}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "[ERROR] Config file not found: ${CONFIG_FILE}" >&2
    exit 1
fi

# -------------------------
# Sanity checks
# -------------------------
# All weight paths in the config are relative to PROJECT_DIR. PF_Wan ships as a
# single bundle under ./checkpoints/PF_Wan/ (backbone + T5 + VAE + tokenizer).
# Fail fast with a clear message if the bundle is missing.
if [ ! -d "./checkpoints/PF_Wan" ]; then
  echo "[ERROR] './checkpoints/PF_Wan' not found under PROJECT_DIR=${PROJECT_DIR}"
  echo "        PF_Wan ships as a single weight bundle. Expected layout:"
  echo "          checkpoints/PF_Wan/backbone.pth                     (PF_Wan DiT)"
  echo "          checkpoints/PF_Wan/models_t5_umt5-xxl-enc-bf16.pth  (T5 text encoder)"
  echo "          checkpoints/PF_Wan/Wan2.1_VAE.pth                   (VAE)"
  echo "          checkpoints/PF_Wan/google/umt5-xxl/                 (tokenizer)"
  echo "        Download it from Hugging Face (DAGroup-PKU/PF_Wan), or symlink an existing copy:"
  echo "          ln -s /path/to/PF_Wan ./checkpoints/PF_Wan"
  exit 1
fi

# -------------------------
# Environment setup
# -------------------------
ulimit -n 1024768
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export NCCL_IB_DISABLE=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_IB_TIMEOUT=31
export TOKENIZERS_PARALLELISM=false

mkdir -p logs

job_name=$(basename "${CONFIG_FILE}" .jsonc)
now=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/infer_${job_name}_1node_${now}.log"
echo "[INFO] log file: ${LOG_FILE}"

# -------------------------
# Conda env (optional)
# -------------------------
# Activate a conda env only if CONDA_ROOT and CONDA_ENV are set, e.g.:
#   export CONDA_ROOT="$HOME/miniconda3"
#   export CONDA_ENV="physisforcing"
if [[ -n "${CONDA_ROOT:-}" && -n "${CONDA_ENV:-}" ]]; then
  conda deactivate 2>/dev/null || true
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

# -------------------------
# Launch
# -------------------------
echo "[INFO] Launching torchrun with:"
echo "  NNODES=${NNODES}"
echo "  NODE_RANK=${NODE_RANK}"
echo "  MASTER_ADDR=${MASTER_ADDR}"
echo "  MASTER_PORT=${MASTER_PORT}"
echo "  NPROC_PER_NODE=${NPROC_PER_NODE}"

torchrun \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --nnodes="${NNODES}" \
  --node-rank="${NODE_RANK}" \
  --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  "${INFER_SCRIPT}" \
  --config-file "${CONFIG_FILE}" \
  2>&1 | tee "${LOG_FILE}"
