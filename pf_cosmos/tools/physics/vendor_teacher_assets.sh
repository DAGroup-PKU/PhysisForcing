#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Fetch the frozen physics-alignment teachers (all public) into checkpoints/physics/:
#   - V-JEPA2 ViT-L/fpc64-256  (relational teacher)  facebook/vjepa2-vitl-fpc64-256
#   - DepthAnythingV2-Large    (depth weighting)     depth-anything/Depth-Anything-V2-Large  [CC-BY-NC-4.0]
#   - CoTracker3 offline       (trajectory teacher)  github.com/facebookresearch/co-tracker
#
# The DepthAnythingV2 *source* is already vendored under
# cosmos_framework/model/vfm/physics/third_party/; this only fetches weights.
#
# Requires: huggingface-cli (pip install huggingface_hub), git.
# Override the destination with CKPT_DIR (default: checkpoints/physics).
set -euo pipefail

CKPT="${CKPT_DIR:-checkpoints/physics}"
mkdir -p "$CKPT"

echo "[vendor] (1/3) V-JEPA2 ViT-L/fpc64-256 ..."
huggingface-cli download facebook/vjepa2-vitl-fpc64-256 \
    config.json model.safetensors \
    --local-dir "$CKPT/vjepa2-vitl-fpc64-256"

echo "[vendor] (2/3) DepthAnythingV2-Large checkpoint ..."
huggingface-cli download depth-anything/Depth-Anything-V2-Large \
    depth_anything_v2_vitl.pth \
    --local-dir "$CKPT/depth_anything_v2"

echo "[vendor] (3/3) CoTracker3 offline hub repo ..."
if [ ! -d "$CKPT/co-tracker/.git" ]; then
    git clone --depth 1 https://github.com/facebookresearch/co-tracker "$CKPT/co-tracker"
fi
# The repo's hubconf loads cotracker3_offline weights from the public URL on
# first use; no manual checkpoint placement is required.

echo "[vendor] done. Layout under $CKPT:"
find "$CKPT" -maxdepth 2 -type f | sort | sed 's/^/  /'
