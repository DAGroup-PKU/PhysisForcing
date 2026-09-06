#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for vision_sft_nano_physics — LoRA i2v SFT on
# Cosmos3-Nano PLUS the three physics-alignment objectives
# (high-level V-JEPA2 relational, low-level CoTracker3 trajectory, physics
# region focus). Drives cosmos_framework.scripts.train against
# examples/toml/sft_config/vision_sft_nano_physics.toml.
#
# Same env-var surface as launch_sft_vision_nano.sh (DATASET_PATH,
# BASE_CHECKPOINT_PATH, WAN_VAE_PATH, NPROC_PER_NODE, ...). The ONE extra
# prerequisite is the frozen teacher assets — vendor them once from your own
# shell BEFORE launching:
#
#   bash tools/physics/vendor_teacher_assets.sh
#
# That populates checkpoints/physics/{co-tracker,depth_anything_v2,vjepa2-*}
# which the recipe's [model.physics_alignment.*] paths point at (repo-relative;
# the launcher cd's to the repo root so they resolve).
#
# Quick smoke (catch crashes fast, single GPU):
#   NPROC_PER_NODE=1 bash examples/launch_sft_vision_nano_physics.sh -- trainer.max_iter=3 checkpoint.save_iter=3

TOML_FILE="examples/toml/sft_config/vision_sft_nano_physics.toml"
: "${DATASET_PATH:=examples/data/bridge-v2-subset-synthetic-captions/sft_dataset_bridge}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}"

EXTRA_DATASET_CHECK='[[ -f "$DATASET_PATH/train/video_dataset_file.jsonl" ]] || { echo "ERROR: missing $DATASET_PATH/train/video_dataset_file.jsonl" >&2; exit 1; }'

# Fail early with a clear message if the teacher assets were not vendored yet.
# Anchored to $WORKDIR because the launcher eval's this BEFORE `cd "$WORKDIR"`.
EXTRA_DATASET_CHECK+='
for _p in co-tracker/hubconf.py depth_anything_v2/depth_anything_v2_vitl.pth vjepa2-vitl-fpc64-256/model.safetensors; do
    [[ -e "$WORKDIR/checkpoints/physics/$_p" ]] || { echo "ERROR: missing physics teacher asset: checkpoints/physics/$_p (run: bash tools/physics/vendor_teacher_assets.sh)" >&2; exit 1; }
done'

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
