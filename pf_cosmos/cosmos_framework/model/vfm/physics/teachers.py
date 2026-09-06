# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Frozen teacher models for the physics-alignment losses.

CoTracker3 (trajectories), DepthAnythingV2 (first-frame depth weighting) and
V-JEPA2 (relational teacher) are held in a plain container (deliberately not an
``nn.Module``) so their frozen weights are never registered as trainable
parameters, never enter the checkpoint, and are never FSDP-sharded. They are
loaded once, on the training device, in eval / no-grad mode.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_DEPTH_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def load_depth_anything(weights_dir: str, encoder: str = "vitl", device: str = "cuda"):
    """Build DepthAnythingV2 and load the ``vitl`` checkpoint (frozen, eval)."""
    from cosmos_framework.model.vfm.physics.third_party.depth_anything_v2.dpt import DepthAnythingV2

    ckpt = os.path.join(weights_dir, f"depth_anything_v2_{encoder}.pth")
    model = DepthAnythingV2(**_DEPTH_MODEL_CONFIGS[encoder])
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    logger.info(f"[physics] DepthAnythingV2({encoder}) loaded from {ckpt}")
    return model


def load_cotracker(repo_dir: str, device: str = "cuda"):
    """Load CoTracker3 offline predictor from a local torch.hub repo (frozen)."""
    # torch.hub.load(source="local") adds repo_dir to sys.path and is CWD-sensitive;
    # resolve to an absolute path so a repo-relative config value still works.
    repo_dir = os.path.abspath(repo_dir)
    model = torch.hub.load(repo_dir, "cotracker3_offline", source="local")
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    logger.info(f"[physics] CoTracker3 (offline) loaded from {repo_dir}")
    return model


class PhysicsTeachers:
    """Lazy, frozen holder for the physics teacher models.

    Not an ``nn.Module`` on purpose — see module docstring. Teachers are loaded
    on first :meth:`ensure_loaded` call (after the device is known) so that
    model construction / checkpoint load is not slowed down by ~7 GB of frozen
    weights, and so a config with a loss disabled skips loading its teacher.
    """

    def __init__(
        self,
        *,
        need_vjepa2: bool,
        need_tracks: bool,
        vjepa2_path: str,
        cotracker_repo: str,
        depth_weights_dir: str,
        depth_encoder: str = "vitl",
        vjepa2_frames_per_clip: Optional[int] = None,
        vjepa2_crop_size: Optional[int] = None,
        use_depth_weighting: bool = True,
        vjepa2_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.need_vjepa2 = need_vjepa2
        self.need_tracks = need_tracks
        self.vjepa2_path = vjepa2_path
        self.cotracker_repo = cotracker_repo
        self.depth_weights_dir = depth_weights_dir
        self.depth_encoder = depth_encoder
        self.vjepa2_frames_per_clip = vjepa2_frames_per_clip
        self.vjepa2_crop_size = vjepa2_crop_size
        self.use_depth_weighting = use_depth_weighting
        self.vjepa2_dtype = vjepa2_dtype

        self.vjepa2 = None
        self.cotracker = None
        self.depth_anything = None
        self._loaded = False

    def ensure_loaded(self, device: str | torch.device) -> "PhysicsTeachers":
        if self._loaded:
            return self
        device = str(device)

        if self.need_vjepa2:
            from cosmos_framework.model.vfm.physics.vjepa2_wrapper import VJEPA2FeatureExtractor

            self.vjepa2 = VJEPA2FeatureExtractor(
                model_name_or_path=self.vjepa2_path,
                frames_per_clip=self.vjepa2_frames_per_clip,
                crop_size=self.vjepa2_crop_size,
                dtype=self.vjepa2_dtype,
            ).to(device)
            self.vjepa2.eval()

        if self.need_tracks:
            self.cotracker = load_cotracker(self.cotracker_repo, device=device)
            if self.use_depth_weighting:
                self.depth_anything = load_depth_anything(
                    self.depth_weights_dir, encoder=self.depth_encoder, device=device
                )

        self._loaded = True
        return self
