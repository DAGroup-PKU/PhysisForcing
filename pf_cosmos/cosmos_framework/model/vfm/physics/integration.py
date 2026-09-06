# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Glue tying the physics heads, frozen teachers, and loss kernels together for
one packed Cosmos3 i2v training batch.

Public surface used by ``OmniMoTModel``: build-time helpers to attach / reset /
unfreeze the ``ψ`` / ``φ`` heads, a frozen-teacher builder, and
:func:`compute_physics_losses` which turns the captured projections + raw RGB
into the weighted high-/low-level losses.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cosmos_framework.model.vfm.physics.mot_capture import split_proj_by_sample
from cosmos_framework.model.vfm.physics.physics_heads import (
    HighLevelPhysicsHead,
    LowLevelPhysicsHead,
)
from cosmos_framework.model.vfm.physics.physics_mask import (
    build_physics_token_mask,
    select_aligned_tokens_with_shared_indices,
)
from cosmos_framework.model.vfm.physics.relational_loss import relational_alignment_loss
from cosmos_framework.model.vfm.physics.teachers import PhysicsTeachers
from cosmos_framework.model.vfm.physics.trajectory_loss import compute_cotracker_topk, track_loss

logger = logging.getLogger(__name__)

PSI_ATTR = "physics_psi_head"
PHI_ATTR = "physics_phi_head"


# ----------------------------------------------------------------------
# Build-time helpers
# ----------------------------------------------------------------------


def attach_physics_heads(net: nn.Module, cfg, dit_dim: int) -> None:
    """Attach ψ / φ heads onto ``net`` (a ``Cosmos3VFMNetwork``) on whatever
    device ``net`` currently lives on (meta, at build time).

    Only the heads needed by the enabled losses are created. ``dit_dim`` is the
    MoT hidden size (``net.hidden_size`` == 4096 for Nano).
    """
    if cfg.high_level.enabled:
        setattr(
            net,
            PSI_ATTR,
            HighLevelPhysicsHead(
                dit_dim=dit_dim,
                target_dim=int(cfg.high_level.target_dim),
                hidden_dim=int(cfg.high_level.psi_hidden_dim),
            ),
        )
    if cfg.low_level.enabled:
        setattr(
            net,
            PHI_ATTR,
            LowLevelPhysicsHead(
                dit_dim=dit_dim,
                low_dim=int(cfg.low_level.low_dim),
                hidden_dim=int(cfg.low_level.phi_hidden_dim),
            ),
        )
    logger.info(
        f"[physics] attached heads psi={cfg.high_level.enabled} phi={cfg.low_level.enabled} (dit_dim={dit_dim})"
    )


def reset_physics_heads(net: nn.Module) -> None:
    """Re-initialize attached head parameters after meta -> real materialization."""
    for attr in (PSI_ATTR, PHI_ATTR):
        head = getattr(net, attr, None)
        if head is not None:
            head.reset_parameters()


def set_physics_params_trainable(net: nn.Module) -> None:
    """Force ``requires_grad=True`` on physics-head params.

    LoRA injection freezes every non-``lora_`` parameter; this re-enables the
    heads so the optimizer's substring ``keys_to_select=["lora_", "physics_"]``
    actually trains them.
    """
    for name, p in net.named_parameters():
        if "physics_" in name:
            p.requires_grad_(True)


def build_physics_teachers(cfg) -> PhysicsTeachers:
    """Construct the lazy, frozen teacher holder from a physics config."""
    need_vjepa2 = bool(cfg.high_level.enabled)
    need_tracks = bool(cfg.low_level.enabled) or bool(cfg.high_level.enabled and cfg.high_level.use_region_mask)
    # Depth is used to weight track selection in BOTH the low-level loss and the
    # region mask; load it whenever either path wants it.
    use_depth = bool(cfg.low_level.enabled and cfg.low_level.use_depth_weighting) or bool(
        cfg.high_level.enabled and cfg.high_level.use_region_mask
    )
    return PhysicsTeachers(
        need_vjepa2=need_vjepa2,
        need_tracks=need_tracks,
        vjepa2_path=cfg.high_level.vjepa2_path,
        cotracker_repo=cfg.cotracker_repo,
        depth_weights_dir=cfg.depth_weights_dir,
        depth_encoder=cfg.depth_encoder,
        vjepa2_frames_per_clip=cfg.high_level.frames_per_clip,
        vjepa2_crop_size=cfg.high_level.crop_size,
        use_depth_weighting=use_depth,
    )


# ----------------------------------------------------------------------
# Per-step loss computation
# ----------------------------------------------------------------------


def _to_btchw_m1p1(v: torch.Tensor) -> torch.Tensor:
    """Normalize a raw-state vision tensor to ``[1, T, C, H, W]`` in [-1, 1]."""
    if v.dim() == 4:  # [C, T, H, W]
        v = v.unsqueeze(0)  # [1, C, T, H, W]
    assert v.dim() == 5, f"expected [B,C,T,H,W], got {tuple(v.shape)}"
    return v.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W]


def _resize_video_height(v_btchw: torch.Tensor, target_h: Optional[int]) -> torch.Tensor:
    """Bilinearly resize ``[1, T, C, H, W]`` to height ``target_h`` (aspect ratio
    preserved, even width).

    Only ever *downsamples* the clip the trackers see; returns the input
    unchanged when ``target_h`` is None or >= the native height (so setting a cap
    larger than the data resolution is a safe no-op, never an upscale). The
    trajectory loss and region mask normalize by H,W, so this is numerically
    safe — it just trades CoTracker/Depth precision for speed.
    """
    if target_h is None:
        return v_btchw
    _, T, C, H, W = v_btchw.shape
    target_h = int(target_h)
    if target_h <= 0 or target_h >= H:
        return v_btchw
    target_w = max(2, int(round(W * target_h / H / 2) * 2))
    x = v_btchw.reshape(T, C, H, W)
    x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return x.reshape(1, T, C, target_h, target_w)


def compute_physics_losses(
    *,
    cfg,
    teachers: PhysicsTeachers,
    psi_proj: Optional[torch.Tensor],
    phi_proj: Optional[torch.Tensor],
    vision,
    raw_state_vision: list[torch.Tensor],
    device: torch.device | str,
) -> torch.Tensor:
    """Compute the weighted high-/low-level physics losses for a packed batch.

    Parameters
    ----------
    cfg : PhysicsAlignmentConfig
    teachers : PhysicsTeachers
        Already ``ensure_loaded(device)``-ed by the caller.
    psi_proj, phi_proj : Optional[Tensor]
        ``[1, N_vis, D]`` projections captured by ``MoTPhysicsCapturer`` (None
        when the corresponding loss is disabled).
    vision : ModalityData
        ``packed_sequence.vision`` (``sequence_indexes`` + ``token_shapes``).
    raw_state_vision : list of per-sample RGB tensors ``[1, C, T, H, W]`` in [-1, 1].

    Returns
    -------
    total_loss
        The weighted sum of the enabled physics losses (autograd-live).
    """
    zero = torch.zeros((), device=device, dtype=torch.float32)

    hi, lo = cfg.high_level, cfg.low_level
    use_high = hi.enabled and psi_proj is not None and teachers.vjepa2 is not None
    use_low = lo.enabled and phi_proj is not None and teachers.cotracker is not None

    # Per-sample chunking (parameter-free; safe with variable per-rank counts).
    psi_chunks, dit_grids = split_proj_by_sample(psi_proj, vision) if use_high else (None, None)
    phi_chunks, dit_grids_lo = split_proj_by_sample(phi_proj, vision) if use_low else (None, None)
    if dit_grids is None:
        dit_grids = dit_grids_lo
    n_samples = len(dit_grids) if dit_grids is not None else 0
    assert n_samples == 0 or len(raw_state_vision) == n_samples, (
        f"raw_state_vision count {len(raw_state_vision)} != vision samples {n_samples}"
    )

    target_grid = teachers.vjepa2.token_grid_shape() if use_high else None

    low_raw, high_raw = [], []
    for i in range(n_samples):
        f_p, h_p, w_p = dit_grids[i]
        v = raw_state_vision[i].to(device=device).float()  # [1,C,T,H,W] in [-1,1]
        v_btchw = _to_btchw_m1p1(v)  # [1,T,C,H,W] in [-1,1]

        # The clip the trackers see (optionally down-scaled; V-JEPA2 keeps the
        # native clip below). Run CoTracker + depth + top-K selection ONCE and
        # share it across the low-level loss and the region mask whenever both
        # want identical tracks (same grid_size + depth setting, i.e. the default
        # recipe); otherwise each path falls back to computing its own.
        v_track = _resize_video_height(v_btchw, getattr(cfg, "track_input_height", None))
        need_mask_tracks = use_high and bool(hi.use_region_mask)
        shared_topk_tracks = None
        shared_topk_vis = None
        if (
            use_low
            and need_mask_tracks
            and int(lo.track_grid_size) == int(hi.track_grid_size)
            and bool(lo.use_depth_weighting) == (teachers.depth_anything is not None)
        ):
            video_255_track = ((v_track + 1.0) * 127.5).clamp(0, 255)  # [1,T,C,H,W]
            shared_topk_tracks, shared_topk_vis = compute_cotracker_topk(
                video_255_track,
                cotracker=teachers.cotracker,
                depth_anything=teachers.depth_anything,
                track_grid_size=int(lo.track_grid_size),
                use_depth_weighting=bool(lo.use_depth_weighting),
            )

        # ---- Low-level trajectory alignment ----
        if use_low:
            # ``reshape_token_seq_to_grid`` is a staticmethod defined on
            # ``HighLevelPhysicsHead`` (shared by both heads).
            feat = HighLevelPhysicsHead.reshape_token_seq_to_grid(
                phi_chunks[i].float(), (f_p, h_p, w_p)
            )  # [1, D_low, F_p, H_p, W_p]
            video_255_tchw = ((v_track + 1.0) * 127.5).clamp(0, 255)[0]  # [T, C, H, W]
            low_raw.append(
                track_loss(
                    video_255_tchw,
                    feat,
                    cotracker=teachers.cotracker,
                    depth_anything=teachers.depth_anything if lo.use_depth_weighting else None,
                    track_grid_size=int(lo.track_grid_size),
                    frame_group_size=int(lo.frame_group_size),
                    use_depth_weighting=bool(lo.use_depth_weighting),
                    precomputed_topk_tracks=shared_topk_tracks,
                )
            )

        # ---- High-level relational alignment ----
        if use_high:
            teacher_tokens = teachers.vjepa2.encode_video_m1p1(v_btchw)  # [1, N_jepa, D]
            grid = HighLevelPhysicsHead.reshape_token_seq_to_grid(
                psi_chunks[i].float(), (f_p, h_p, w_p)
            )  # [1, target_dim, F_p, H_p, W_p]
            resized = HighLevelPhysicsHead.resize_to_token_grid(grid, target_grid)  # [1, D, Tt, Ht, Wt]
            B, D, T_t, H_t, W_t = resized.shape
            student_tokens = resized.permute(0, 2, 3, 4, 1).reshape(B, T_t * H_t * W_t, D)  # [1, N_jepa, D]

            token_mask = build_physics_token_mask(
                v_track,
                target_grid=target_grid,
                cotracker=teachers.cotracker,
                depth_anything=teachers.depth_anything,
                use_region_mask=bool(hi.use_region_mask),
                track_grid_size=int(hi.track_grid_size),
                mask_radius=int(hi.mask_radius),
                precomputed_topk_tracks=shared_topk_tracks,
                precomputed_topk_visibility=shared_topk_vis,
            )  # [1, N_jepa]

            student_sel, teacher_sel = select_aligned_tokens_with_shared_indices(
                student_tokens=student_tokens,
                teacher_tokens=teacher_tokens.float(),
                mask=token_mask,
                max_K=int(hi.max_K) if hi.max_K is not None else None,
                min_K=int(hi.min_K),
            )
            high_raw.append(
                relational_alignment_loss(
                    student_sel,
                    teacher_sel,
                    norm=str(hi.norm),
                    temperature=float(hi.temperature),
                )["loss"]
            )

    total = zero
    if low_raw:
        total = total + torch.stack(low_raw).mean() * float(lo.weight)
    if high_raw:
        total = total + torch.stack(high_raw).mean() * float(hi.weight)

    return total
