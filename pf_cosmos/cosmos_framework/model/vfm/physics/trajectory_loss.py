# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Low-level trajectory alignment loss (Sec. 3.2).

CoTracker3 selects the most physics-active tracks; the student's per-frame
feature grid is aligned to those trajectories with a spatial soft-argmax
readout. Tracks are compressed in time (factor 4 for the Wan2.2 VAE) to match
the model's latent-time patch grid.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def select_topk_active_tracks(
    tracks: torch.Tensor,
    visibilitys: torch.Tensor,
    K: Optional[int] = None,
    depth_norm: Optional[np.ndarray] = None,
):
    """Select the K most "physics-active" tracks (motion-magnitude, optionally
    depth-weighted toward the foreground).

    Parameters
    ----------
    tracks : ``[B, T, N, 2]``
    visibilitys : ``[B, T, N]``
    K : int or None
        If None, keep all tracks whose motion magnitude is at or above the
        mean (average) magnitude threshold (paper Eq. 3).
    depth_norm : Optional[np.ndarray]
        ``[H, W]`` normalized depth (0..1) sampled at each track's frame-0
        position to weight its motion score.

    Returns
    -------
    (``[B, T, K, 2]``, ``[B, T, K]``)
    """

    def sample_depth_weights(depth_norm: np.ndarray, track_positions: torch.Tensor):
        H, W = depth_norm.shape
        x_coords = track_positions[:, 0].clamp(0, W - 1).long().cpu().numpy()
        y_coords = track_positions[:, 1].clamp(0, H - 1).long().cpu().numpy()
        depth_weights = depth_norm[y_coords, x_coords]  # (N,)
        return torch.from_numpy(depth_weights).to(track_positions.device)

    B, T, N, _ = tracks.shape
    topk_tracks_list = []
    topk_visibility_list = []
    for b in range(B):
        track = tracks[b]  # [T, N, 2]
        visibility = visibilitys[b]  # [T, N]
        diffs = track[1:] - track[:-1]  # [T-1, N, 2]
        dists = torch.norm(diffs, dim=2)  # [T-1, N]
        motion_magnitude = dists.sum(dim=0)  # [N]

        if depth_norm is not None:
            depth_weights = sample_depth_weights(depth_norm, track[0])  # [N]
            motion_magnitude = motion_magnitude * depth_weights.to(motion_magnitude.dtype)

        K_b = K
        if K_b is None:
            # Adaptive threshold: mean motion magnitude (paper Eq. 3).
            threshold = motion_magnitude.mean()
            above_threshold_indices = torch.where(motion_magnitude >= threshold)[0]
            K_b = len(above_threshold_indices)

        K_b = max(int(K_b), 0)
        K_b = min(K_b, N)
        if K_b == 0:
            topk_tracks_list.append(track[:, :0, :])
            topk_visibility_list.append(visibility[:, :0])
            continue

        _, topk_indices = torch.topk(motion_magnitude, K_b)
        topk_tracks_list.append(track[:, topk_indices, :])  # [T, K, 2]
        topk_visibility_list.append(visibility[:, topk_indices])  # [T, K]
    return torch.stack(topk_tracks_list, dim=0), torch.stack(topk_visibility_list, dim=0)


@torch.no_grad()
def compute_cotracker_topk(
    video_btchw_255: torch.Tensor,
    *,
    cotracker,
    depth_anything=None,
    track_grid_size: int = 25,
    use_depth_weighting: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run CoTracker3 (+ optional first-frame depth weighting) once and return
    the top-K most physics-active tracks, shared by the low-level trajectory
    loss and the high-level region mask (pass as ``precomputed_topk_tracks`` to
    :func:`track_loss` and :func:`build_physics_token_mask`).

    Parameters
    ----------
    video_btchw_255 : torch.Tensor
        ``[1, T, C, H, W]`` in [0, 255] (the resolution at which the tracks live;
        downstream coordinates are normalized by this H, W).

    Returns
    -------
    (topk_tracks ``[1, T, K, 2]``, topk_visibility ``[1, T, K]``)
    """
    video_tracks, video_visibility = cotracker(
        video_btchw_255,
        grid_size=int(track_grid_size),
        grid_query_frame=0,
        backward_tracking=None,
    )

    depth_norm = None
    if use_depth_weighting and depth_anything is not None:
        import cv2

        H, W = int(video_btchw_255.shape[3]), int(video_btchw_255.shape[4])
        first_frame_rgb = video_btchw_255[0, 0].permute(1, 2, 0).contiguous().cpu().numpy()  # (H,W,3)
        first_frame_bgr = cv2.cvtColor(first_frame_rgb.astype("uint8"), cv2.COLOR_RGB2BGR)
        depth = depth_anything.infer_image(first_frame_bgr)  # (H, W) raw depth
        if depth.shape != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
        d_min, d_max = float(depth.min()), float(depth.max())
        depth_norm = np.clip((depth - d_min) / max(d_max - d_min, 1e-6), 0.0, 1.0)

    return select_topk_active_tracks(video_tracks, video_visibility, K=None, depth_norm=depth_norm)


def track_loss(
    video: torch.Tensor,
    input_feature: torch.Tensor,
    *,
    cotracker=None,
    depth_anything=None,
    track_grid_size: int = 25,
    frame_group_size: int = 4,
    use_depth_weighting: bool = True,
    precomputed_topk_tracks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Low-level trajectory alignment loss L^{phy}_{low} (eq. 5-8).

    Parameters
    ----------
    video : torch.Tensor
        Ground-truth video ``[T, C, H_pix, W_pix]`` in [0, 255].
    input_feature : torch.Tensor
        Refined student feature already in spatial-temporal grid layout, shape
        ``[1, D_low, F_p, H_p, W_p]``. The caller is expected to have run the
        ``LowLevelPhysicsHead`` so gradients flow back to the backbone through
        the same ``H^l`` the high-level loss uses.
    cotracker, depth_anything :
        Frozen teacher models (``depth_anything`` may be ``None``).
    track_grid_size : int
        CoTracker3 query grid size.
    frame_group_size : int
        Number of pixel frames the VAE groups into one latent-time step (4 for
        Wan2.2). Used to temporally compress the pixel-frame tracks onto the
        model's latent-time patch grid.
    """
    device = input_feature.device
    video = video.to(device=device, dtype=input_feature.dtype)
    video = video.unsqueeze(0)  # [1, T, C, H, W]

    # Tracks are detached targets (CoTracker is frozen, ``video`` has no grad).
    # When the caller already ran CoTracker once (shared with the region mask),
    # reuse those top-K tracks; otherwise compute them here.
    if precomputed_topk_tracks is not None:
        topk_tracks = precomputed_topk_tracks.to(device=device)
    else:
        assert cotracker is not None, "track_loss needs a cotracker or precomputed_topk_tracks"
        topk_tracks, _ = compute_cotracker_topk(
            video,
            cotracker=cotracker,
            depth_anything=depth_anything,
            track_grid_size=track_grid_size,
            use_depth_weighting=use_depth_weighting,
        )

    def normalize_coords(tracks, H, W):
        x = tracks[..., 0] / (W - 1) * 2 - 1
        y = tracks[..., 1] / (H - 1) * 2 - 1
        return torch.stack([x, y], dim=-1)

    def temporal_compress(x, frame_group_size):
        B, T, K, C = x.shape
        first_frame = x[:, :1]  # [B, 1, K, 2]
        remaining_frames = x[:, 1:]  # [B, T-1, K, 2]
        grouped_frames = remaining_frames.reshape(B, -1, frame_group_size, K, C)
        averaged_frames = grouped_frames.mean(dim=2)
        return torch.cat([first_frame, averaged_frames], dim=1)

    assert input_feature.dim() == 5, f"track_loss expects [1, D_low, F_p, H_p, W_p], got {tuple(input_feature.shape)}"
    latent = input_feature

    # No usable tracks -> zero loss that still connects to ``latent`` (keeps
    # FSDP backward consistent across ranks).
    if topk_tracks.shape[2] == 0:
        return latent.sum() * 0.0

    # Trim/pad the CoTracker frames so temporal_compress yields exactly F_p
    # latent frames (needs (n_keep - 1) divisible by frame_group_size).
    F_p = latent.shape[2]
    n_keep = 1 + frame_group_size * (F_p - 1)
    T_pix_tracks = topk_tracks.shape[1]
    if T_pix_tracks >= n_keep:
        topk_tracks = topk_tracks[:, :n_keep]
    else:
        # Not enough tracked frames: repeat the last frame to reach n_keep.
        pad = topk_tracks[:, -1:].expand(-1, n_keep - T_pix_tracks, -1, -1)
        topk_tracks = torch.cat([topk_tracks, pad], dim=1)

    topk_tracks = temporal_compress(topk_tracks, frame_group_size)
    assert topk_tracks.shape[1] == F_p, (
        f"temporal align mismatch: got {topk_tracks.shape[1]} frames, expected F_p={F_p}"
    )

    H_pix, W_pix = video.shape[3], video.shape[4]
    norm_tracks = normalize_coords(topk_tracks, H=H_pix, W=W_pix)  # [B, T, K, 2]
    K = topk_tracks.shape[2]

    B, C, T, H, W = latent.shape

    # Frame-0 grid coordinates.
    norm_grid_0 = norm_tracks[:, 0, :, :].unsqueeze(2)  # [B, K, 1, 2]
    feat_frame0 = latent[:, :, 0, :, :]  # [B, C, H, W]
    query_feats = (
        F.grid_sample(feat_frame0, norm_grid_0, mode="bilinear", align_corners=True).squeeze(-1).permute(0, 2, 1)
    )  # [B, K, C]

    # 4. Similarity between query and all pixels of the remaining frames.
    other_feats = latent[:, :, 1:, :, :]  # [B, C, T-1, H, W]
    other_feats = other_feats.permute(0, 2, 1, 3, 4)  # [B, T-1, C, H, W]
    other_feats = other_feats.reshape(B, T - 1, C, H * W)  # [B, T-1, C, HW]

    sim = torch.einsum("bkc,btch->bkth", query_feats, other_feats)  # [B,K,T-1,HW]
    sim = sim.view(B, K, T - 1, H, W)  # [B,K,T-1,H,W]

    # 5. Spatial soft-argmax -> expected coordinate.
    sim_flat = sim.view(B, K, T - 1, -1)  # [B,K,T-1,HW]
    logits = sim_flat.float()
    logits = logits - logits.amax(dim=-1, keepdim=True)
    weights = F.softmax(logits, dim=-1).to(dtype=sim_flat.dtype)

    yy = torch.arange(H, device=sim.device, dtype=weights.dtype).view(H, 1).expand(H, W)
    xx = torch.arange(W, device=sim.device, dtype=weights.dtype).view(1, W).expand(H, W)
    yy_flat = yy.reshape(1, 1, 1, -1)
    xx_flat = xx.reshape(1, 1, 1, -1)

    exp_x = (weights * xx_flat).sum(dim=-1)
    exp_y = (weights * yy_flat).sum(dim=-1)
    pred_xy_norm = torch.stack([exp_x / W, exp_y / H], dim=-1)

    # 6. GT coordinates -> RMSD loss (both in [0, 1] normalized space).
    gt_xy = topk_tracks[:, 1:, :, :]  # [B,T-1,K,2]
    gt_xy = gt_xy.permute(0, 2, 1, 3)  # [B,K,T-1,2]
    scale = torch.tensor([1.0 / W_pix, 1.0 / H_pix], device=gt_xy.device, dtype=gt_xy.dtype)
    gt_xy_norm = gt_xy * scale

    # RMSD (sqrt of the coordinate MSE); eps keeps the gradient finite as MSE -> 0.
    mse = F.mse_loss(pred_xy_norm, gt_xy_norm)
    return torch.sqrt(mse + 1e-6)
