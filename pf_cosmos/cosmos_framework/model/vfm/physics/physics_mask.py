# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Physics region focus — the token mask ``M`` (Sec. 3.1).

Rasterize the selected tracks into a pixel mask, downsample it to the teacher
token grid, and select the same token positions on both the student and the
teacher side for the relational loss.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def project_tracks_to_pixel_mask(
    tracks: torch.Tensor,
    visibility: Optional[torch.Tensor],
    *,
    H: int,
    W: int,
    radius: int = 1,
) -> torch.Tensor:
    """Eq. (4): rasterize a set of tracks into a binary pixel mask.

    Parameters
    ----------
    tracks : torch.Tensor
        ``[B, T, K, 2]`` track coordinates in pixel space (x, y).
    visibility : Optional[torch.Tensor]
        ``[B, T, K]`` track visibility (1 = visible). If given, invisible
        track points are skipped. Pass ``None`` to ignore visibility.
    H, W : int
        Pixel resolution of the video that produced ``tracks``.
    radius : int
        Optional spatial dilation radius (in pixels) around each track
        point. The default of 1 produces a 3x3 mark per track point.

    Returns
    -------
    torch.Tensor
        ``[B, T, H, W]`` binary float mask.
    """
    assert tracks.dim() == 4 and tracks.size(-1) == 2, f"expected [B,T,K,2], got {tuple(tracks.shape)}"
    B, T, K, _ = tracks.shape
    if K == 0:
        return tracks.new_zeros((B, T, H, W))

    xs = tracks[..., 0].round().long().clamp(0, W - 1)  # [B,T,K]
    ys = tracks[..., 1].round().long().clamp(0, H - 1)  # [B,T,K]
    flat = ys * W + xs  # [B,T,K]

    mask_flat = tracks.new_zeros((B, T, H * W))
    if visibility is not None:
        vis = visibility.to(mask_flat.dtype)
    else:
        vis = torch.ones_like(flat, dtype=mask_flat.dtype)

    # Scatter-add 1.0 at each (t, ys, xs).
    mask_flat.scatter_add_(2, flat, vis)
    mask = (mask_flat > 0).to(mask_flat.dtype).view(B, T, H, W)

    if radius and radius > 0:
        k = 2 * radius + 1
        mask = F.max_pool2d(mask.view(B * T, 1, H, W), kernel_size=k, stride=1, padding=radius)
        mask = mask.view(B, T, H, W)
    return mask


def downsample_mask_to_grid(
    mask: torch.Tensor,
    target_grid: tuple,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Downsample a ``[B, T, H, W]`` pixel mask to a ``[B, Tt, Ht, Wt]``
    token-level binary mask using average pooling.

    A token is considered active if the fraction of active pixels inside it is
    strictly greater than ``threshold``. The default ``threshold = 0.0`` means
    any pixel inside a tubelet/patch flips the whole token to 1, matching the
    spirit of eq. (10).
    """
    B, T, H, W = mask.shape
    T_t, H_t, W_t = target_grid
    x = mask.unsqueeze(1).float()  # [B, 1, T, H, W]
    down = F.adaptive_avg_pool3d(x, output_size=(T_t, H_t, W_t)).squeeze(1)
    return (down > threshold).to(mask.dtype)


def select_aligned_tokens_with_shared_indices(
    student_tokens: torch.Tensor,
    teacher_tokens: torch.Tensor,
    mask: torch.Tensor,
    *,
    max_K: Optional[int] = None,
    min_K: int = 2,
) -> tuple:
    """Apply the *same* token selection to student and teacher.

    Both feature tensors must already be in the same canonical layout
    ``[B, N, D_*]`` with the same ``N``. We pick one index set per batch element
    (so the two relation matrices are over the same token positions) and return
    the gathered features.

    Fixed ``K_eff`` plus deterministic stride sampling keep the output shape
    identical across ranks.

    Returns
    -------
    (student_sel, teacher_sel) : tuple of torch.Tensor
        Both with shape ``[B, K_eff, D_*]``.
    """
    assert student_tokens.size(0) == teacher_tokens.size(0)
    assert student_tokens.size(1) == teacher_tokens.size(1) == mask.size(1)
    B, N = mask.shape

    if max_K is not None:
        K_eff = int(min(max_K, N))
    else:
        K_eff = int(N)
    K_eff = max(K_eff, int(min_K))

    s_out, t_out = [], []
    for b in range(B):
        idx_pos = mask[b].nonzero(as_tuple=False).squeeze(-1)
        n_pos = int(idx_pos.numel())

        if n_pos >= K_eff:
            sel = torch.linspace(0, n_pos - 1, steps=K_eff, device=mask.device).round().long()
            idx = idx_pos[sel]
        elif n_pos >= int(min_K):
            pad_idx = idx_pos[-1:].expand(K_eff - n_pos)
            idx = torch.cat([idx_pos, pad_idx], dim=0)
        else:
            # No useful mask -> fall back to ``arange`` over the whole token
            # sequence (then sub-sample if N > K_eff). Behaves identically on
            # every rank because N is constant.
            if N >= K_eff:
                sel = torch.linspace(0, N - 1, steps=K_eff, device=mask.device).round().long()
                idx = sel
            else:
                base = torch.arange(N, device=mask.device)
                pad_idx = base[-1:].expand(K_eff - N)
                idx = torch.cat([base, pad_idx], dim=0)

        s_out.append(student_tokens[b].index_select(0, idx))
        t_out.append(teacher_tokens[b].index_select(0, idx))

    student_sel = torch.stack(s_out, dim=0)  # [B, K_eff, D_s]
    teacher_sel = torch.stack(t_out, dim=0)  # [B, K_eff, D_t]
    return student_sel, teacher_sel


@torch.no_grad()
def build_physics_token_mask(
    gt_video_btchw: torch.Tensor,
    target_grid: tuple,
    *,
    cotracker=None,
    depth_anything=None,
    use_region_mask: bool = True,
    track_grid_size: int = 25,
    mask_radius: int = 2,
    precomputed_topk_tracks: Optional[torch.Tensor] = None,
    precomputed_topk_visibility: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the physics-informative token mask M for a single video sample.

    Parameters
    ----------
    gt_video_btchw : torch.Tensor
        ``[1, T, 3, H, W]`` in [-1, 1].
    target_grid : (T_tok, H_tok, W_tok)
        V-JEPA2 token grid.
    cotracker, depth_anything :
        Frozen teacher models (``depth_anything`` may be ``None``).
    use_region_mask : bool
        When False, returns an all-ones mask (align over everything).
    track_grid_size, mask_radius : int
        CoTracker query grid size and pixel dilation radius for rasterization.

    Returns
    -------
    torch.Tensor
        ``[1, T_tok * H_tok * W_tok]`` binary mask.
    """
    # Local import avoids any import cycle at module load time.
    from cosmos_framework.model.vfm.physics.trajectory_loss import select_topk_active_tracks

    T_tok, H_tok, W_tok = target_grid
    N_tok = T_tok * H_tok * W_tok

    # If the user disabled region focus, align over everything.
    if not use_region_mask:
        return torch.ones((1, N_tok), device=gt_video_btchw.device, dtype=torch.float32)

    H_pix = gt_video_btchw.shape[3]
    W_pix = gt_video_btchw.shape[4]

    # Steps 1-3 (CoTracker -> depth-weighted top-K) are identical to the
    # low-level loss's, so the caller can run them once and pass the result in.
    if precomputed_topk_tracks is not None:
        topk_tracks = precomputed_topk_tracks
        topk_visibility = precomputed_topk_visibility
    else:
        if cotracker is None:
            logger.warning("[high_level_loss] CoTracker not available, falling back to uniform mask")
            return torch.ones((1, N_tok), device=gt_video_btchw.device, dtype=torch.float32)

        # Convert to the [0, 255] format cotracker expects.
        video_pix = ((gt_video_btchw + 1.0) * 127.5).clamp(0, 255)  # [1, T, 3, H, W]

        # 1) CoTracker3 -> dense trajectories.
        video_tracks, video_visibility = cotracker(
            video_pix,
            grid_size=int(track_grid_size),
            grid_query_frame=0,
            backward_tracking=None,
        )

        # 2) First-frame depth weighting.
        depth_norm = None
        if depth_anything is not None:
            import cv2
            import numpy as np

            first_frame = video_pix[0, 0].permute(1, 2, 0).contiguous().cpu().numpy()
            # DepthAnythingV2 expects BGR uint8.
            first_bgr = cv2.cvtColor(first_frame.astype("uint8"), cv2.COLOR_RGB2BGR)
            depth = depth_anything.infer_image(first_bgr)
            if depth.shape != (H_pix, W_pix):
                depth = cv2.resize(depth, (W_pix, H_pix), interpolation=cv2.INTER_LINEAR)
            d_min, d_max = float(depth.min()), float(depth.max())
            depth_norm = (depth - d_min) / max(d_max - d_min, 1e-6)
            depth_norm = np.clip(depth_norm, 0.0, 1.0)

        # 3) Select top-K active tracks (eqs. 1-3).
        topk_tracks, topk_visibility = select_topk_active_tracks(
            video_tracks, video_visibility, K=None, depth_norm=depth_norm
        )

    # 4) Rasterize to pixel mask (eq. 4).
    pixel_mask = project_tracks_to_pixel_mask(
        topk_tracks.to(device=gt_video_btchw.device),
        topk_visibility.to(device=gt_video_btchw.device),
        H=H_pix,
        W=W_pix,
        radius=int(mask_radius),
    )  # [1, T, H, W]

    # 5) Downsample to V-JEPA2 token grid and flatten.
    token_mask = downsample_mask_to_grid(pixel_mask, target_grid=target_grid)  # [1, T_tok, H_tok, W_tok]
    return token_mask.view(1, -1)
