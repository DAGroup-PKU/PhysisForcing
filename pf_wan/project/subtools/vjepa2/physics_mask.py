from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


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
        point. The default of 1 produces a 3x3 mark per track point,
        which is a touch more forgiving than the strict single-pixel
        rasterization in eq. (4).

    Returns
    -------
    torch.Tensor
        ``[B, T, H, W]`` binary float mask.
    """
    assert tracks.dim() == 4 and tracks.size(-1) == 2, (
        f"expected [B,T,K,2], got {tuple(tracks.shape)}"
    )
    B, T, K, _ = tracks.shape
    if K == 0:
        return tracks.new_zeros((B, T, H, W))

    device = tracks.device
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

    A token is considered active if the fraction of active pixels inside
    it is strictly greater than ``threshold``. The default
    ``threshold = 0.0`` means any pixel inside a tubelet/patch flips the
    whole token to 1, which matches the spirit of eq. (10): the token is
    selected whenever it overlaps an interaction region.
    """
    B, T, H, W = mask.shape
    T_t, H_t, W_t = target_grid
    x = mask.unsqueeze(1).float()  # [B, 1, T, H, W]
    down = F.adaptive_avg_pool3d(x, output_size=(T_t, H_t, W_t)).squeeze(1)
    return (down > threshold).to(mask.dtype)


def select_tokens_with_mask(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    *,
    max_K: Optional[int] = None,
    min_K: int = 2,
) -> torch.Tensor:
    """Slice tokens ``[B, N, D]`` with a flattened token mask ``[B, N]``.

    Because different batch elements may have different numbers of
    active tokens, we run this *per batch element* and concatenate along
    a synthetic "K" dim, returning a ragged-friendly representation of
    shape ``[B, K_eff, D]`` where ``K_eff`` is the largest count in the
    batch and shorter samples are padded by repeating their own tokens
    (so the cosine relation matrix stays meaningful — repeats produce a
    diagonal block of 1s which is identical between student/teacher and
    therefore contributes zero to the L1 loss).

    Parameters
    ----------
    tokens : torch.Tensor
        ``[B, N, D]``.
    mask : torch.Tensor
        ``[B, N]`` binary token mask (same N).
    max_K : Optional[int]
        Hard cap on the number of selected tokens per sample. Useful to
        bound the memory of the ``K x K`` relation matrix.
    min_K : int
        If a batch element has fewer than ``min_K`` active tokens, we
        skip masking and use all tokens. Returning at least two tokens
        ensures the cosine matrix is non-degenerate.

    Returns
    -------
    torch.Tensor
        ``[B, K_eff, D]``.
    """
    assert tokens.dim() == 3, f"expected [B,N,D], got {tuple(tokens.shape)}"
    assert mask.dim() == 2 and mask.size(0) == tokens.size(0)
    B, N, D = tokens.shape
    out_lists = []
    counts = []
    for b in range(B):
        idx = mask[b].nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() < min_K:
            idx = torch.arange(N, device=tokens.device)
        if max_K is not None and idx.numel() > max_K:
            perm = torch.randperm(idx.numel(), device=tokens.device)[:max_K]
            idx = idx[perm]
        out_lists.append(tokens[b].index_select(0, idx))
        counts.append(idx.numel())
    K_eff = max(counts)
    padded = []
    for sel in out_lists:
        if sel.size(0) < K_eff:
            # Pad by repeating the last token. Repeats are identical
            # across student/teacher branches, so they contribute exactly
            # zero to the L1 loss.
            n_missing = K_eff - sel.size(0)
            pad = sel[-1:].expand(n_missing, -1)
            sel = torch.cat([sel, pad], dim=0)
        padded.append(sel)
    return torch.stack(padded, dim=0)


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
    ``[B, N, D_*]`` with the same ``N``. We pick one index set per batch
    element (so the two relation matrices are over the same token
    positions) and return the gathered features.

    Returns
    -------
    (student_sel, teacher_sel) : tuple of torch.Tensor
        Both with shape ``[B, K_eff, D_*]``.
    """
    assert student_tokens.size(0) == teacher_tokens.size(0)
    assert student_tokens.size(1) == teacher_tokens.size(1) == mask.size(1)
    B, N = mask.shape

    # Fixed K_eff plus deterministic stride sampling keep the output
    # shape identical across ranks.
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
            # No useful mask -> fall back to ``arange`` over the whole
            # token sequence (then sub-sample if N > K_eff). This
            # behaves identically on every rank because N is constant.
            if N >= K_eff:
                sel = torch.linspace(0, N - 1, steps=K_eff, device=mask.device).round().long()
                idx = sel
            else:
                # Token grid smaller than K_eff: pad with last index.
                base = torch.arange(N, device=mask.device)
                pad_idx = base[-1:].expand(K_eff - N)
                idx = torch.cat([base, pad_idx], dim=0)

        s_out.append(student_tokens[b].index_select(0, idx))
        t_out.append(teacher_tokens[b].index_select(0, idx))

    student_sel = torch.stack(s_out, dim=0)  # [B, K_eff, D_s]
    teacher_sel = torch.stack(t_out, dim=0)  # [B, K_eff, D_t]
    return student_sel, teacher_sel
