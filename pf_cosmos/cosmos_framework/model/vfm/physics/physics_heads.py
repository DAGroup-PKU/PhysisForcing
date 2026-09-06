# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Trainable projection heads ``ψ`` (high-level) and ``φ`` (low-level).

Both operate on a hidden token sequence ``[B, L, D]`` plus an explicit patch
grid ``(F_p, H_p, W_p)``: ``ψ`` projects to the teacher hidden dim and resizes
to the teacher token grid; ``φ`` projects to a compact dim and reshapes back to
the patch grid for the trajectory readout.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HighLevelPhysicsHead(nn.Module):
    """Trainable projection ``ψ`` from MoT hidden dim to V-JEPA2 hidden dim.

    Architecture: ``LayerNorm -> Linear -> GELU -> Linear -> LayerNorm``.
    """

    def __init__(self, dit_dim: int, target_dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or target_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(dit_dim),
            nn.Linear(dit_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, target_dim),
            nn.LayerNorm(target_dim),
        )
        self.dit_dim = dit_dim
        self.target_dim = target_dim

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """(Re)initialize all parameters (also called after meta-device materialization)."""
        for m in self.proj:
            if isinstance(m, nn.LayerNorm):
                m.reset_parameters()
        nn.init.normal_(self.proj[1].weight, std=0.02)
        nn.init.zeros_(self.proj[1].bias)
        nn.init.normal_(self.proj[3].weight, std=0.02)
        nn.init.zeros_(self.proj[3].bias)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        dit_grid: Optional[Tuple[int, int, int]] = None,
        target_grid: Optional[Tuple[int, int, int]] = None,
        valid_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Run the full project + reshape + resize pipeline.

        When called as a plain projection (``dit_grid is None and target_grid is
        None``) this simply returns ``ψ(hidden)`` with shape ``[B, L, D_target]``.
        When ``dit_grid`` and ``target_grid`` are both provided, it additionally
        reshapes the token sequence to ``[B, F_p, H_p, W_p, D_target]`` and
        trilinearly resizes the spatial-temporal grid to ``target_grid``,
        returning the flattened ``[B, T_tok*H_tok*W_tok, D_target]`` tensor
        expected by the relational loss.
        """
        proj = self.proj(hidden)
        if dit_grid is None and target_grid is None:
            return proj
        assert dit_grid is not None and target_grid is not None, (
            "forward(...) needs both dit_grid and target_grid for the alignment pipeline."
        )
        grid = self.reshape_token_seq_to_grid(proj, dit_grid, valid_len=valid_len)
        resized = self.resize_to_token_grid(grid, target_grid)  # [B, D, Tt, Ht, Wt]
        B, D, T_t, H_t, W_t = resized.shape
        return resized.permute(0, 2, 3, 4, 1).reshape(B, T_t * H_t * W_t, D)

    @staticmethod
    def reshape_token_seq_to_grid(
        tokens: torch.Tensor,
        grid_sizes: Tuple[int, int, int],
        valid_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Reshape ``[B, L, D]`` -> ``[B, D, F_p, H_p, W_p]``.

        ``valid_len`` must equal ``F_p * H_p * W_p``; any trailing padding is
        sliced off before reshaping.
        """
        B, L, D = tokens.shape
        F_p, H_p, W_p = grid_sizes
        N = F_p * H_p * W_p
        if valid_len is None:
            valid_len = N
        assert valid_len == N, f"valid_len {valid_len} != F_p*H_p*W_p {N}"
        if L > valid_len:
            tokens = tokens[:, :valid_len, :]
        elif L < valid_len:
            raise ValueError(f"Sequence too short to reshape: got L={L}, expected at least {valid_len}")
        x = tokens.view(B, F_p, H_p, W_p, D).permute(0, 4, 1, 2, 3).contiguous()
        return x

    @staticmethod
    def resize_to_token_grid(
        feat: torch.Tensor,
        target_grid: Tuple[int, int, int],
    ) -> torch.Tensor:
        """Trilinear resize from ``[B, D, F_p, H_p, W_p]`` to ``[B, D, T_tok, H_tok, W_tok]``."""
        B, D, F_p, H_p, W_p = feat.shape
        T_t, H_t, W_t = target_grid
        if (F_p, H_p, W_p) == (T_t, H_t, W_t):
            return feat
        # F.interpolate trilinear needs fp32 for stability.
        out = F.interpolate(
            feat.float(),
            size=(T_t, H_t, W_t),
            mode="trilinear",
            align_corners=False,
        )
        return out.to(feat.dtype)


class LowLevelPhysicsHead(nn.Module):
    """Trainable projection ``φ`` from MoT hidden dim to a compact feature dim
    used for the trajectory-alignment ``grid_sample`` similarity (Sec. 3.2,
    eq. 5-7).

    Architecture mirrors :class:`HighLevelPhysicsHead`. The forward pipeline
    reshapes the projected token sequence back to its native patch grid
    ``[B, D_low, F_p, H_p, W_p]`` so the per-frame ``H_p × W_p`` feature maps can
    serve as query/key features for the soft-argmax against tracked points.
    """

    def __init__(self, dit_dim: int, low_dim: int = 128, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or low_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(dit_dim),
            nn.Linear(dit_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, low_dim),
            nn.LayerNorm(low_dim),
        )
        self.dit_dim = dit_dim
        self.low_dim = low_dim

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """(Re)initialize all parameters (see :meth:`HighLevelPhysicsHead.reset_parameters`)."""
        for m in self.proj:
            if isinstance(m, nn.LayerNorm):
                m.reset_parameters()
        nn.init.normal_(self.proj[1].weight, std=0.02)
        nn.init.zeros_(self.proj[1].bias)
        nn.init.normal_(self.proj[3].weight, std=0.02)
        nn.init.zeros_(self.proj[3].bias)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        dit_grid: Optional[Tuple[int, int, int]] = None,
        valid_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Project the hidden token sequence and (optionally) reshape it back to
        its native spatial-temporal grid.

        When ``dit_grid is None`` returns ``φ(hidden)`` with shape ``[B, L, D_low]``.
        When ``dit_grid`` is provided, reshapes to ``[B, D_low, F_p, H_p, W_p]``.
        """
        proj = self.proj(hidden)
        if dit_grid is None:
            return proj
        return HighLevelPhysicsHead.reshape_token_seq_to_grid(proj, dit_grid, valid_len=valid_len)
