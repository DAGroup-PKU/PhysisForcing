"""DiT-side projection heads and hook utilities for the two physics
alignment objectives.

Both losses operate on the *same* hidden feature ``H^l`` extracted from
the ``l``-th DiT block (see Sec. 3.2 / 3.3 of the paper). They differ in
which lightweight MLP refines that feature before the loss is computed:

    F_hat^u = Resize( ψ(H^l) )                                  (eq. 9)
    F_hat   =        φ(H^l)                                     (eq. 5)

``H^l`` for the Wan DiT has shape ``[B, L, D_dit]`` with
``L = F_p * H_p * W_p`` where ``(F_p, H_p, W_p)`` is the latent-time
patch grid (after the ``patch_size=(1, 2, 2)`` Conv3D embedding) for the
specific clip.

Three pieces live here:

* :class:`HighLevelPhysicsHead` — projection MLP ``ψ`` for the
  high-level (V-JEPA2) relational alignment loss; reshapes + trilinearly
  resamples to the V-JEPA2 token grid ``(T_tok, H_tok, W_tok)``.
* :class:`LowLevelPhysicsHead` — projection MLP ``φ`` for the low-level
  (CoTracker3) trajectory alignment loss; reshapes back to
  ``[B, D_low, F_p, H_p, W_p]`` so the per-frame feature maps can be
  used as Q/K for ``grid_sample`` at the tracked points.
* :class:`DiTBlockFeatureCapturer` — a small context manager that
  registers a forward hook on the chosen DiT block and stashes its
  output for the duration of one forward pass.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Projection head ψ(·)
# ----------------------------------------------------------------------


class HighLevelPhysicsHead(nn.Module):
    """Trainable projection ``ψ`` from DiT hidden dim to V-JEPA2 hidden dim.

    Architecture: ``LayerNorm -> Linear -> GELU -> Linear -> LayerNorm``
    matching the lightweight projection style used by Wan's existing
    ``MLPProj`` (see ``project/models/backbone/dit.py``).
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

        When called as a plain projection (``dit_grid is None and
        target_grid is None``) this returns ``ψ(hidden)`` with shape
        ``[B, L, D_target]``. When ``dit_grid`` and ``target_grid`` are
        both provided, it reshapes the token sequence to
        ``[B, F_p, H_p, W_p, D_target]``, trilinearly resizes to
        ``target_grid`` and returns the flattened
        ``[B, T_tok*H_tok*W_tok, D_target]`` tensor expected by the
        relational loss.
        """
        proj = self.proj(hidden)
        if dit_grid is None and target_grid is None:
            return proj
        assert dit_grid is not None and target_grid is not None, (
            "forward(...) needs both dit_grid and target_grid for the "
            "alignment pipeline."
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

        The Wan DiT zero-pads token sequences to a common ``seq_len``
        across the batch (see ``WanModel.forward`` in
        ``project/models/backbone/dit.py``). When that is the case,
        ``valid_len`` must equal ``F_p * H_p * W_p`` and we slice the
        unused padding off before reshaping.
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
            raise ValueError(
                f"Sequence too short to reshape: got L={L}, expected at least {valid_len}"
            )
        x = tokens.view(B, F_p, H_p, W_p, D).permute(0, 4, 1, 2, 3).contiguous()
        return x

    @staticmethod
    def resize_to_token_grid(
        feat: torch.Tensor,
        target_grid: Tuple[int, int, int],
    ) -> torch.Tensor:
        """Trilinear resize from ``[B, D, F_p, H_p, W_p]`` to
        ``[B, D, T_tok, H_tok, W_tok]``."""
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


# ----------------------------------------------------------------------
# Projection head φ(·) for low-level trajectory alignment
# ----------------------------------------------------------------------


class LowLevelPhysicsHead(nn.Module):
    """Trainable projection ``φ`` from DiT hidden dim to a compact
    feature dim used for the trajectory-alignment ``grid_sample``
    similarity (Sec. 3.2, eq. 5-7).

    Architecture mirrors :class:`HighLevelPhysicsHead` (lightweight
    LayerNorm-Linear-GELU-Linear-LayerNorm MLP) so the two refiners are
    symmetric and both qualify as "simple MLP" per the paper.

    The forward pipeline for low-level loss differs from the high-level
    one: instead of trilinearly resampling to the V-JEPA2 token grid, we
    reshape the projected token sequence back to its native DiT patch
    grid ``[B, D_low, F_p, H_p, W_p]`` so the per-frame ``H_p × W_p``
    feature maps can serve as query/key features for the soft-argmax
    against tracked points.
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
        """Project the hidden token sequence and (optionally) reshape it
        back to its native spatial-temporal grid.

        When called as a plain projection (``dit_grid is None``) this
        returns ``φ(hidden)`` with shape ``[B, L, D_low]``. When
        ``dit_grid`` is provided, it reshapes to
        ``[B, D_low, F_p, H_p, W_p]`` (slicing off the DiT's zero
        padding via ``valid_len`` if necessary) — the format the
        trajectory-alignment loss consumes for ``grid_sample`` queries.
        """
        proj = self.proj(hidden)
        if dit_grid is None:
            return proj
        return HighLevelPhysicsHead.reshape_token_seq_to_grid(
            proj, dit_grid, valid_len=valid_len
        )


# ----------------------------------------------------------------------
# Forward hook for DiT block output
# ----------------------------------------------------------------------


class DiTBlockFeatureCapturer:
    """Register a forward hook on one DiT block to capture its output.

    Usage
    -----
    >>> capturer = DiTBlockFeatureCapturer(backbone, block_idx=20)
    >>> with capturer.capture():
    ...     pred = backbone(x=...)
    ...     h_l = capturer.feature   # [B, L, D_dit]

    Notes
    -----
    * We hook the block at ``backbone.blocks[block_idx]`` (or one level
      down if the backbone is wrapped). The hook indirection survives
      FSDP wrapping because FSDP delegates attribute access back to the
      wrapped module.
    * The captured tensor is the *output* of the block, i.e. the ``x``
      returned by ``WanAttentionBlock.forward``. Its gradient flows back
      to the trainable backbone weights.
    """

    def __init__(self, backbone: nn.Module, block_idx: int) -> None:
        self.backbone = backbone
        self.block_idx = block_idx
        self._handle = None
        self.feature: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------

    def _resolve_block(self) -> nn.Module:
        """Walk through possible wrapping layers to reach the actual
        ``blocks`` ModuleList."""
        candidates: List[nn.Module] = []
        m = self.backbone
        candidates.append(m)
        # some wrappers expose the module under `base_model.model`
        if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
            candidates.append(m.base_model.model)
        # FSDP wraps with `_fsdp_wrapped_module`
        for c in list(candidates):
            inner = getattr(c, "_fsdp_wrapped_module", None)
            if inner is not None:
                candidates.append(inner)
                if hasattr(inner, "base_model") and hasattr(inner.base_model, "model"):
                    candidates.append(inner.base_model.model)

        for cand in candidates:
            blocks = getattr(cand, "blocks", None)
            if blocks is not None:
                if self.block_idx < 0 or self.block_idx >= len(blocks):
                    raise IndexError(
                        f"block_idx={self.block_idx} out of range (num_blocks={len(blocks)})"
                    )
                return blocks[self.block_idx]
        raise AttributeError(
            "Could not resolve DiT blocks ModuleList on backbone. "
            "Expected `backbone.blocks` or `backbone.base_model.model.blocks`."
        )

    # ------------------------------------------------------------------

    @contextmanager
    def capture(self):
        block = self._resolve_block()

        def _hook(_module, _inputs, output):
            # WanAttentionBlock returns a single tensor (the block output).
            if isinstance(output, tuple):
                output = output[0]
            self.feature = output
            return None  # do not modify the forward output

        self._handle = block.register_forward_hook(_hook)
        self.feature = None
        try:
            yield self
        finally:
            self._handle.remove()
            self._handle = None
