# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Capture and project an intermediate MoT hidden feature.

Cosmos3's MoT runs a sequence-packed batch (many samples in one token stream).
A forward hook on decoder layer ``block_idx`` reconstructs the plain sequence
(``get_all_seq``), selects the vision tokens via ``vision.sequence_indexes``,
and runs the trainable ``ψ`` / ``φ`` heads on them inside ``net.forward`` (so
their FSDP-gathered weights are used). The captured projections keep their
autograd graph; all parameter-free work happens later in :mod:`integration`.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from cosmos_framework.data.vfm.sequence_packing import get_all_seq

logger = logging.getLogger(__name__)


def _resolve_decoder_layers(net: nn.Module) -> nn.ModuleList:
    """Walk the (possibly FSDP/AC/LoRA-wrapped) module tree to the MoT decoder
    ``layers`` ModuleList. Canonical path is ``net.language_model.model.layers``.
    """
    seeds: List[nn.Module] = [net]
    lm = getattr(net, "language_model", None)
    if lm is not None:
        seeds.append(lm)

    def _unwrap(m: nn.Module) -> List[nn.Module]:
        outs = [m]
        for attr in ("_fsdp_wrapped_module", "module", "_checkpoint_wrapped_module"):
            inner = getattr(m, attr, None)
            if isinstance(inner, nn.Module):
                outs.append(inner)
        base = getattr(m, "base_model", None)
        if base is not None and hasattr(base, "model"):
            outs.append(base.model)
        return outs

    visited: set[int] = set()
    frontier = list(seeds)
    while frontier:
        m = frontier.pop(0)
        if id(m) in visited:
            continue
        visited.add(id(m))
        for cand in _unwrap(m):
            layers = getattr(cand, "layers", None)
            if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                return layers
            inner_model = getattr(cand, "model", None)
            if isinstance(inner_model, nn.Module):
                inner_layers = getattr(inner_model, "layers", None)
                if isinstance(inner_layers, nn.ModuleList) and len(inner_layers) > 0:
                    return inner_layers
                frontier.append(inner_model)
            if id(cand) != id(m):
                frontier.append(cand)
    raise AttributeError("Could not resolve MoT decoder 'layers' ModuleList (net.language_model.model.layers).")


class MoTPhysicsCapturer:
    """Hook one MoT decoder layer, select + project its vision tokens.

    Usage
    -----
    >>> cap = MoTPhysicsCapturer(net, block_idx=18, vision=packed.vision,
    ...                          psi_head=net.physics_psi_head, phi_head=net.physics_phi_head)
    >>> with cap.capture():
    ...     out_net = model.denoise(...)
    >>> cap.psi_proj   # [1, N_vis, target_dim] or None
    >>> cap.phi_proj   # [1, N_vis, low_dim]    or None
    """

    def __init__(
        self,
        net: nn.Module,
        block_idx: int,
        vision,
        psi_head: Optional[nn.Module] = None,
        phi_head: Optional[nn.Module] = None,
    ) -> None:
        self.net = net
        self.block_idx = block_idx
        self.vision = vision
        self.psi_head = psi_head
        self.phi_head = phi_head
        self._handle = None
        self.psi_proj: Optional[torch.Tensor] = None
        self.phi_proj: Optional[torch.Tensor] = None
        self._fired = False

    def _resolve_block(self) -> nn.Module:
        layers = _resolve_decoder_layers(self.net)
        n = len(layers)
        idx = self.block_idx if self.block_idx >= 0 else n + self.block_idx
        if idx < 0 or idx >= n:
            raise IndexError(f"block_idx={self.block_idx} out of range (num_layers={n})")
        return layers[idx]

    @contextmanager
    def capture(self):
        block = self._resolve_block()
        seq_idx = self.vision.sequence_indexes

        def _hook(_module, _inputs, output):
            # Guard against multiple fires (e.g. AC recompute): keep the first.
            if self._fired:
                return None
            pack = output[0] if isinstance(output, tuple) else output
            hidden_all = get_all_seq(pack)  # [N_total, hidden]
            hidden_vis = hidden_all.index_select(0, seq_idx.to(hidden_all.device))  # [N_vis, hidden]
            hidden_vis = hidden_vis.unsqueeze(0)  # [1, N_vis, hidden]
            # Match each head's own parameter dtype so its LayerNorm never sees
            # mismatched scalar types; the fp32 upcast for the alignment math
            # happens later in integration.compute_physics_losses.
            if self.psi_head is not None:
                psi_dtype = next(self.psi_head.parameters()).dtype
                self.psi_proj = self.psi_head(hidden_vis.to(psi_dtype))  # [1, N_vis, target_dim]
            if self.phi_head is not None:
                phi_dtype = next(self.phi_head.parameters()).dtype
                self.phi_proj = self.phi_head(hidden_vis.to(phi_dtype))  # [1, N_vis, low_dim]
            self._fired = True
            return None

        self._handle = block.register_forward_hook(_hook)
        self.psi_proj = None
        self.phi_proj = None
        self._fired = False
        try:
            yield self
        finally:
            self._handle.remove()
            self._handle = None


def split_proj_by_sample(
    proj: torch.Tensor,
    vision,
) -> Tuple[List[torch.Tensor], List[Tuple[int, int, int]]]:
    """Split a projected vision-token sequence ``[1, N_vis, D]`` into per-sample
    chunks using ``vision.token_shapes``.

    Returns ``(chunks, dit_grids)`` where ``chunks[i]`` is ``[1, F_p*H_p*W_p, D]``
    in the exact frame-major ``(t, h_patch, w_patch)`` order produced by
    ``Cosmos3VFMNetwork.patchify_and_pack_latents`` and ``dit_grids[i]`` is
    ``(F_p, H_p, W_p)``.
    """
    assert proj.dim() == 3 and proj.size(0) == 1, f"expected [1, N_vis, D], got {tuple(proj.shape)}"
    assert vision.token_shapes is not None, "vision.token_shapes required to un-pack"
    chunks: List[torch.Tensor] = []
    dit_grids: List[Tuple[int, int, int]] = []
    start = 0
    for shape in vision.token_shapes:
        f_p, h_p, w_p = int(shape[0]), int(shape[1]), int(shape[2])
        n = f_p * h_p * w_p
        chunks.append(proj[:, start : start + n, :])  # [1, n, D]
        dit_grids.append((f_p, h_p, w_p))
        start += n
    if start != proj.size(1):
        raise RuntimeError(
            f"vision token_shapes account for {start} tokens but the projected "
            f"sequence has {proj.size(1)}; packing geometry mismatch."
        )
    return chunks, dit_grids
