"""High-level relational alignment loss (Sec. 3.3).

Given a set of *physics-informative* tokens selected by the mask M
(produced from CoTracker3 trajectories + first-frame depth, see Sec. 3.1
of the paper and ``physics_mask.py``), the student (DiT side, after the
projection MLP ``ψ``) and the teacher (frozen V-JEPA2) produce two token
matrices

    F_hat = {F_hat^M_{t,n} : (t,n) ∈ M}  ∈ R^{K x C}    (student)
    F     = {F^M_{t,n}     : (t,n) ∈ M}  ∈ R^{K x C}    (teacher)

The relational matrices are the pairwise cosine similarities

    R_hat(i,j) = <F_hat_i, F_hat_j> / (||F_hat_i|| ||F_hat_j||)
    R(i,j)     = <F_i,     F_j>     / (||F_i||     ||F_j||)

and the high-level physics alignment loss is the elementwise L1

    L^phy_high = (1 / K^2) * sum_{i,j} | R_hat(i,j) - R(i,j) |    (eq. 12)

The student matrix retains its autograd graph (it depends on the DiT
hidden feature through the trainable MLP ψ). The teacher matrix is
detached.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def cosine_relation_matrix(features: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity over the last-but-one dimension.

    Parameters
    ----------
    features : torch.Tensor
        ``[..., K, C]``.

    Returns
    -------
    torch.Tensor
        ``[..., K, K]``.
    """
    f = _l2_normalize(features.float())
    return f @ f.transpose(-1, -2)


def relational_alignment_loss(
    student_tokens: torch.Tensor,
    teacher_tokens: torch.Tensor,
    *,
    norm: str = "l1",
    temperature: float = 1.0,
) -> dict:
    """Relational L1 loss (Eq. 12), with optional alternative norms.

    Parameters
    ----------
    student_tokens : torch.Tensor
        Shape ``[B, K, C]`` (student-side selected tokens, autograd live).
    teacher_tokens : torch.Tensor
        Shape ``[B, K, C]`` (teacher-side selected tokens, detached).
    norm : {"l1", "l2", "kl"}
        ``"l1"`` reproduces the paper exactly. ``"l2"`` and ``"kl"`` are
        optional alternatives for ablations.
    temperature : float
        Used only when ``norm == "kl"``.

    Returns
    -------
    dict with key ``"loss"``.
    """
    assert student_tokens.shape == teacher_tokens.shape, (
        f"shape mismatch: {student_tokens.shape} vs {teacher_tokens.shape}"
    )
    if student_tokens.size(1) < 2:
        # K < 2: no pair to align, return a zero with the right graph.
        zero = student_tokens.float().sum() * 0.0
        return {"loss": zero}

    teacher_tokens = teacher_tokens.detach()

    R_s = cosine_relation_matrix(student_tokens)
    R_t = cosine_relation_matrix(teacher_tokens)

    if norm == "l1":
        loss = (R_s - R_t).abs().mean()
    elif norm == "l2":
        loss = F.mse_loss(R_s, R_t)
    elif norm == "kl":
        log_s = F.log_softmax(R_s / temperature, dim=-1)
        t = F.softmax(R_t / temperature, dim=-1)
        loss = F.kl_div(log_s, t, reduction="batchmean") * (temperature ** 2)
    else:
        raise ValueError(f"unknown norm: {norm!r}")

    return {"loss": loss}
