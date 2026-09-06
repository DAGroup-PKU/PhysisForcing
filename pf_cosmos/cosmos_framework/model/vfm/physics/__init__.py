# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Physics-alignment training objectives for Cosmos3 i2v (MoT) fine-tuning.

Three auxiliary objectives, active only when ``physics_alignment.enabled``:

* High-level — V-JEPA2 relational alignment via a trainable MLP ``ψ`` (see
  :mod:`relational_loss`).
* Low-level — CoTracker3 trajectory alignment via a trainable MLP ``φ`` (see
  :mod:`trajectory_loss`).
* Region focus — a token mask ``M`` from CoTracker3 tracks + first-frame depth
  selecting which tokens enter the high-level loss (see :mod:`physics_mask`).

The MoT capture/un-pack glue lives in :mod:`mot_capture`; the per-batch
orchestration in :mod:`integration`.
"""

from cosmos_framework.model.vfm.physics.physics_heads import (
    HighLevelPhysicsHead,
    LowLevelPhysicsHead,
)
from cosmos_framework.model.vfm.physics.relational_loss import relational_alignment_loss

__all__ = [
    "HighLevelPhysicsHead",
    "LowLevelPhysicsHead",
    "relational_alignment_loss",
]
