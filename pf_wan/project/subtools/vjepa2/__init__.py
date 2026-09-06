"""High-level physics alignment utilities (Sec. 3.3)."""

from project.subtools.vjepa2.vjepa2_wrapper import VJEPA2FeatureExtractor
from project.subtools.vjepa2.physics_alignment_head import (
    HighLevelPhysicsHead,
    LowLevelPhysicsHead,
    DiTBlockFeatureCapturer,
)
from project.subtools.vjepa2.physics_mask import (
    project_tracks_to_pixel_mask,
    downsample_mask_to_grid,
    select_aligned_tokens_with_shared_indices,
)
from project.subtools.vjepa2.relational_loss import (
    cosine_relation_matrix,
    relational_alignment_loss,
)

__all__ = [
    "VJEPA2FeatureExtractor",
    "HighLevelPhysicsHead",
    "LowLevelPhysicsHead",
    "DiTBlockFeatureCapturer",
    "project_tracks_to_pixel_mask",
    "downsample_mask_to_grid",
    "select_aligned_tokens_with_shared_indices",
    "cosine_relation_matrix",
    "relational_alignment_loss",
]
