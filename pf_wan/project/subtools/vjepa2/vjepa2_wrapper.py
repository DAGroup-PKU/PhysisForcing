
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class VJEPA2FeatureExtractor(nn.Module):
    """Frozen V-JEPA2 encoder for the *teacher* branch of the high-level
    physics alignment loss.

    Parameters
    ----------
    model_name_or_path : str
        HF hub id (e.g. ``facebook/vjepa2-vitl-fpc16-256``) or a local
        directory containing the model weights.
    frames_per_clip : Optional[int]
        Number of frames consumed per forward. ``None`` -> read from
        config (16 for ``vitl-fpc16``).
    crop_size : Optional[int]
        Square spatial resolution. ``None`` -> read from config
        (256 for ``vitl-fpc16-256``).
    dtype : torch.dtype
        Compute dtype for the encoder weights. Defaults to ``bfloat16``
        to keep VRAM low; it does not matter that this is mixed with the
        autocast dtype used elsewhere because the encoder is never in
        the backward graph.
    """

    def __init__(
        self,
        model_name_or_path: str = "./weights/vjepa2-vitl-fpc64-256",
        frames_per_clip: Optional[int] = None,
        crop_size: Optional[int] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()

        from transformers import VJEPA2Config, VJEPA2Model

        logger.info(f"[VJEPA2] Loading encoder from {model_name_or_path} (dtype={dtype})")
        config = VJEPA2Config.from_pretrained(model_name_or_path)
        model = VJEPA2Model.from_pretrained(model_name_or_path, torch_dtype=dtype)

        self.encoder = model.encoder
        del model  # we only need the encoder side
        self.config = config
        self.frames_per_clip = frames_per_clip or int(config.frames_per_clip)
        self.crop_size = crop_size or int(config.crop_size)
        self.patch_size = int(config.patch_size)
        self.tubelet_size = int(config.tubelet_size)
        self.hidden_size = int(config.hidden_size)
        self.dtype = dtype

        # Token grid for a single canonical clip.
        self.token_t = self.frames_per_clip // self.tubelet_size
        self.token_h = self.crop_size // self.patch_size
        self.token_w = self.crop_size // self.patch_size
        self.num_tokens = self.token_t * self.token_h * self.token_w

        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

        mean = torch.tensor(_IMAGENET_MEAN).view(1, 1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD).view(1, 1, 3, 1, 1)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)

        logger.info(
            f"[VJEPA2] Ready: frames_per_clip={self.frames_per_clip}, "
            f"crop_size={self.crop_size}, patch={self.patch_size}, "
            f"tubelet={self.tubelet_size}, hidden={self.hidden_size}, "
            f"token_grid=(T={self.token_t}, H={self.token_h}, W={self.token_w}), "
            f"num_tokens={self.num_tokens}"
        )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def train(self, mode: bool = True):  # type: ignore[override]
        # Never let drop-path / dropout activate.
        super().train(mode)
        self.encoder.eval()
        return self

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def _resample(self, video: torch.Tensor) -> torch.Tensor:
        """Resample ``[B, T, C, H, W]`` -> ``[B, fpc, C, crop, crop]``."""
        B, T, C, H, W = video.shape
        assert C == 3, f"V-JEPA2 expects 3-channel RGB input, got C={C}"
        if T == self.frames_per_clip and H == self.crop_size and W == self.crop_size:
            return video
        x = video.permute(0, 2, 1, 3, 4)
        x = F.interpolate(
            x.float(),  # interp on fp32 is the most predictable
            size=(self.frames_per_clip, self.crop_size, self.crop_size),
            mode="trilinear",
            align_corners=False,
        )
        x = x.permute(0, 2, 1, 3, 4)
        return x

    def _to_clip(self, video: torch.Tensor) -> torch.Tensor:
        """Convert ``[-1, 1]`` -> ``[0, 1]`` -> ImageNet normalized clip."""
        video01 = (video + 1.0) * 0.5
        video01 = video01.clamp_(0.0, 1.0)
        video01 = self._resample(video01)
        video01 = (video01 - self._mean.to(video01.dtype)) / self._std.to(video01.dtype)
        return video01.to(self.dtype)

    # ------------------------------------------------------------------
    # Forward API (teacher side, always no_grad)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_video_m1p1(self, video: torch.Tensor) -> torch.Tensor:
        """Encode a clip in ``[-1, 1]`` to patch-token features.

        Parameters
        ----------
        video : torch.Tensor
            Shape ``[B, T, 3, H, W]`` (any T, H, W; pixel range
            ``[-1, 1]``).

        Returns
        -------
        torch.Tensor
            Shape ``[B, N, D]`` where ``N = token_t * token_h * token_w``
            and ``D = hidden_size``. Detached, never builds a graph.
        """
        assert video.dim() == 5, f"expected [B,T,C,H,W], got {tuple(video.shape)}"
        clip = self._to_clip(video)
        out = self.encoder(pixel_values_videos=clip).last_hidden_state
        return out.detach().float()

    @torch.no_grad()
    def encode_video01(self, video01: torch.Tensor) -> torch.Tensor:
        """Same as :meth:`encode_video_m1p1` but accepts ``[0, 1]`` input."""
        assert video01.dim() == 5
        video = (video01 * 2.0) - 1.0
        return self.encode_video_m1p1(video)

    def forward(self, video: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.encode_video_m1p1(video)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def token_grid_shape(self) -> Tuple[int, int, int]:
        """``(T_tok, H_tok, W_tok)`` per clip."""
        return self.token_t, self.token_h, self.token_w
