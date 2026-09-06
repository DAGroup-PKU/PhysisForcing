import logging
import os
from random import Random
import json, random


import cv2
import torch
import torchvision.transforms.functional as TVF

from project.data import DATASET_REGISTRY
from project.data.parquet import ParquetDataset

logger = logging.getLogger()


@DATASET_REGISTRY.register()
class VideoI18NParquetDataset(ParquetDataset):
    """
    Built for tiktok I18N video data.
    """
    def __init__(self, data_cfg, state_dict):
        super().__init__(data_cfg, state_dict)

        self.tos_url_key = data_cfg.tos_url_key
        self.text_key = data_cfg.text_key
        self.window_size = data_cfg.window_size
        self.stride = data_cfg.stride
        self.max_output_frames = data_cfg.max_frames
        # When True and the stride-decimated sequence is still longer than
        # ``max_frames``, pick exactly ``max_frames`` indices uniformly across
        # the whole video (covers the entire clip). When False, fall back to
        # the original behaviour: take ``4n+1`` strided frames from a random
        # contiguous window. The ``4n+1`` rule comes from the Wan VAE temporal
        # stride (=4); ``max_frames`` itself is already 4n+1 (e.g. 81=4*20+1),
        # so the uniform branch is automatically a valid target length.
        self.uniform_sampling = getattr(data_cfg, "uniform_sampling", False)
        # Short-clip override. If a raw clip's total frame count is <=
        # ``no_stride_max_frames``, ignore ``stride`` and treat the clip as
        # stride=1 (consecutive frames). Designed for high-quality clips that
        # are already fps-aligned to the target window length (e.g. exactly 81
        # frames at 16 fps), where decimation would just throw away signal.
        # Default 0 disables the override (legacy behaviour preserved).
        self.no_stride_max_frames = int(getattr(data_cfg, "no_stride_max_frames", 0) or 0)
        # robotics video transform
        self.aspect_ratio_buckets = data_cfg.aspect_ratio_buckets
        
    def process_data(self, data: dict, rng: Random):
        results = {}
        results["text"] = data[self.text_key]
        results["video"] = self._process_video(data[self.tos_url_key], rng)
        return results

    def _video_transform(self, frames: torch.Tensor):
        """
        Transform video frames to match the closest resolution from aspect_ratio_buckets.
        
        Args:
            frames (torch.Tensor): Input video frames with shape (T, H, W, 3)
        
        Returns:
            torch.Tensor: Transformed video frames with shape (T, H', W', 3)
        """
        # Get original frame dimensions and aspect ratio
        frame_height, frame_width = frames.shape[1:3]
        frame_aspect_ratio = frame_width / frame_height

        # Select the bucket with the closest aspect ratio
        best_bucket = min(
            self.aspect_ratio_buckets,
            key=lambda b: abs((b[1] / b[0]) - frame_aspect_ratio)
        )
        bucket_height, bucket_width = best_bucket
        target_aspect_ratio = bucket_width / bucket_height

        # Calculate cropping dimensions to match target aspect ratio
        if frame_aspect_ratio > target_aspect_ratio:
            # Too wide, crop width
            crop_height = frame_height
            crop_width = int(round(frame_height * target_aspect_ratio))
        else:
            # Too tall, crop height
            crop_width = frame_width
            crop_height = int(round(frame_width / target_aspect_ratio))

        # Ensure crop size does not exceed original size
        crop_width = min(crop_width, frame_width)
        crop_height = min(crop_height, frame_height)

        # Convert frames from (T, H, W, 3) -> (T, 3, H, W)
        frames = frames.permute(0, 3, 1, 2)
        
        # Center crop and resize to target resolution
        frames = TVF.center_crop(frames, output_size=[crop_height, crop_width])
        frames = TVF.resize(frames, size=[bucket_height, bucket_width])
        
        # Convert back to (T, H, W, 3)
        frames = frames.permute(0, 2, 3, 1)
        
        return frames

    def _process_video(self, key: str, rng: Random):
        cap = cv2.VideoCapture(key)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_idxs = self._sampling(
            total_frames=total_frames,
            stride=self.stride,
            max_output_frames=self.max_output_frames,
            rng=rng,
        )

        # frame_idxs is strictly increasing; use a set for O(1) membership
        # and stop reading once we've passed the last needed index.
        frame_idxs_set = set(frame_idxs)
        last_needed = frame_idxs[-1]

        frames_by_idx = {}
        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if idx in frame_idxs_set:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB
                frames_by_idx[idx] = torch.tensor(frame)

            if idx >= last_needed:
                break
            idx += 1

        cap.release()

        # Preserve the order/multiplicity defined by frame_idxs (uniform mode
        # may emit duplicates for very short clips after de-duplication).
        frames = torch.stack([frames_by_idx[i] for i in frame_idxs if i in frames_by_idx])
        frames = self._video_transform(frames)  # (T, H, W, 3)
        return frames
    
    def _sampling(self, total_frames, stride, max_output_frames, rng: Random):
        """Pick frame indices.

        Two modes:
        1. default (``uniform_sampling=False``): random start in
           [0, max_start_index], then take strided frames so the count is 4n+1
           (and <= ``max_output_frames`` and the clip length).
        2. uniform (``uniform_sampling=True``): if the stride-decimated length
           exceeds ``max_output_frames``, take exactly ``max_output_frames``
           evenly-spaced indices across the whole clip (must be 4n+1 for the Wan
           VAE temporal stride); otherwise fall back to the default logic.

        Short-clip bypass (``no_stride_max_frames > 0``): if
        ``total_frames <= no_stride_max_frames`` force stride=1 (consecutive
        frames); applies to both modes.
        """
        if self.no_stride_max_frames > 0 and total_frames <= self.no_stride_max_frames:
            stride = 1

        if self.uniform_sampling:
            # effective length after stride decimation
            effective = (total_frames + stride - 1) // stride
            if effective > max_output_frames:
                # max_output_frames must be 4n+1 (e.g. 81 = 4*20 + 1); take that
                # many evenly-spaced points over [0, total_frames-1].
                assert (max_output_frames - 1) % 4 == 0, (
                    f"max_output_frames={max_output_frames} must be 4n+1 to "
                    f"satisfy Wan VAE temporal stride."
                )
                step = (total_frames - 1) / (max_output_frames - 1)
                frame_indices = [
                    min(total_frames - 1, int(round(i * step)))
                    for i in range(max_output_frames)
                ]
                # force strictly increasing (rounding can tie at very low fps)
                for i in range(1, len(frame_indices)):
                    if frame_indices[i] <= frame_indices[i - 1]:
                        frame_indices[i] = min(total_frames - 1, frame_indices[i - 1] + 1)
                return frame_indices
            # short clip: fall through to the default logic

        # default: random window + stride + 4n+1 alignment
        max_possible_frame_count = min((total_frames + stride - 1) // stride, max_output_frames)
        max_n = (max_possible_frame_count - 1) // 4
        target_frame_count = 4 * max_n + 1
        total_length_needed = (target_frame_count - 1) * stride + 1

        max_start_index = max(0, total_frames - total_length_needed)
        start_index = rng.randint(0, max_start_index)

        frame_indices = [start_index + i * stride for i in range(target_frame_count)]
        return frame_indices



if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from project.utils import Dataclass
    dataset = VideoI18NParquetDataset(
        data_cfg=Dataclass(
            batch_size=1,
            seed=0,
            paths=[
                "examples/data/robvid_example/metadata.csv"
            ],
            tos_url_key="video_path",
            text_key="short_caption",
            window_size=120,
            stride=4,
            max_frames=81,
            aspect_ratio_buckets = [
                [832, 480], [480, 832]
            ],
        ),
        state_dict={}
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=4,
        prefetch_factor=4,
        pin_memory=True
    )
    loader_iter = iter(dataloader)
    
    for i in range (16):
        batch = next(loader_iter)
        print(batch["video"].shape)
