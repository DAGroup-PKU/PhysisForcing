import json
import logging
import random
import re
from random import Random

import cv2
import requests
import torch

from project.data import DATASET_REGISTRY
from project.data.parquet import ParquetDataset

logger = logging.getLogger()


@DATASET_REGISTRY.register()
class VideoCNParquetDataset(ParquetDataset):
    """
    Built for douyin CN video data.
    """
    def process_data(self, data: dict, rng: Random):
        results = {}
        meta = self._get_meta_info(data)
        results["text"] = self._process_text(meta, rng)
        results["video"] = self._process_video(meta, rng)
        return results

    def _get_meta_info(self, sample):
        # get meta url.
        url = sample["video_meta_url"]
        # resolve meta url.
        rsp = requests.get(url=url, timeout=5)
        rsp.raise_for_status()
        rsp = rsp.json()

        # save meta info.
        meta = {}
        meta['vid'] = rsp['vid']
        ## video info
        meta['ori_video_url'] = rsp['ori_video_url']
        video_info = {}
        video_info['width'] = rsp['width']
        video_info['height'] = rsp['height']
        video_info['duration'] = rsp['duration']
        video_info['fps'] = rsp['fps']
        meta['video_info'] = video_info
        frame_info = {}
        st_ed = (json.loads(rsp['properties']['dump_latent_input'])['st_ed'])
        frame_info['st_frame'], frame_info['ed_frame'] = st_ed['st'], st_ed['ed']
        frame_info['num_frames'] = frame_info['ed_frame'] - frame_info['st_frame'] + 1
        meta['frame_info'] = frame_info
        ## text info
        meta['text'] = json.loads(rsp['properties']['dump_latent_input'])['text']
        # meta['text'] = {**meta['text'], **rsp['properties']['origin_properties_meta']['title']}
        return meta

    def _process_text(self, meta, rng: random.Random):
        text_dict = meta['text'] # text = {"": "", "": ""}
        text_keys = rng.choice(list(text_dict.keys()))
        text = text_dict[text_keys]
        return re.sub(r'[\(\（][^)\）]*[\)\）][,，]?\s*', '', text)

    def _process_video(self, meta, rng: random.Random):
        frame_info = meta['frame_info']
        st_frame = frame_info['st_frame']
        ed_frame = frame_info['ed_frame']
        frame_idxs = list(range(st_frame, ed_frame))
        frames = self._decode_video(meta['ori_video_url'], frame_idxs)
        return frames

    def _decode_video(self, video_url, frame_idxs):
        cap = cv2.VideoCapture(video_url)
        frames = []

        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if idx in frame_idxs:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB
                frame_tensor = torch.tensor(frame)
                frames.append(frame_tensor)

            idx += 1

        cap.release()

        return torch.stack(frames)  # (T, H, W, 3)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from project.utils import Dataclass
    dataset = VideoCNParquetDataset(
        data_cfg=Dataclass(
            batch_size=1,
            seed=0,
            paths=[
                "/path/to/i2v_parquet_dataset"
            ]
        ),
        state_dict={}
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=None,
        num_workers=4,
        prefetch_factor=4,
        pin_memory=True
    )
    loader_iter = iter(dataloader)
    for _ in range(16):
        batch = next(loader_iter)
        print(batch["parquet_offsets"], batch["offset"])
