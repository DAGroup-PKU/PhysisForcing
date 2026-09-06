import json
import logging

from torch.utils.data import DataLoader

from project.data import DATASET_REGISTRY
from project.utils import comm, maybe_download

logger = logging.getLogger()


class BaseDataLoader:
    def __init__(self, data_cfg, ckpt_path=None):
        self.rank = comm.get_rank()
        self.state_dict = self.load(ckpt_path)
        dataset_cls = DATASET_REGISTRY.get(data_cfg.name)
        self.dataset = dataset_cls(data_cfg, self.state_dict)

        dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=None,
            num_workers=data_cfg.num_workers,
            prefetch_factor=data_cfg.prefetch_factor,
            pin_memory=True
        )
        self.loader_iter = iter(dataloader)

    def load(self, ckpt_path):
        if ckpt_path is not None:
            with open(maybe_download(ckpt_path, distributed=True), "r", encoding="utf8") as f:
                state_dict = json.load(f)
            logger.info(f"Resume data loader from {ckpt_path}")
        else:
            state_dict = dict()
        return state_dict

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self.loader_iter)
        self._update_state(batch)
        return batch

    def _update_state(self, batch):
        worker_id = str(batch["worker_id"])
        if worker_id not in self.state_dict:
            self.state_dict[worker_id] = dict()
        self.state_dict[worker_id]["offset"] = batch["offset"]
        # Persist sample offsets with string keys for stable JSON roundtrip.
        self.state_dict[worker_id].update({
            str(k): v for k, v in batch["parquet_offsets"].items()
        })
        self.state_dict["last_worker_id"] = batch["worker_id"]
        self.state_dict = dict(sorted(self.state_dict.items(), key=lambda x: str(x[0])))
