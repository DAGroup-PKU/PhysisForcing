import itertools
import logging
import random
from contextlib import contextmanager
from typing import Any, List, Union

from pyarrow.fs import HadoopFileSystem, LocalFileSystem
from pyarrow.parquet import ParquetFile
from torch.utils.data import IterableDataset, default_collate

from project.utils import comm, common, hdfs, yield_seed
import pandas as pd

logger = logging.getLogger()


def partition_by_groups(data: List[Any], groups: int) -> List[List[Any]]:
    """
    Partition a list by groups.
    When indivisible, some groups may have more items than others.

    Examples:
        - data: [1,2,3,4,5]
        - groups: 2
        - return: [[1,3,5], [2,4]]
    """
    assert groups > 0
    return [data[i::groups] for i in range(groups)]


def get_portion_for_rank_and_worker(
    items: List[Any],
    rank: int,
    world_size:int,
    worker_id: int,
    num_workers: int,
    seed: int  # should be identical across all ranks
) -> List[Any]:
    """
    Get the portion of items for current rank and worker.
    """
    if world_size * num_workers < len(items):
        items = partition_by_groups(items, world_size)[rank]
        items = partition_by_groups(items, num_workers)[worker_id]
    else:
        seed = seed + rank * num_workers + worker_id
        random.Random(seed).shuffle(items)
    return items


class ParquetDataset(IterableDataset):
    def __init__(self, data_cfg, state_dict):
        super().__init__()
        self.batch_size = data_cfg.batch_size
        self.seed = data_cfg.seed  # for readers partition

        self.paths = [pd.read_csv(csv_path) for csv_path in data_cfg.paths]
        self.paths = pd.concat(self.paths, ignore_index=True)
        assert len(self.paths) > 0

        # for resume
        self.offset = self.seed
        self.parquet_offsets = dict()  # Dict[file_path: str, offset: int]
        self.state_dict = state_dict
        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()

    def load(self):
        real_worker_id = common.get_worker_id()
        last_worker_id = self.state_dict.get("last_worker_id", -1)
        num_workers = common.get_num_workers()
        worker_id = (real_worker_id + last_worker_id + 1) % num_workers
        # each worker has different beginning offset
        offset = self.offset + self.rank * num_workers + real_worker_id
        worker_state = self.state_dict.get(str(worker_id), {})
        self.offset = worker_state.get("offset", offset)
        
        self.parquet_offsets = {
            str(k): v for k, v in worker_state.items() if k != "offset"
        }
        logger.info(f"Worker {real_worker_id} shift to resume worker {worker_id}")
        return worker_id

    def __iter__(self):
        num_workers = common.get_num_workers()
        worker_id = self.load()

        total = len(self.paths)
        indices = list(range(total))
        indices = get_portion_for_rank_and_worker(indices, self.rank, self.world_size, worker_id, num_workers, self.seed)
        seed = self.seed + len(indices) * (self.rank * num_workers + worker_id)

        items = [
            (
                idx,
                self.paths.iloc[idx],
                self.parquet_offsets.get(str(idx), self.parquet_offsets.get(idx, seed + index))
            )
            for index, idx in enumerate(indices)
        ]

        # Epoch-based traversal: shuffle then walk through sequentially,
        # reshuffle when all items have been visited (= one epoch).
        epoch_order = list(range(len(items)))
        cursor = len(epoch_order)            # force shuffle on first access
        epoch_seed = self.offset              # independent seed chain for shuffling

        while True:
            rng = random.Random(self.offset)
            ret = []
            parquet_offsets = {}

            while len(ret) < self.batch_size:
                if cursor >= len(epoch_order):
                    random.Random(epoch_seed).shuffle(epoch_order)
                    epoch_seed = yield_seed(epoch_seed)
                    cursor = 0
                    logger.info(
                        f"Rank {self.rank} worker {worker_id} "
                        f"starting new epoch ({len(items)} samples)"
                    )

                idx, row, offset = items[epoch_order[cursor]]
                cursor += 1

                try:
                    parquet_offsets[str(idx)] = offset
                    ret.append(self.process_data(row, rng))
                except Exception as ex:
                    logger.warning(f"{self.__class__.__name__} got unexpected exception: {ex}.")
                    continue

            ret = default_collate(ret)
            ret["worker_id"] = worker_id
            ret["parquet_offsets"] = parquet_offsets
            ret["offset"] = self.offset
            self.offset = yield_seed(self.offset)
            yield ret


def get_filesystem(path: str) -> Union[LocalFileSystem, HadoopFileSystem]:
    """
    Get filesystem based on the path.
    """
    if path.startswith("hdfs://"):
        return HadoopFileSystem.from_uri(path)
    else:
        return LocalFileSystem()


@contextmanager
def open_parquet_files(path: str):
    fs = get_filesystem(path)
    file = ParquetFile(path, filesystem=fs)

    try:
        yield file
    finally:
        file.close()


class ParquetFileReader:
    def __init__(self, file_path, columns=None):
        self.file_path = file_path
        self.columns = columns

    def __iter__(self, offset):
        with open_parquet_files(self.file_path) as parquet_file:
            while True:
                rng = random.Random(offset)
                group = rng.randint(0, parquet_file.num_row_groups - 1)
                num_rows = parquet_file.metadata.row_group(group).num_rows
                batch = parquet_file.read_row_group(group, self.columns).to_pandas()
                row = rng.randint(0, num_rows - 1)
                ret = batch.iloc[row].to_dict()
                ret["offset"] = offset
                yield ret
                offset = yield_seed(offset)
