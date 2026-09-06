from project.utils import Registry

DATASET_REGISTRY = Registry("DATASET")

from .base_dataloader import BaseDataLoader
from .video_cn import VideoCNParquetDataset
from .video_i18n import VideoI18NParquetDataset
