import hashlib
import json
import logging
import os
import os.path as osp
from urllib.parse import urlparse
import pickle
import requests
import torch
from huggingface_hub import hf_hub_download as _hf_hub_download
from huggingface_hub import snapshot_download as _snapshot_download

from project.utils import comm, hdfs

__all__ = [
    "maybe_download",
    "maybe_upload",
    "hf_hub_download",
    "snapshot_download"
]

logger = logging.getLogger()


def maybe_download(path, local_dir=None, distributed=False):
    if local_dir is None:
        local_dir = "/tmp/pf_wan_cache"
    sha256 = hashlib.sha256(path.encode("utf-8")).hexdigest()[:12]
    local_dir = osp.join(local_dir, sha256)

    if not distributed and comm.get_local_rank() != 0:
        comm.barrier()
        local_path = comm.local_broadcast_object(None, local_src=0)
        return local_path

    if path.startswith("hdfs://"):  # maybe a directory
        os.makedirs(local_dir, exist_ok=True)
        filename = osp.basename(osp.normpath(path))
        local_path = osp.join(local_dir, filename)
        if not osp.exists(local_path):
            logger.info(f"Downloading {path} to {local_dir}")
            hdfs.copy(path, local_path)
    else:
        parsed = urlparse(path)
        if parsed.scheme in ('http', 'https'):
            os.makedirs(local_dir, exist_ok=True)
            filename = osp.basename(parsed.path)
            if not filename:
                filename = "downloaded_file"

            local_path = osp.join(local_dir, filename)
            if not osp.exists(local_path):
                logger.info(f"Downloading {path} to {local_path}")
                try:
                    response = requests.get(path, stream=True)
                    response.raise_for_status()

                    with open(local_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                except Exception as e:
                    if osp.exists(local_dir) and not os.listdir(local_dir):
                        os.rmdir(local_dir)
                    raise RuntimeError(f"Failed to download {path}") from e
        else:
            local_path = path

    if not distributed:
        comm.barrier()
        local_path = comm.local_broadcast_object(local_path, local_src=0)

    return local_path


def maybe_upload(obj, filename: str, save_dir: str, local_dir=None):
    # this is necessary in case too many ckpts are saved in local disk
    if local_dir is None:
        local_dir = "/tmp/pf_wan_cache"
    sha256 = hashlib.sha256(filename.encode("utf-8")).hexdigest()[:12]
    local_dir = osp.join(local_dir, sha256)

    os.makedirs(local_dir, exist_ok=True)
    if filename.endswith(".pth"):
        torch.save(obj, osp.join(local_dir, filename))  # override
    elif filename.endswith(".json"):
        with open(osp.join(local_dir, filename), "w", encoding="utf8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=4)
    else:
        with open(osp.join(local_dir, filename), "wb") as f:
            pickle.dump(obj, f)

    hdfs.copy(osp.join(local_dir, filename), osp.join(save_dir, filename))


@comm.local_main_process_only
def hf_hub_download(*args, **kwargs):
    return _hf_hub_download(*args, **kwargs)


@comm.local_main_process_only
def snapshot_download(*args, **kwargs):
    return _snapshot_download(*args, **kwargs)
