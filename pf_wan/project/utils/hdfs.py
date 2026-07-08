import logging
import os
import pickle
import shutil
import subprocess
from typing import List

from project.utils import comm

logger = logging.getLogger()


def mkdir(path: str):
    """
    Create directory. Support either hdfs or local path.
    Create all parent directory if not present. No-op if directory already present.
    """
    if path.startswith('hdfs://'):
        if comm.get_rank() == 0:
            subprocess.run(["hdfs", "dfs", "-mkdir", "-p", path])
    else:
        os.makedirs(path, exist_ok=True)


def copy(src: str, tgt: str):
    """
    Copy file. Source and destination supports either hdfs or local path.
    """
    src_hdfs = src.startswith("hdfs://")
    tgt_hdfs = tgt.startswith("hdfs://")

    chunk_size = os.environ.get("HDFS_PARALLEL_CHUNK_SIZE", 512)
    chunk_number = os.environ.get("HDFS_PARALLEL_NUM_CHUNKS", 8)
    file_number = os.environ.get("HDFS_PARALLEL_NUM_FILES", 8)

    args = []
    if chunk_size is not None:
        args.append(f"-c{chunk_size}")
    if chunk_number is not None:
        args.append(f"--ct={chunk_number}")
    if file_number is not None:
        args.append(f"-t{file_number}")

    try:
        if src_hdfs and tgt_hdfs:
            subprocess.run(["hdfs", "dfs", "-cp", "-f", src, tgt])
        elif src_hdfs and not tgt_hdfs:
            subprocess.run(["hdfs", "dfs", "-copyToLocal", *args, src, tgt])
        elif not src_hdfs and tgt_hdfs:
            subprocess.run(["hdfs", "dfs", "-copyFromLocal", *args, src, tgt])
        else:
            shutil.copy(src, tgt)
    except:
        logger.info("Failed to copy file from {} to {}".format(src, tgt))
        logger.info("Remember to save it on your own!!!")


def exists(path: str) -> bool:
    """
    Check whether a path exists. Support either hdfs or local path
    Return True if the path exists.
    """
    if path.startswith('hdfs://'):
        process = subprocess.run(["hdfs", "dfs", "-test", "-e", path], capture_output=True)
        return process.returncode == 0
    return os.path.exists(path)


def listdir(path: str) -> List[str]:
    """
    List directory. Supports either hdfs or local path. Returns full path.

    Examples:
        - listdir("hdfs://dir") -> ["hdfs://dir/file1", "hdfs://dir/file2"]
        - listdir("/dir") -> ["/dir/file1", "/dir/file2"]
    """
    files = []

    if path.startswith('hdfs://'):
        metafile = os.path.join(path, "metafile.pkl") # A metafile contains a list of file paths
        if exists(metafile):
            from project.utils import maybe_download
            with open(maybe_download(metafile), "rb") as f:
                return pickle.loads(f.read())

        pipe = subprocess.Popen(
            args=["hdfs", "dfs", "-ls", path],
            shell=False,
            stdout=subprocess.PIPE)

        for line in pipe.stdout:
            parts = line.strip().split()

            # drwxr-xr-x   - user group  4 file
            if len(parts) < 5:
                continue

            files.append(parts[-1].decode("utf8"))

        pipe.stdout.close()
        pipe.wait()

    else:
        files = [os.path.join(path, file) for file in os.listdir(path)]

    return files
