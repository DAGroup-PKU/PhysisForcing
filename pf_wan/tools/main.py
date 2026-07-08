import argparse
import builtins
import logging
import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist

from project.engines import ENGINE_REGISTRY
from project.utils import comm, gcfg

logger = logging.getLogger()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("opts", help="Modify config options using the command-line 'KEY VALUE' pairs",
                        default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()

    gcfg.init()
    gcfg.merge_from_file(args.config_file)
    gcfg.merge_from_list(args.opts)
    if not gcfg.get("exp_name", None):
        gcfg.exp_name = os.path.splitext(os.path.basename(args.config_file))[0]
    gcfg.freeze()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=60)
        )

    logging_dir = os.path.join(gcfg.output_dir, gcfg.proj_name, gcfg.exp_name)
    os.makedirs(logging_dir, exist_ok=True)
    # this is necessary in case several packages call logging first and break the configuration here
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    global_rank = comm.get_rank()
    local_rank = comm.get_local_rank()

    fmt = "[%(asctime)s %(filename)s:%(lineno)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        filename=f"{logging_dir}/log_rank{global_rank}.txt",
        filemode="a"
    )

    if local_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(fmt, datefmt))
        logger.addHandler(console_handler)

    # # suppress redundant print from others
    # def print_pass(*args, **kwargs):
    #     pass
    # builtins.print = print_pass

    # logger.info("Global config:\n{}".format(gcfg.dump()))

    engine_cls = ENGINE_REGISTRY.get(gcfg.engine)
    engine = engine_cls(gcfg)
    engine.run()
