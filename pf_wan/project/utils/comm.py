import functools
import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger()

GLOBAL_CPU_GROUP = None
LOCAL_CPU_GROUP = None
LOCAL_GPU_GROUP = None


def get_rank(group=None):
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank(group)


def get_local_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_local_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))


def get_device():
    return torch.device("cuda", get_local_rank())


def get_cpu_group():
    world_size = get_world_size()
    global GLOBAL_CPU_GROUP
    if GLOBAL_CPU_GROUP is None:
        GLOBAL_CPU_GROUP = dist.new_group(backend="gloo")
        logger.info(f"Created global CPU group with {world_size} ranks.")
    return GLOBAL_CPU_GROUP


def get_local_cpu_group():
    global LOCAL_CPU_GROUP
    if LOCAL_CPU_GROUP is None:
        local_world_size = get_local_world_size()
        num_machines = get_world_size() // local_world_size
        machine_rank = get_rank() // local_world_size
        for i in range(num_machines):
            ranks_on_i = list(range(i * local_world_size, (i + 1) * local_world_size))
            pg = dist.new_group(ranks_on_i, backend="gloo")
            if i == machine_rank:
                LOCAL_CPU_GROUP = pg
                logger.info(f"Created local CPU group with {local_world_size} ranks.")
    return LOCAL_CPU_GROUP


def get_local_gpu_group():
    global LOCAL_GPU_GROUP
    if LOCAL_GPU_GROUP is None:
        local_world_size = get_local_world_size()
        num_machines = get_world_size() // local_world_size
        machine_rank = get_rank() // local_world_size
        for i in range(num_machines):
            ranks_on_i = list(range(i * local_world_size, (i + 1) * local_world_size))
            pg = dist.new_group(ranks_on_i, backend="nccl")
            if i == machine_rank:
                LOCAL_GPU_GROUP = pg
                logger.info(f"Created local GPU group with {local_world_size} ranks.")
    return LOCAL_GPU_GROUP


def barrier():
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    dist.barrier()  # default nccl process group for gpu barrier
    dist.barrier(get_cpu_group())  # global cpu group for cpu barrier


def local_barrier():
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    dist.barrier(get_local_gpu_group())  # local gpu barrier
    dist.barrier(get_local_cpu_group())  # local cpu barrier


def all_gather_object(data):
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    gather_list = [None for _ in range(world_size)]
    dist.all_gather_object(gather_list, data, group=get_cpu_group())
    return gather_list


def broadcast_object(data, src=0):
    world_size = get_world_size()
    if world_size == 1:
        return data
    broadcast_list = [data] if get_rank() == src else [None]
    dist.broadcast_object_list(broadcast_list, src=src, group=get_cpu_group())
    return broadcast_list[0]


def gather_object(data, dst=0):
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    gather_list = [None for _ in range(world_size)] if get_rank() == dst else None
    dist.gather_object(data, gather_list, dst=dst, group=get_cpu_group())
    return gather_list


def local_broadcast_object(data, local_src=0):
    local_world_size = get_local_world_size()
    if local_world_size == 1:
        return data
    machine_rank = get_rank() // local_world_size
    src = machine_rank * local_world_size + local_src
    broadcast_list = [data] if get_local_rank() == local_src else [None]
    dist.broadcast_object_list(broadcast_list, src=src, group=get_local_cpu_group())
    return broadcast_list[0]


def main_process_first(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank = get_rank()
        if rank == 0:
            result = func(*args, **kwargs)
            barrier()
            return result
        else:
            barrier()
            return func(*args, **kwargs)
    return wrapper


def main_process_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank = get_rank()
        result = None
        if rank == 0:
            logger.warning(f"Rank {rank} is running {func.__name__}.")
            result = func(*args, **kwargs)
        else:
            logger.warning(f"Rank {rank} is waiting for rank 0 to finish {func.__name__}.")
        barrier()
        result = broadcast_object(result, src=0)
        return result
    return wrapper


def local_main_process_first(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        local_rank = get_local_rank()
        if local_rank == 0:
            result = func(*args, **kwargs)
            barrier()
            return result
        else:
            barrier()
            return func(*args, **kwargs)
    return wrapper


def local_main_process_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        local_rank = get_local_rank()
        result = None
        if local_rank == 0:
            logger.warning(f"Local rank {local_rank} is running {func.__name__}.")
            result = func(*args, **kwargs)
        else:
            logger.warning(f"Local rank {local_rank} is waiting for local rank 0 to finish {func.__name__}.")
        barrier()
        result = local_broadcast_object(result, local_src=0)
        logger.warning(f"Local rank {local_rank} finished {func.__name__}.")
        return result
    return wrapper
