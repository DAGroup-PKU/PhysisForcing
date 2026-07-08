import copy
import logging
from typing import Callable, Union

import imageio
import torch
import torchvision
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.utils.data import get_worker_info

logger = logging.getLogger()


def to_torch_dtype(dtype: str):
    if dtype in ("bf16", "bfloat16", "torch.bfloat16"):
        return torch.bfloat16
    if dtype in ("fp16", "float16", "torch.float16"):
        return torch.float16
    if dtype in ("fp32", "float32", "torch.float32"):
        return torch.float32
    if dtype in ("fp64", "float64", "torch.float64"):
        return torch.float64
    if dtype in ("double", "torch.double"):
        return torch.double

    raise ValueError(f"Unrecognized dtype {dtype}")


def get_attribute(x, attr_str, default=None):
    """
    Get the attribute value of the given object.

    Parameters:
    x (object): The input Python object.
    attr_str (str): The attribute path, in the format of "a.b.c".

    Returns:
    Any type: The value of the corresponding attribute.
    """
    res, attrs = x, attr_str.split(".")
    while attrs:
        res = getattr(res, attrs.pop(0), None)
        if res is None:
            return default
    return res


def save_video(
    tensor,
    save_file=None,
    fps=30,
    nrow=8,
    normalize=True,
    value_range=(-1, 1),
):
    # preprocess
    tensor = tensor.clamp(min(value_range), max(value_range))
    tensor = torch.stack([
        torchvision.utils.make_grid(
            u, nrow=nrow, normalize=normalize, value_range=value_range)
        for u in tensor.unbind(2)
    ], dim=1).permute(1, 2, 3, 0)
    tensor = (tensor * 255).type(torch.uint8).cpu()

    # write video
    writer = imageio.get_writer(save_file, fps=fps, codec='libx264', quality=8)
    for frame in tensor.numpy():
        writer.append_data(frame)
    writer.close()
    return save_file


def get_worker_id() -> int:
    """
    Get the current dataloader worker id.
    """
    return get_worker_info().id if get_worker_info() is not None else 0


def get_num_workers() -> int:
    """
    Get the total dataloader worker count.
    """
    return get_worker_info().num_workers if get_worker_info() is not None else 1


def gradient_checkpointing(module: Union[Callable, nn.Module], *args, use_reentrant, enabled: bool, **kwargs):
    if enabled:
        return checkpoint(
            module,
            *args,
            use_reentrant=use_reentrant,
            **kwargs,
        )
    else:
        return module(*args, **kwargs)


def maybe_checkpoint(module, *args, enabled=True, use_reentrant=False, **kwargs):
    # if module.training is False, we should still enable gradient checkpointing if it is not within torch.no_grad
    if enabled and torch.is_grad_enabled():
        return gradient_checkpointing(module, *args, use_reentrant=use_reentrant, enabled=enabled, **kwargs)
    else:
        return module(*args, **kwargs)


def deepcopy_with_tensor_clone(obj):
    if isinstance(obj, dict):
        return {k: deepcopy_with_tensor_clone(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deepcopy_with_tensor_clone(v) for v in obj]
    elif torch.is_tensor(obj):
        return obj.clone()
    else:
        return copy.deepcopy(obj)


def expand_dims(tensor: torch.Tensor, ndim: int):
    """
    Expand tensor to target ndim. New dims are added to the right.
    For example, if the tensor shape was (8,), target ndim is 4, return (8, 1, 1, 1).
    """
    shape = tensor.shape + (1,) * (ndim - tensor.ndim)
    return tensor.reshape(shape)
