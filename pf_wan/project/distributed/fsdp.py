import logging
from datetime import timedelta

import torch
from peft.tuners.lora.layer import LoraLayer
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.distributed_c10d import _set_pg_timeout
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

from project.utils import Registry, comm, common

DEVICE_MESH = None
FSDP_WRAP_POLICY_REGISTRY = Registry("FSDP_WRAP_POLICY")
logger = logging.getLogger()


def setup_fsdp(model, meta_model_cfg, model_cfg):
    device_id = comm.get_local_rank()
    training = model.training

    # model_cfg.fsdp override meta_model.fsdp
    fsdp_cfg = meta_model_cfg.fsdp
    reduce_dtype = model_cfg.fsdp.get("reduce_dtype", fsdp_cfg.reduce_dtype)
    buffer_dtype = model_cfg.fsdp.get("buffer_dtype", fsdp_cfg.buffer_dtype)
    auto_wrap_policy = model_cfg.fsdp.get("auto_wrap_policy", fsdp_cfg.auto_wrap_policy)
    sync_module_states = model_cfg.fsdp.get("sync_module_states", fsdp_cfg.sync_module_states)
    cpu_offload = model_cfg.fsdp.get("cpu_offload", fsdp_cfg.cpu_offload)
    sharding_strategy = model_cfg.fsdp.get("sharding_strategy", fsdp_cfg.sharding_strategy)
    use_orig_params = model_cfg.fsdp.get("use_orig_params", fsdp_cfg.use_orig_params)

    auto_wrap_policy = (
        FSDP_WRAP_POLICY_REGISTRY.get(auto_wrap_policy)
        if auto_wrap_policy is not None
        else None
    )

    device_mesh = None
    if "HYBRID_SHARD" in sharding_strategy:
        global DEVICE_MESH
        if DEVICE_MESH is None:
            world_size = comm.get_world_size()
            hybrid_gpu_num = min(world_size, meta_model_cfg.hybrid_gpu_num)
            assert world_size % hybrid_gpu_num == 0
            DEVICE_MESH = init_device_mesh("cuda", (world_size // hybrid_gpu_num, hybrid_gpu_num))
            _set_pg_timeout(timedelta(minutes=30), DEVICE_MESH.get_group(mesh_dim=0))
            _set_pg_timeout(timedelta(minutes=30), DEVICE_MESH.get_group(mesh_dim=1))
        device_mesh = DEVICE_MESH

    params = sum([p.numel() for p in model.parameters()]) / 1e6
    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad]) / 1e6
    logger.info(f"before FSDP, {model_cfg._class_name} with total params: {params:.1f}M, "
                f"trainable params: {trainable_params:.1f}M")

    model = FSDP(
        module=model,
        sharding_strategy=ShardingStrategy[sharding_strategy],
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=MixedPrecision(
            param_dtype=common.to_torch_dtype(model_cfg.weight_dtype),
            reduce_dtype=common.to_torch_dtype(reduce_dtype),
            buffer_dtype=common.to_torch_dtype(buffer_dtype)
        ),
        device_id=device_id,
        sync_module_states=sync_module_states,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        device_mesh=device_mesh,
        use_orig_params=use_orig_params
    )

    logger.info(f"wrapped FSDP for {model_cfg._class_name}: {model}")
    params = sum([p.numel() for p in model.parameters()]) / 1e6
    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad]) / 1e6
    logger.info(f"after FSDP, {model_cfg._class_name} with total params: {params:.1f}M, "
                f"trainable params: {trainable_params:.1f}M")

    model.train(training)
    return model


@FSDP_WRAP_POLICY_REGISTRY.register()
def dit_wrap_policy(module, recurse, **kwargs):
    from project.models.backbone.dit import WanAttentionBlock
    wanmodel_modules_to_wrap = (WanAttentionBlock, LoraLayer)
    if recurse:
        return True
    if module.layer_name.endswith(".base_layer") and isinstance(module, torch.nn.Linear):
        return True
    if module.layer_name.endswith(".modules_to_save.default"):
        return True
    return isinstance(module, wanmodel_modules_to_wrap)


@FSDP_WRAP_POLICY_REGISTRY.register()
def t5_wrap_policy(module, recurse, **kwargs):
    from project.models.text_encoder.t5 import T5SelfAttention
    t5encoder_modules_to_wrap = (T5SelfAttention, LoraLayer)
    if recurse:
        return True
    if module.layer_name.endswith(".base_layer") and isinstance(module, torch.nn.Linear):
        return True
    if module.layer_name.endswith(".modules_to_save.default"):
        return True
    return isinstance(module, t5encoder_modules_to_wrap)
