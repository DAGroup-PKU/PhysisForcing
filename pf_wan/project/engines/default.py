import functools
import itertools
import logging
from contextlib import ExitStack

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file as safe_load

from project.distributed.fsdp import setup_fsdp
from project.engines import ENGINE_REGISTRY
from project.models import META_MODEL_REGISTRY, MODEL_REGISTRY
from project.utils import CfgNode, Dataclass, comm, common, maybe_download

logger = logging.getLogger()


@ENGINE_REGISTRY.register()
class DefaultEngine:
    def __new__(cls, config: CfgNode):
        # this __new__ function will initialize meta model class before engine class
        # such that engine can access, inherit or override meta model function conviently
        assert hasattr(config.meta_model, "_class_name"), "You must specify a meta model for engine via _class_name."
        meta_model_class = META_MODEL_REGISTRY.get(config.meta_model._class_name)
        return super().__new__(type("MixedEngine", (cls, meta_model_class), {}))

    def __init__(self, config: CfgNode):
        self.config = config

    def run(self):
        raise NotImplementedError

    def build_models(self, meta_model_cfg):
        def context_wrapper(*args, grad_enabled, autocast_cfg):
            def wrapper(orig_func):
                if hasattr(orig_func, "__self__"):
                    func = orig_func.__func__
                else:
                    func = orig_func

                @functools.wraps(func)
                def wrapped(*args, **kwargs):
                    with ExitStack() as stack:
                        if not grad_enabled:
                            stack.enter_context(torch.no_grad())
                        if autocast_cfg.enabled:
                            dtype = common.to_torch_dtype(autocast_cfg.dtype)
                            cache_enabled = autocast_cfg.get("cache_enabled", True)
                            stack.enter_context(torch.autocast("cuda", dtype=dtype, cache_enabled=cache_enabled))
                        return func(*args, **kwargs)

                if hasattr(orig_func, "__self__"):
                    self = orig_func.__self__
                    return wrapped.__get__(self, self.__class__)
                else:
                    return wrapped

            if args and callable(args[0]):
                return wrapper(args[0])
            else:
                return wrapper

        def annotate_layer_name(model, model_name):
            for name, module in model.named_modules():
                assert not hasattr(module, "layer_name")
                module.layer_name = f"{model_name}.{name}"

        for k, v in meta_model_cfg.items():
            if not isinstance(v, CfgNode) or not hasattr(v, "_class_name"):
                continue
            logger.info(f"Building {k}...")

            name, config = v._class_name, v.config
            model_cls = MODEL_REGISTRY.get(name)
            if hasattr(model_cls, "from_pretrained") and hasattr(config, "pretrained_model_name_or_path"):
                config.defrost()
                config.pretrained_model_name_or_path = maybe_download(config.pretrained_model_name_or_path)
                config.freeze()
                model = model_cls.from_pretrained(**config)
            else:
                with torch.device("meta"):
                    model = model_cls(**config)

            if isinstance(model, nn.Module):
                # pretrained model
                if v.get("weight", None):
                    weight = maybe_download(v.weight)
                    logger.info(f"Loading {name} from {weight}")
                    if weight.endswith(".safetensor") or weight.endswith(".safetensors"):
                        state_dict = safe_load(weight, device="cpu")
                    elif weight.endswith(".pkl"):
                        state_dict = torch.load(weight, map_location="cpu")
                    else:
                        state_dict = torch.load(weight, map_location="cpu", mmap=True)
                    msg = model.load_state_dict(state_dict, strict=False, assign=True)
                    logger.info(f"Loading {name} from {weight} with missing keys: {msg.missing_keys}")
                    logger.info(f"Loading {name} from {weight} with unexpected keys: {msg.unexpected_keys}")
                else:
                    model = model_cls(**config)

                if v.get("requires_grad", None) is not None:
                    model.requires_grad_(v.requires_grad)

                # lora: attach adapter and (optionally) load + merge a LoRA checkpoint.
                # For inference the config sets ``merge: true`` so the adapter is
                # folded into the base weights and unloaded before generation.
                if v.get("lora", None) and v.lora.enabled:
                    lora_config = v.lora.clone()
                    lora_config.defrost()
                    lora_config.pop("enabled")
                    lora_weight = lora_config.pop("weight", None)
                    lora_merge = lora_config.pop("merge", False)
                    lora_cfg = LoraConfig(**lora_config)
                    model = get_peft_model(model, lora_cfg)

                    if lora_weight is not None:
                        lora_weight = maybe_download(lora_weight)
                        logger.info(f"Loading lora from {lora_weight}")
                        state_dict = torch.load(lora_weight, map_location="cpu", mmap=True)
                        msg = set_peft_model_state_dict(model, state_dict)
                        logger.info(f"Loading lora from {lora_weight} with missing keys: {msg.missing_keys}")

                        if lora_merge:  # lora merge should only happen when given weights
                            model = model.merge_and_unload()
                            v.lora.defrost()
                            v.lora.enabled = False
                            v.lora.freeze()

                # check meta
                metas = [n for n, b in itertools.chain(model.named_parameters(), model.named_buffers()) if b.is_meta]
                assert not metas, f"{name} got meta tensor: {metas}"

                # eval mode + wrap forward with the configured no_grad / autocast context
                if v.get("training_state", None) is not None:
                    model.train(v.training_state)

                for wrap_key in v.get("wrapped_func", ["forward"]):
                    grad_enabled = v.get("grad_enabled", False)
                    autocast_cfg = v.get("autocast", Dataclass(enabled=False))
                    logger.info(f"Wrapping {k}.{wrap_key} with grad_enabled={grad_enabled}, "
                                f"autocast_enabled={autocast_cfg.enabled}")
                    setattr(model, wrap_key, context_wrapper(grad_enabled=grad_enabled, autocast_cfg=autocast_cfg)(
                        common.get_attribute(model, wrap_key)))

                # move to gpu
                setattr(model, "weight_dtype", common.to_torch_dtype(v.weight_dtype))
                if v.get("fsdp", None) is not None and v.fsdp.enabled:
                    annotate_layer_name(model, k)
                    model = setup_fsdp(model, meta_model_cfg, v)
                else:
                    model.to(device=comm.get_device(), dtype=common.to_torch_dtype(v.weight_dtype))

            setattr(self, k, model)
            logger.info(f"Building {k} done")
