import contextlib
import logging
import os
import time
from random import Random

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from tqdm import tqdm

from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video

from project.data import BaseDataLoader
from project.diffusion import SAMPLER_REGISTRY, SCHEDULE_REGISTRY, TIMESTEP_REGISTRY
from project.distributed.unified_parallel import SPDistForward, get_unified_parallel_world_size, init_unified_parallel
from project.engines import ENGINE_REGISTRY, DefaultEngine
from project.utils import CfgNode, comm, hdfs, maybe_download, maybe_upload

from project.subtools.depth_anything_v2.dpt import DepthAnythingV2
from project.subtools.vjepa2 import (
    VJEPA2FeatureExtractor,
    HighLevelPhysicsHead,
    LowLevelPhysicsHead,
    DiTBlockFeatureCapturer,
    project_tracks_to_pixel_mask,
    downsample_mask_to_grid,
    select_aligned_tokens_with_shared_indices,
    relational_alignment_loss,
)

logger = logging.getLogger()


@ENGINE_REGISTRY.register()
class DiffusionFinetuning(DefaultEngine):
    def __init__(self, config: CfgNode):
        super().__init__(config)
        self.device = comm.get_device()
        self.configure_persistence()
        self.build_dataloaders(config.data)
        self.build_models(config.meta_model)
        self.build_optimizers(config.meta_model)
        self.configure_diffusion(config.diffusion)

        self.average_meter = {}
        # Gradient accumulation. ``accum_steps`` micro steps share a
        # single optimizer step / log / save / iter increment. The
        # micro-step counter is what drives timestep / noise seeds so
        # that all micro steps inside one accumulation window train on
        # *different* timesteps (otherwise accumulating is wasted).
        self.accum_steps = max(1, self.config.training.get("accum_steps", 1))
        self._micro_iter = self.iter * self.accum_steps
        self.writer = wandb.init(
            project=config.exp_name,
            name=config.proj_name,
            dir=os.path.join(config.output_dir, config.exp_name, config.proj_name),
            resume="allow",
            id=None
        ) if comm.get_rank() == 0 else None

        if self.sp_size > 1:
            self.sync_inputs = SPDistForward(name="inputs", comm_shape=True)
        else:
            self.sync_inputs = lambda inputs: [inputs]

        self.video_processor = VideoProcessor(vae_scale_factor=8)

        # Decide whether we need CoTracker / DepthAnything: either
        # the low-level loss is on (trajectory alignment, Sec. 3.2) or
        # the high-level alignment loss is on with
        # ``use_region_mask=True`` (Sec. 3.1 mask).
        _high_level_cfg_peek = config.training.get("high_level_loss", None)
        _need_track_stack = (
            config.training.low_level_loss.enabled
            or (
                _high_level_cfg_peek is not None
                and _high_level_cfg_peek.get("enabled", False)
                and _high_level_cfg_peek.get("use_region_mask", True)
            )
        )

        self.depth_anything = None
        self.cotracker = None
        if _need_track_stack:
            # loading cotracker (and DepthAnythingV2 for background filtering)
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }

            encoder = "vitl"
            self.depth_anything = DepthAnythingV2(**model_configs[encoder])
            self.depth_anything.load_state_dict(
                torch.load(f"./weights/depth_anything_v2/depth_anything_v2_{encoder}.pth", map_location='cpu')
            )
            self.depth_anything = self.depth_anything.to(self.device)
            self.depth_anything.eval()

            # CoTracker3 (offline) is the trajectory teacher for both the
            # low-level loss and the high-level region mask.
            self.cotracker = torch.hub.load(
                "./weights/cotracker",
                "cotracker3_offline",
                source="local",
            ).to(self.device)
            
        # High-level physics alignment (Sec. 3.3): frozen V-JEPA2 encodes the
        # GT video (teacher); the student projects the captured DiT feature H^l
        # through a trainable MLP psi and the loss is the L1 between the two
        # token-relation matrices over the mask-selected tokens.
        high_level_cfg = config.training.get("high_level_loss", None)
        if high_level_cfg is not None and high_level_cfg.get("enabled", False):
            logger.info("[high_level_loss] Building V-JEPA2 teacher encoder...")
            self.vjepa2 = VJEPA2FeatureExtractor(
                model_name_or_path=high_level_cfg.get(
                    "model_name_or_path",
                    "./weights/vjepa2-vitl-fpc64-256",
                ),
                frames_per_clip=high_level_cfg.get("frames_per_clip", None),
                crop_size=high_level_cfg.get("crop_size", None),
                dtype=torch.bfloat16,
            ).to(self.device)
            num_trainable = sum(
                p.numel() for p in self.vjepa2.parameters() if p.requires_grad
            )
            assert num_trainable == 0, (
                f"[high_level_loss] V-JEPA2 must be frozen, found {num_trainable} "
                f"trainable parameters."
            )
            logger.info(
                f"[high_level_loss] V-JEPA2 ready, "
                f"params={sum(p.numel() for p in self.vjepa2.parameters())/1e6:.1f}M (all frozen)"
            )

            # Trainable projection psi (Eq. 9). Targets V-JEPA2 hidden
            # dim by default; if config provides ``target_dim`` we use
            # that instead.
            target_dim = int(high_level_cfg.get("target_dim", self.vjepa2.hidden_size))
            head = HighLevelPhysicsHead(
                dit_dim=int(high_level_cfg.get("dit_dim", 5120)),  # Wan2.2 A14B
                target_dim=target_dim,
                hidden_dim=high_level_cfg.get("psi_hidden_dim", None),
            ).to(self.device, dtype=torch.float32)

            # Small head: wrap in DDP (broadcast rank-0 weights first so all
            # ranks start identical).
            if comm.get_world_size() > 1:
                for p in head.parameters():
                    torch.distributed.broadcast(p.data, src=0)
                self.physics_head = torch.nn.parallel.DistributedDataParallel(
                    head,
                    device_ids=[comm.get_local_rank()],
                    output_device=comm.get_local_rank(),
                    broadcast_buffers=False,
                    find_unused_parameters=False,
                )
            else:
                self.physics_head = head

            self.physics_block_idx = int(high_level_cfg.get("dit_block_idx", 20))
            self.physics_capturer = DiTBlockFeatureCapturer(
                self.backbone, block_idx=self.physics_block_idx
            )
            head_module = self.physics_head.module if hasattr(self.physics_head, "module") else self.physics_head
            logger.info(
                f"[high_level_loss] HighLevelPhysicsHead ready (DDP={isinstance(self.physics_head, torch.nn.parallel.DistributedDataParallel)}): "
                f"dit_dim={head_module.dit_dim} -> target_dim={head_module.target_dim}; "
                f"hook DiT block {self.physics_block_idx}"
            )
        else:
            self.vjepa2 = None
            self.physics_head = None
            self.physics_capturer = None

        # Low-level physics alignment (Sec. 3.2): tap the same DiT feature H^l,
        # refine it through a trainable MLP phi, reshape to the patch grid, and
        # run the trajectory loss as a soft-argmax against CoTracker3 points.
        low_level_cfg = config.training.get("low_level_loss", None)
        if low_level_cfg is not None and low_level_cfg.get("enabled", False):
            low_dim = int(low_level_cfg.get("low_dim", 128))
            low_head = LowLevelPhysicsHead(
                dit_dim=int(low_level_cfg.get("dit_dim", 5120)),  # Wan2.2 A14B
                low_dim=low_dim,
                hidden_dim=low_level_cfg.get("phi_hidden_dim", None),
            ).to(self.device, dtype=torch.float32)

            # Same FSDP-vs-DDP rationale as the high-level head.
            if comm.get_world_size() > 1:
                for p in low_head.parameters():
                    torch.distributed.broadcast(p.data, src=0)
                self.low_level_head = torch.nn.parallel.DistributedDataParallel(
                    low_head,
                    device_ids=[comm.get_local_rank()],
                    output_device=comm.get_local_rank(),
                    broadcast_buffers=False,
                    find_unused_parameters=False,
                )
            else:
                self.low_level_head = low_head

            # Reuse the same DiT block index as the high-level loss
            # whenever both are on, so both losses act on the same H^l
            # (matches the paper). The capturer is also shared — we
            # build one if high-level didn't already.
            self.low_level_block_idx = int(
                low_level_cfg.get("dit_block_idx", getattr(self, "physics_block_idx", 20))
            )
            if self.physics_capturer is None:
                self.physics_capturer = DiTBlockFeatureCapturer(
                    self.backbone, block_idx=self.low_level_block_idx
                )
            elif self.low_level_block_idx != self.physics_block_idx:
                logger.warning(
                    f"[low_level_loss] dit_block_idx={self.low_level_block_idx} differs from "
                    f"high_level dit_block_idx={self.physics_block_idx}; using the high-level "
                    f"capturer (block {self.physics_block_idx}) for both."
                )
                self.low_level_block_idx = self.physics_block_idx

            head_module = self.low_level_head.module if hasattr(self.low_level_head, "module") else self.low_level_head
            logger.info(
                f"[low_level_loss] LowLevelPhysicsHead ready "
                f"(DDP={isinstance(self.low_level_head, torch.nn.parallel.DistributedDataParallel)}): "
                f"dit_dim={head_module.dit_dim} -> low_dim={head_module.low_dim}; "
                f"hook DiT block {self.low_level_block_idx}"
            )
        else:
            self.low_level_head = None
            self.low_level_block_idx = None

        # Initialize the list of models that take an optimizer.step() each
        # iteration. We do this *before* we splice in physics_head so the
        # below membership check works on the freshly-loaded list.
        self.step_models = self.config.training.get("step_models", ["backbone"])

        if self.physics_head is not None:
            head_optim_cfg = high_level_cfg.get("optimizer", None)
            if head_optim_cfg is None:
                # Fall back to the backbone's optimizer setting if the
                # user didn't customize anything for the head.
                head_optim_cfg = dict(
                    config.meta_model.backbone.optimizer.to_dict()
                    if hasattr(config.meta_model.backbone.optimizer, "to_dict")
                    else config.meta_model.backbone.optimizer
                )
            else:
                head_optim_cfg = dict(
                    head_optim_cfg.to_dict()
                    if hasattr(head_optim_cfg, "to_dict")
                    else head_optim_cfg
                )
            logger.info(f"[high_level_loss] Building optimizer for physics_head: {head_optim_cfg}")
            self.optimizers["physics_head"] = torch.optim.AdamW(
                [p for p in self.physics_head.parameters() if p.requires_grad],
                **head_optim_cfg,
            )
            # Always step the physics_head alongside the backbone
            # whenever physics alignment is on.
            if "physics_head" not in self.step_models:
                self.step_models = list(self.step_models) + ["physics_head"]

            # Resume physics_head weights + optimizer state if a
            # checkpoint is present. Since ``physics_head`` is not in
            # ``meta_model`` it is not handled by ``build_optimizers``;
            # we do the plain (non-FSDP) load here.
            if self.resume is not None and "physics_head" in self.resume.get("models", {}):
                head_ckpt = maybe_download(self.resume["models"]["physics_head"])
                logger.info(f"[high_level_loss] Resuming physics_head from {head_ckpt}")
                sd = torch.load(head_ckpt, map_location="cpu")
                bare = self.physics_head.module if hasattr(self.physics_head, "module") else self.physics_head
                bare.load_state_dict(sd)
            if self.resume is not None and "physics_head" in self.resume.get("optimizers", {}):
                opt_ckpt = maybe_download(self.resume["optimizers"]["physics_head"])
                logger.info(f"[high_level_loss] Resuming optimizer_physics_head from {opt_ckpt}")
                self.optimizers["physics_head"].load_state_dict(
                    torch.load(opt_ckpt, map_location="cpu")
                )

        # Symmetric optimizer / resume / step_models setup for the
        # low-level head. Same fall-back (backbone optim cfg) as
        # high-level if the user didn't customise.
        if self.low_level_head is not None:
            low_optim_cfg = low_level_cfg.get("optimizer", None)
            if low_optim_cfg is None:
                low_optim_cfg = dict(
                    config.meta_model.backbone.optimizer.to_dict()
                    if hasattr(config.meta_model.backbone.optimizer, "to_dict")
                    else config.meta_model.backbone.optimizer
                )
            else:
                low_optim_cfg = dict(
                    low_optim_cfg.to_dict()
                    if hasattr(low_optim_cfg, "to_dict")
                    else low_optim_cfg
                )
            logger.info(f"[low_level_loss] Building optimizer for low_level_head: {low_optim_cfg}")
            self.optimizers["low_level_head"] = torch.optim.AdamW(
                [p for p in self.low_level_head.parameters() if p.requires_grad],
                **low_optim_cfg,
            )
            if "low_level_head" not in self.step_models:
                self.step_models = list(self.step_models) + ["low_level_head"]

            if self.resume is not None and "low_level_head" in self.resume.get("models", {}):
                head_ckpt = maybe_download(self.resume["models"]["low_level_head"])
                logger.info(f"[low_level_loss] Resuming low_level_head from {head_ckpt}")
                sd = torch.load(head_ckpt, map_location="cpu")
                bare = self.low_level_head.module if hasattr(self.low_level_head, "module") else self.low_level_head
                bare.load_state_dict(sd)
            if self.resume is not None and "low_level_head" in self.resume.get("optimizers", {}):
                opt_ckpt = maybe_download(self.resume["optimizers"]["low_level_head"])
                logger.info(f"[low_level_loss] Resuming optimizer_low_level_head from {opt_ckpt}")
                self.optimizers["low_level_head"].load_state_dict(
                    torch.load(opt_ckpt, map_location="cpu")
                )

    def configure_persistence(self):
        # find possible states
        self.save_dir = os.path.join(self.config.save_dir, self.config.exp_name, self.config.proj_name)
        states_dir = os.path.join(self.save_dir, "states")
        hdfs.mkdir(states_dir)
        steps = sorted([int(os.path.basename(path)) for path in hdfs.listdir(states_dir)])
        if len(steps) == 0:
            self.iter = 0
            return

        # state directories
        step = steps[-1]
        states_dir = os.path.join(states_dir, f"{step:010}")
        models_dir = os.path.join(states_dir, "models")
        optimizers_dir = os.path.join(states_dir, "optimizers")
        dataloaders_dir = os.path.join(states_dir, "dataloaders")

        def get_state(dir):
            names = [os.path.splitext(os.path.basename(path))[0] for path in hdfs.listdir(dir)]
            return {name: os.path.join(dir, f"{name}.pth") for name in names}

        # resume dict
        self.resume = dict(
            models=get_state(models_dir),
            optimizers=get_state(optimizers_dir),
            dataloaders=dataloaders_dir
        )
        self.iter = step

    def build_dataloaders(self, data_cfg):
        ckpt_path = None
        if self.resume:
            ckpt_path = self.resume["dataloaders"]
            ckpt_path = os.path.join(ckpt_path, f"rank_{comm.get_rank()}.json")
        self.dataloader = BaseDataLoader(data_cfg, ckpt_path)
    
    def build_models(self, meta_model_cfg):
        # initialize distributed environment
        if meta_model_cfg.get("ulysses_size", 1) > 1:
            init_unified_parallel(meta_model_cfg.ulysses_size)
            self.sp_size = get_unified_parallel_world_size()
        else:
            self.sp_size = 1
        super().build_models(meta_model_cfg)

    def build_optimizers(self, meta_model_cfg):
        self.optimizers = dict()
        for k, v in meta_model_cfg.items():
            if not isinstance(v, CfgNode) or not hasattr(v, "_class_name"):
                continue
            model = getattr(self, k)
            if not isinstance(model, nn.Module) or not v.get("trainable", False):
                continue
            logger.info(f"Building optimizer_{k}...")

            self.optimizers[k] = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                **v.optimizer
            )

            if self.resume is not None:
                assert v.get("fsdp", None) and v.fsdp.enabled
                ckpt_path = maybe_download(self.resume["optimizers"][k])
                optim_state_dict = torch.load(ckpt_path, map_location="cpu")
                logger.info(f"Resuming optimizer_{k} from {ckpt_path}")

                with FSDP.state_dict_type(
                    model,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                    FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
                ):
                    optim_state_dict = FSDP.optim_state_dict_to_load(model, self.optimizers[k], optim_state_dict)

                self.optimizers[k].load_state_dict(optim_state_dict)
                logger.info(f"Resuming optimizer_{k} from {ckpt_path} done")

            logger.info(f"Building optimizer_{k} done")

    def save(self):
        if (self.iter + 1) % self.config.save_interval == 0:
            logger.info(f"Saving state at iter {self.iter + 1}...")
            states_dir = os.path.join(self.save_dir, f"states/{(self.iter + 1):010}")
            models_dir = os.path.join(states_dir, "models")
            optimizers_dir = os.path.join(states_dir, "optimizers")
            dataloaders_dir = os.path.join(states_dir, "dataloaders")
            hdfs.mkdir(models_dir)
            hdfs.mkdir(optimizers_dir)
            hdfs.mkdir(dataloaders_dir)

            # save trainable models and optimziers
            for k, v in self.optimizers.items():
                logger.info(f"Saving {k}...")
                model = getattr(self, k)

                # Non-FSDP modules (e.g. DDP-wrapped or plain
                # nn.Module like ``physics_head``) don't support the
                # FSDP.state_dict_type context manager. Save them via
                # the regular state_dict path on rank-0.
                is_fsdp = isinstance(model, FSDP)
                if is_fsdp:
                    with FSDP.state_dict_type(
                        model,
                        StateDictType.FULL_STATE_DICT,
                        FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                        FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
                    ):
                        state_dict = model.state_dict()
                        optim_state_dict = FSDP.optim_state_dict(model, v)
                else:
                    # Unwrap DDP so the saved keys match the bare module.
                    bare = model.module if hasattr(model, "module") else model
                    state_dict = bare.state_dict()
                    optim_state_dict = v.state_dict()

                if comm.get_rank() == 0:
                    maybe_upload(state_dict, f"{k}.pth", models_dir)
                    maybe_upload(optim_state_dict, f"{k}.pth", optimizers_dir)
                    
                comm.barrier()
                logger.info(f"Saving {k} done")

                if getattr(self, f"{k}_ema", None):
                    logger.info(f"Saving {k}_ema...")
                    model = getattr(self, f"{k}_ema")
                    if isinstance(model, FSDP):
                        with FSDP.state_dict_type(
                            model,
                            StateDictType.FULL_STATE_DICT,
                            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                            FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
                        ):
                            state_dict = model.state_dict()
                    else:
                        bare = model.module if hasattr(model, "module") else model
                        state_dict = bare.state_dict()

                    if comm.get_rank() == 0:
                        maybe_upload(state_dict, f"{k}_ema.pth", models_dir)
                    comm.barrier()
                    logger.info(f"Saving {k}_ema done")

            # save dataloaders
            logger.info(f"Saving dataloaders...")
            state_dict = self.dataloader.state_dict
            maybe_upload(state_dict, f"rank_{comm.get_rank()}.json", dataloaders_dir)
            logger.info(f"Saving dataloaders done")

    def configure_diffusion(self, diffusion_cfg):
        # timestep
        timestep_cls = TIMESTEP_REGISTRY.get(diffusion_cfg.training_timestep._class_name)
        self.training_timesteps = timestep_cls(T=diffusion_cfg.T, **diffusion_cfg.training_timestep)
        # schedule
        schedule_cls = SCHEDULE_REGISTRY.get(diffusion_cfg.schedule._class_name)
        self.schedule = schedule_cls(T=diffusion_cfg.T, **diffusion_cfg.schedule)
        # sampler
        sampler_cls = SAMPLER_REGISTRY.get(diffusion_cfg.sampler._class_name)
        self.sampler = sampler_cls(schedule=self.schedule, **diffusion_cfg.sampler)

    def run(self):
        pbar = tqdm(
            initial=self.iter,
            total=self.config.training.steps,
            dynamic_ncols=True,
            disable=comm.get_local_rank() != 0,
        )

        logger.info(f"start training (accum_steps={self.accum_steps})...")
        while self.iter < self.config.training.steps:
            dataloading_start_time = time.perf_counter()
            batch = next(self.dataloader)
            inputs = self.prepare_inputs(batch)
            # logger.info(f"video_shape: {inputs['latents'][0].shape}...")
            dataloading_time = time.perf_counter() - dataloading_start_time

            # Logs accumulated within one optimizer step (one
            # accumulation window). After the window closes we average
            # them and feed the result to the average_meter / wandb.
            window_log_dicts = []
            for sp_idx, inputs in enumerate(self.sync_inputs(inputs)):
                training_start_time = time.perf_counter()
                num_sp_groups = comm.get_world_size() // self.sp_size
                # Seed off ``_micro_iter`` (not ``iter``) so different
                # micro steps inside the same accumulation window
                # sample different timesteps / noises.
                self.seed = (
                    self.config.training.seed
                    + self._micro_iter * num_sp_groups
                    + comm.get_rank() // self.sp_size
                )
                self.random = Random(self.seed)
                self.generator = torch.Generator(device=self.device).manual_seed(self.seed)

                log_dict = self.training_loop(inputs)

                log_dict = dict(
                    batch_time=time.perf_counter() - training_start_time,
                    data_time=dataloading_time,
                    **log_dict
                )
                window_log_dicts.append(log_dict)
                pbar.set_postfix(log_dict)

            # One full dataloader batch consumed -> one micro step done.
            self._micro_iter += 1

            # Only run optimizer / log / save / iter increment when the
            # accumulation window is full.
            if self._micro_iter % self.accum_steps != 0:
                continue

            norm_dict = self.optimize(self.step_models)
            pbar.update()

            # Average per-window logs (so curves don't depend on
            # ``accum_steps``), then mix in the post-step grad norms.
            if window_log_dicts:
                avg_log_dict = {
                    k: sum(d[k] for d in window_log_dicts) / len(window_log_dicts)
                    for k in window_log_dicts[0].keys()
                }
            else:
                avg_log_dict = {}
            avg_log_dict.update(norm_dict)
            pbar.set_postfix(avg_log_dict)

            if not self.average_meter:
                self.average_meter = avg_log_dict
                self.average_meter["cnt"] = 1
            else:
                for k, v in avg_log_dict.items():
                    self.average_meter[k] = self.average_meter[k] * self.average_meter["cnt"] / (
                        self.average_meter["cnt"] + 1) + v / (self.average_meter["cnt"] + 1)
                self.average_meter["cnt"] += 1

            if (self.iter + 1) % self.config.log_interval == 0:
                self.average_meter.pop("cnt")
                # guard against all_gather_object timeouts so training doesn't crash
                try:
                    gather_meters = comm.all_gather_object(self.average_meter)
                except RuntimeError as e:
                    if "Timed out" in str(e) or "timeout" in str(e).lower():
                        logger.warning(f"[Steps {self.iter+1:07d}] all_gather_object timeout, using local meter only. Error: {e}")
                        # on timeout, use the local meter only (no global reduce)
                        gather_meters = [self.average_meter]
                    else:
                        raise

                if self.writer is not None:
                    average_meter = {k: sum([v[k] for v in gather_meters]) / len(gather_meters)
                                     for k in self.average_meter}
                    self.writer.log(average_meter, step=self.iter, commit=True)
                sstring = f"[Steps {self.iter+1:07d}/{self.config.training.steps:07d}]"
                keys = sorted(self.average_meter.keys())
                for k in keys:
                    sstring += f" | {k}: {self.average_meter[k]:.4f}"
                logger.info(sstring)
                self.average_meter = {}

            if (self.iter + 1) % self.config.save_interval == 0:
                self.save()

            self.iter += 1
        if self.writer is not None:
            self.writer.finish()

    def training_loop(self, inputs: dict):
        # Backward only. Optimizer step / zero_grad / EMA / grad-norm
        # are deferred to ``optimize`` and only fired once per
        # accumulation window from ``run``. Loss is divided by
        # ``accum_steps`` so the accumulated gradient equals the mean
        # over the window, matching a "real" batch of size accum_steps.
        loss_dict = self.run_step(inputs)
        loss = loss_dict["loss"] / self.accum_steps
        loss.backward()
        # Log the un-scaled loss values so curves stay comparable
        # regardless of ``accum_steps``.
        log_dict = {k: v.item() for k, v in loss_dict.items()}
        return log_dict

    def optimize(self, step_models):
        norm_dict = dict()
        for name in step_models:
            model = getattr(self, name)
            optim = self.optimizers[name]

            # FSDP-wrapped modules expose ``clip_grad_norm_`` themselves.
            # Plain ``nn.Module`` (e.g. the small physics_head) does not,
            # so we fall back to the functional util.
            meta_cfg = self.config.meta_model.get(name, None)
            clip_val = self.config.training.clip_grad_norm
            if meta_cfg is not None and hasattr(meta_cfg, "get"):
                clip_val = meta_cfg.get("clip_grad_norm", clip_val)
            if hasattr(model, "clip_grad_norm_"):
                norm = model.clip_grad_norm_(clip_val)
            else:
                norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=clip_val,
                )
            norm_dict[f"norm_{name[0].upper()}"] = norm.item()
            optim.step()
            optim.zero_grad()

            if getattr(self, f"{name}_ema", None) and meta_cfg is not None:
                ema_model = getattr(self, f"{name}_ema")
                ema_decay = meta_cfg.ema.decay
                with torch.no_grad():
                    for src, tgt in zip(model.parameters(), ema_model.parameters()):
                        if not src.requires_grad:
                            continue
                        tgt.detach().mul_(ema_decay).add_(src, alpha=1.-ema_decay)
        return norm_dict

    @torch.no_grad()
    def _build_physics_token_mask(
        self,
        gt_video_btchw: torch.Tensor,
        target_grid: tuple,
        high_level_cfg,
    ) -> torch.Tensor:
        """Compute the physics-informative token mask M for a single
        video sample.

        Parameters
        ----------
        gt_video_btchw : torch.Tensor
            ``[1, T, 3, H, W]`` in [-1, 1].
        target_grid : (T_tok, H_tok, W_tok)
            V-JEPA2 token grid.
        high_level_cfg : CfgNode
            ``training.high_level_loss`` section.

        Returns
        -------
        torch.Tensor
            ``[1, T_tok * H_tok * W_tok]`` binary mask.
        """
        T_tok, H_tok, W_tok = target_grid
        N_tok = T_tok * H_tok * W_tok

        # If the user disabled region focus, just align over everything.
        if not high_level_cfg.get("use_region_mask", True):
            return torch.ones((1, N_tok), device=gt_video_btchw.device, dtype=torch.float32)

        # Bail out gracefully if cotracker is not loaded (e.g. low-level
        # loss disabled and we have no track source). In that case fall
        # back to a uniform mask -- the loss is still meaningful, just
        # not region-focused.
        cotracker = getattr(self, "cotracker", None)
        if cotracker is None:
            logger.warning_once if False else logger.warning(
                "[high_level_loss] CoTracker not available, falling back to uniform mask"
            )
            return torch.ones((1, N_tok), device=gt_video_btchw.device, dtype=torch.float32)

        # Convert to the [0, 255] uint8-like format cotracker expects.
        # ``track_loss`` (see Wan2_1_I2V.track_loss) already does this
        # path; we replicate the pre-processing here so we don't have to
        # rewire that code.
        video_pix = ((gt_video_btchw + 1.0) * 127.5).clamp(0, 255)  # [1, T, 3, H, W]
        H_pix = video_pix.shape[3]
        W_pix = video_pix.shape[4]

        # 1) CoTracker3 -> dense trajectories.
        with torch.no_grad():
            video_tracks, video_visibility = cotracker(
                video_pix,
                grid_size=int(high_level_cfg.get("track_grid_size", 25)),
                grid_query_frame=0,
                backward_tracking=None,
            )

        # 2) First-frame depth weighting.
        depth_norm = None
        if getattr(self, "depth_anything", None) is not None:
            first_frame = video_pix[0, 0].permute(1, 2, 0).contiguous().cpu().numpy()
            # DepthAnythingV2 expects BGR uint8.
            import cv2
            first_bgr = cv2.cvtColor(first_frame.astype("uint8"), cv2.COLOR_RGB2BGR)
            with torch.no_grad():
                depth = self.depth_anything.infer_image(first_bgr)
            if depth.shape != (H_pix, W_pix):
                depth = cv2.resize(depth, (W_pix, H_pix), interpolation=cv2.INTER_LINEAR)
            d_min, d_max = float(depth.min()), float(depth.max())
            depth_norm = (depth - d_min) / max(d_max - d_min, 1e-6)
            depth_norm = depth_norm.clip(0.0, 1.0)

        # 3) Select top-K active tracks (eqs. 1-3).
        topk_tracks, topk_visibility = self.select_topk_active_tracks(
            video_tracks, video_visibility, K=None, depth_norm=depth_norm
        )

        # 4) Rasterize to pixel mask (eq. 4).
        pixel_mask = project_tracks_to_pixel_mask(
            topk_tracks.to(device=gt_video_btchw.device),
            topk_visibility.to(device=gt_video_btchw.device),
            H=H_pix,
            W=W_pix,
            radius=int(high_level_cfg.get("mask_radius", 2)),
        )                                                              # [1, T, H, W]

        # 5) Downsample to V-JEPA2 token grid and flatten.
        token_mask = downsample_mask_to_grid(
            pixel_mask, target_grid=target_grid
        )                                                              # [1, T_tok, H_tok, W_tok]
        return token_mask.view(1, -1)

    def run_step(self, inputs: dict):
        inputs = self.sample_timesteps(inputs, self.training_timesteps)
        inputs = self.sample_noises(inputs)
        inputs = self.add_noise(inputs)

        # Decide which physics losses are active for this step. Both
        # the low-level (Sec. 3.2) and the high-level (Sec. 3.3) loss
        # operate on the SAME hidden feature H^l from a chosen DiT
        # block, so we attach the forward hook whenever either one is
        # on.
        high_level_cfg = self.config.training.get("high_level_loss", None)
        use_high_level = (
            high_level_cfg is not None
            and high_level_cfg.get("enabled", False)
            and self.vjepa2 is not None
            and self.physics_head is not None
            and self.physics_capturer is not None
        )
        low_level_cfg = self.config.training.low_level_loss
        use_low_level = bool(low_level_cfg.enabled) and self.low_level_head is not None
        need_dit_hidden = use_high_level or use_low_level

        if need_dit_hidden:
            with self.physics_capturer.capture():
                pred = self.pred_single(self.backbone, inputs)
            dit_hidden = self.physics_capturer.feature  # [B, L, D_dit]
        else:
            pred = self.pred_single(self.backbone, inputs)
            dit_hidden = None

        x0 = self.convert_x0(inputs, pred) # list[torch.Size([16, 21, 88, 70])]
        pred = self.convert_pred(inputs, pred)
        target = self.convert_target(inputs)

        loss_trim = getattr(self, "diffusion_training_loss_latents", None)
        losses = []
        for i in range(len(pred)):
            pi, ti = pred[i].float(), target[i].float().detach()
            if loss_trim is not None:
                pi, ti = loss_trim(pi, ti)
            losses.append(F.mse_loss(pi, ti))
        loss_diffusion = torch.stack(losses).mean()

        # Low-level physics alignment (Sec. 3.2; eq. 5-8): per sample, the
        # student (DiT H^l -> phi -> grid_sample + spatial soft-argmax) predicts
        # 2D trajectories aligned to CoTracker3 reference tracks on the GT video.
        if use_low_level and dit_hidden is not None:
            assert len(x0) == len(inputs["latents"])
            losses_low_level = []
            B_samples = len(x0)
            is_low_ddp = isinstance(
                self.low_level_head, torch.nn.parallel.DistributedDataParallel
            )
            for i in range(B_samples):
                # Resolve DiT patch grid for this sample (same logic
                # as the high-level branch below).
                latent_i = inputs["latents"][i]
                C_lat, T_lat, H_lat, W_lat = latent_i.shape
                ph, pw, pt = 2, 2, 1
                F_p = T_lat // pt
                H_p = H_lat // ph
                W_p = W_lat // pw
                dit_grid = (F_p, H_p, W_p)
                valid_len = F_p * H_p * W_p

                hidden_i = dit_hidden[i : i + 1].float()                   # [1, L, D_dit]
                # DDP no_sync() on all but last sample so backward only
                # triggers a single all-reduce on phi per step.
                if is_low_ddp and i < B_samples - 1:
                    sync_ctx = self.low_level_head.no_sync()
                else:
                    sync_ctx = contextlib.nullcontext()
                with sync_ctx:
                    feat_i = self.low_level_head(
                        hidden_i,
                        dit_grid=dit_grid,
                        valid_len=valid_len,
                    )                                                       # [1, D_low, F_p, H_p, W_p]
                # ``track_loss`` expects pixel-space video and the
                # refined feature in 5D layout.
                video_i = inputs["videos"][i].to(dtype=feat_i.dtype)
                losses_low_level.append(
                    self.track_loss(
                        video_i,
                        feat_i,
                    )
                )

            loss_low_level_raw = torch.stack(losses_low_level).mean()
            loss_low_level = loss_low_level_raw * low_level_cfg.low_level_weight
        else:
            loss_low_level = torch.tensor(0.0, device=pred[0].device)
            loss_low_level = loss_low_level * low_level_cfg.low_level_weight

        # High-level physics alignment (Sec. 3.3, eq. 9-12): per sample, the
        # frozen V-JEPA2 teacher and the student (DiT H^l -> psi -> resize) each
        # produce token-relation matrices; the L1 between them over the
        # mask-selected tokens is the loss (gradient flows only via psi + backbone).
        if use_high_level and dit_hidden is not None:
            assert len(x0) == len(inputs["latents"]), \
                "pred x0 batch must match GT latents batch"

            losses_high_level = []
            B_samples = len(x0)
            # Wrap all but the last per-sample forward in ``no_sync()`` so only
            # the final one triggers the DDP backward all-reduce.
            is_ddp = isinstance(self.physics_head, torch.nn.parallel.DistributedDataParallel)
            for i in range(B_samples):

                gt_vid_tchw = inputs["videos"][i]                       # [T, 3, H, W], in [0, 255]
                gt_vid_btchw_255 = gt_vid_tchw.unsqueeze(0)             # [1, T, 3, H, W]
                # Convert to V-JEPA2's expected pixel-range proxy
                # ([-1, 1]); the wrapper renormalizes internally.
                gt_vid_btchw = gt_vid_btchw_255 / 127.5 - 1.0           # [1, T, 3, H, W] in [-1, 1]

                # Teacher tokens (no_grad).
                teacher_tokens = self.vjepa2.encode_video_m1p1(gt_vid_btchw)  # [1, N_jepa, D]

                # ---- Student tokens ----
                # Resolve DiT patch grid for this sample. Wan DiT uses
                # patch_size = (1, 2, 2) so F_p = T_lat, (H_p, W_p) =
                # (lat_H/2, lat_W/2).
                latent_i = inputs["latents"][i]  # [C_lat, T_lat, H_lat, W_lat]
                C_lat, T_lat, H_lat, W_lat = latent_i.shape
                ph, pw, pt = 2, 2, 1
                F_p = T_lat // pt
                H_p = H_lat // ph
                W_p = W_lat // pw
                dit_grid = (F_p, H_p, W_p)
                target_grid = self.vjepa2.token_grid_shape()
                # Slice this sample's hidden feature from the padded
                # batch tensor. The DiT zero-pads each sample's token
                # sequence up to seq_len.
                B_b = dit_hidden.size(0)
                assert B_b == len(x0), (
                    f"DiT hidden batch {B_b} mismatch batch {len(x0)}"
                )
                hidden_i = dit_hidden[i : i + 1]                       # [1, L, D_dit]
                valid_len = F_p * H_p * W_p
                # Push the projection through the trainable head.
                # ``physics_head`` lives in fp32 for stability; cast on
                # the way in/out.
                hidden_i_fp = hidden_i.float()
                # IMPORTANT: route through ``__call__`` (not a custom
                # method) so DDP can install backward hooks for
                # gradient all-reduce on ``physics_head``. Use
                # ``no_sync()`` for all but the last sample so DDP only
                # triggers all-reduce once per step.
                if is_ddp and i < B_samples - 1:
                    sync_ctx = self.physics_head.no_sync()
                else:
                    sync_ctx = contextlib.nullcontext()
                with sync_ctx:
                    student_tokens_all = self.physics_head(
                        hidden_i_fp,
                        dit_grid=dit_grid,
                        target_grid=target_grid,
                        valid_len=valid_len,
                    )                                                   # [1, N_jepa, D]

                # ---- Physics-informative mask M ----
                token_mask = self._build_physics_token_mask(
                    gt_vid_btchw,
                    target_grid=target_grid,
                    high_level_cfg=high_level_cfg,
                )                                                       # [1, N_jepa]

                # ---- Select K tokens consistently on both sides ----
                max_K = high_level_cfg.get("max_K", 512)
                student_sel, teacher_sel = select_aligned_tokens_with_shared_indices(
                    student_tokens=student_tokens_all,
                    teacher_tokens=teacher_tokens.float(),
                    mask=token_mask,
                    max_K=int(max_K) if max_K is not None else None,
                    min_K=int(high_level_cfg.get("min_K", 2)),
                )

                # ---- Relational L1 loss (eq. 12) ----
                loss_high_level_dict = relational_alignment_loss(
                    student_sel,
                    teacher_sel,
                    norm=high_level_cfg.get("norm", "l1"),
                    temperature=high_level_cfg.get("temperature", 1.0),
                )
                losses_high_level.append(loss_high_level_dict["loss"])

            loss_high_level_raw = torch.stack(losses_high_level).mean()
            high_level_weight = float(high_level_cfg.get("high_level_weight", 0.1))
            loss_high_level = loss_high_level_raw * high_level_weight
        else:
            loss_high_level = torch.tensor(0.0, device=pred[0].device)

        loss = loss_diffusion + loss_low_level + loss_high_level

        loss_dict = dict(loss=loss)
        return loss_dict
