import logging
import os
from random import Random

import torch
from tqdm import tqdm
from xfuser.core.distributed import get_sequence_parallel_world_size, init_distributed_environment, initialize_model_parallel

from project.diffusion import SAMPLER_REGISTRY, SCHEDULE_REGISTRY, TIMESTEP_REGISTRY
from project.distributed.unified_parallel import get_unified_parallel_world_size, init_unified_parallel
from project.engines import ENGINE_REGISTRY, DefaultEngine
from project.utils import CfgNode, comm, common, hdfs, maybe_download

logger = logging.getLogger()


@ENGINE_REGISTRY.register()
class GenerateI2V(DefaultEngine):
    def __init__(self, config: CfgNode):
        super().__init__(config)
        self.device = comm.get_device()
        self.build_models(config.meta_model)
        self.configure_diffusion(config.diffusion)

    def build_models(self, meta_model_cfg):
        # initialize distributed environment
        if meta_model_cfg.get("ulysses_size", None) and meta_model_cfg.get("ring_size", None):
            assert meta_model_cfg.ulysses_size * meta_model_cfg.ring_size == comm.get_world_size()
            init_distributed_environment(
                world_size=comm.get_world_size(),
                rank=comm.get_rank(),
                local_rank=comm.get_local_rank()
            )
            initialize_model_parallel(
                sequence_parallel_degree=comm.get_world_size(),
                ring_degree=meta_model_cfg.ring_size,
                ulysses_degree=meta_model_cfg.ulysses_size
            )
            self.sp_size = get_sequence_parallel_world_size()
        elif meta_model_cfg.get("ulysses_size", None):
            assert meta_model_cfg.ulysses_size == comm.get_world_size()
            init_unified_parallel(meta_model_cfg.ulysses_size)
            self.sp_size = get_unified_parallel_world_size()
        else:
            assert comm.get_world_size() == 1
            self.sp_size = 1
        super().build_models(meta_model_cfg)

    def configure_diffusion(self, diffusion_cfg):
        # timestep
        timestep_cls = TIMESTEP_REGISTRY.get(diffusion_cfg.sampling_timestep._class_name)
        self.sampling_timesteps = timestep_cls(T=diffusion_cfg.T, **diffusion_cfg.sampling_timestep)
        # schedule
        schedule_cls = SCHEDULE_REGISTRY.get(diffusion_cfg.schedule._class_name)
        self.schedule = schedule_cls(T=diffusion_cfg.T, **diffusion_cfg.schedule)
        # sampler
        sampler_cls = SAMPLER_REGISTRY.get(diffusion_cfg.sampler._class_name)
        self.sampler = sampler_cls(schedule=self.schedule, **diffusion_cfg.sampler)

    @torch.no_grad()
    def run(self):
        inference_cfg = self.config.inference

        from datetime import datetime

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        local_dir = os.path.join(self.config.output_dir, self.config.proj_name, now)
        os.makedirs(local_dir, exist_ok=True)
        if inference_cfg.get("hdfs_save_dir", None):
            hdfs_dir = os.path.join(inference_cfg.hdfs_save_dir, self.config.proj_name, self.config.exp_name, "videos")
            logger.info(f"Creating hdfs directory {hdfs_dir} for saving videos")
            hdfs.mkdir(hdfs_dir)

        # ``positive_prompt`` MUST point to a manifest file (local path
        # or hdfs://...), one entry per line, in the format:
        #     <text prompt>@@<conditioning image path>
        # The image path on the right of ``@@`` is the I2V conditioning
        # frame, so the prompt and image stay paired even when running
        # multiple samples in one job.
        if not (
            os.path.exists(inference_cfg.positive_prompt)
            or inference_cfg.positive_prompt.startswith("hdfs://")
        ):
            raise FileNotFoundError(
                f"inference.positive_prompt must be a manifest file with "
                f"'<text>@@<image_path>' lines (got: {inference_cfg.positive_prompt!r}). "
                f"Inline string prompts are no longer supported."
            )
        positive_prompt = maybe_download(inference_cfg.positive_prompt)
        with open(positive_prompt, "r") as f:
            indices = []
            positive_prompts = []
            filenames = []
            for index, line in enumerate(f.readlines()):
                items = line.strip().split('@@')
                text = items[0]
                img_paths = items[1].strip()

                positive_prompts.append(text.strip())
                filenames.append(img_paths)
                indices.append(index)

        if os.path.exists(inference_cfg.negative_prompt) or inference_cfg.negative_prompt.startswith("hdfs://"):
            negative_prompt = maybe_download(inference_cfg.negative_prompt)
            with open(negative_prompt, "r") as f:
                negative_prompt = f.read().strip()
        else:
            negative_prompt = inference_cfg.negative_prompt

        x0 = None
        save_filename = None
        save_seed = None
        for index, (positive_prompt, filename, fn_index) in enumerate(zip(positive_prompts, filenames, indices)):
            self.seed = inference_cfg.seed
            self.random = Random(self.seed)
            self.generator = torch.Generator(device=self.device).manual_seed(self.seed)

            # prepare inference inputs
            inputs = dict(
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                filename=filename
            )
            inputs = self.prepare_inputs(inputs)
            inputs, neg_inputs = inputs["inputs"], inputs["neg_inputs"]
            scale = torch.tensor([inference_cfg.guidance_scale], device=self.device)

            # I2V：首帧 latent 与训练/ DiffSynth 一致 — 初始钉死 + 每步 step 后钉回
            ff = inputs.get("first_frame_latents")
            if ff is not None and len(ff) > 0:
                inputs["x"] = self.pin_latents_first_video_frame(inputs["x"], ff)
                neg_ff = neg_inputs.get("first_frame_latents", ff)
                neg_inputs["x"] = self.pin_latents_first_video_frame(neg_inputs["x"], neg_ff)

            # generate
            with torch.autocast(
                device_type="cuda",
                dtype=common.to_torch_dtype(self.config.meta_model.get("autocast", {}).get("dtype", "float32")),
                enabled=self.config.meta_model.get("autocast", {}).get("enabled", False),
                cache_enabled=self.config.meta_model.get("autocast", {}).get("cache_enabled", True)
            ):
                for t_curr in tqdm(self.sampling_timesteps.timesteps, disable=comm.get_local_rank() != 0):
                    t_for_idx = t_curr.view(1) if t_curr.dim() == 0 else t_curr.reshape(1)
                    s_curr = self.sampling_timesteps.get_next_timesteps(t_for_idx)

                    t_tokens = self.expand_timestep_to_tokens(t_curr, inputs["x"][0], inputs["seq_len"])
                    s_tokens = self.expand_timestep_to_tokens(s_curr, inputs["x"][0], inputs["seq_len"])

                    inputs, neg_inputs = self.set_timesteps(t_tokens, inputs, neg_inputs)
                    if self.backbone.config.use_cfg_emb:
                        inputs = self.set_scale(scale, inputs)
                        pred = self.pred_single(self.backbone, inputs)
                    else:
                        pred = self.pred_cfg(self.backbone, inputs, neg_inputs, scale)
                    noisy_latents = self.step_to(pred, inputs, s_tokens)
                    if ff is not None and len(ff) > 0:
                        noisy_latents = self.pin_latents_first_video_frame(
                            noisy_latents, inputs["first_frame_latents"]
                        )
                    inputs, neg_inputs = self.set_noisy_latents(noisy_latents, inputs, neg_inputs)

            result = noisy_latents  # List[torch.Tensor]
            if index % comm.get_world_size() == comm.get_rank():  # each rank take one different result
                x0 = [res.cpu() for res in result]
                real_index = fn_index
                save_filename = filename
                save_seed = self.seed
                logger.info(f"receive one result, ready for decoding")

            if (index + 1) % comm.get_world_size() == 0 or index == len(positive_prompts) - 1:  # decode
                if x0 is not None:
                    logger.info(f"start decoding")
                    x0 = [x.to(self.device) for x in x0]
                    video = self.vae_decode(x0)[0]

                    img_path = (
                        save_filename.strip().split()[0]
                        if save_filename and save_filename.strip()
                        else str(real_index)
                    )
                    image_id = os.path.splitext(os.path.basename(img_path))[0] or f"{real_index:03d}"
                    out_name = f"{image_id}.mp4"
                    local_path = os.path.join(local_dir, out_name)
                    logger.info(f"save video to {local_path}")
                    common.save_video(video[None], local_path, fps=inference_cfg.fps, nrow=1)

                    if inference_cfg.get("hdfs_save_dir", None):
                        hdfs_path = os.path.join(hdfs_dir, out_name)
                        logger.info(f"copy video to {hdfs_path}")
                        hdfs.copy(local_path, hdfs_path)

                    x0 = None
                    save_filename = None
                    save_seed = None
                    comm.barrier()
                else:
                    logger.info(f"no result to decode")
                    comm.barrier()
