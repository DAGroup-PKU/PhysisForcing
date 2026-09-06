import math
from random import Random
from typing import List, Tuple, Union
import os
import cv2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from diffusers.utils import export_to_video
import imageio

from project.diffusion.timesteps import SamplingTimesteps, Timesteps
from project.models import META_MODEL_REGISTRY
from project.utils import comm, common, local_seed, yield_seed


@META_MODEL_REGISTRY.register()
class Wan2_1_I2V:
    def encode_prompts(self, texts) -> List[torch.Tensor]:
        ids, mask = self.tokenizer(texts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)
        return [u[:v] for u, v in zip(context, seq_lens)]

    def vae_encode(self, videos: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        videos: A list of videos each with shape [C, T, H, W].
        """
        return [
            self.vae.encode(u.unsqueeze(0)).float().squeeze(0)
            for u in videos
        ]

    def vae_decode(self, zs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        zs: A list of latents each with shape [lat_C, lat_T, lat_H, lat_W]
        """
        return [
            self.vae.decode(u.unsqueeze(0)).float().clamp_(-1, 1).squeeze(0)
            for u in zs
        ]

    @torch.no_grad()
    def prepare_inputs(self, batch: dict) -> dict:
        if self.backbone.training:
            return self.prepare_training_inputs(batch)
        else:
            return self.prepare_inference_inputs(batch)

    @torch.no_grad()
    def prepare_inference_inputs(self, batch: dict) -> dict:
        positive_prompt = batch["positive_prompt"]
        negative_prompt = batch["negative_prompt"]
        filename = batch["filename"]

        inference_cfg = self.config.inference
        img = Image.open(filename).convert("RGB")
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        max_area = inference_cfg.max_area
        F = inference_cfg.num_frames
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae.stride[1] //
            self.backbone.patch_size[1] * self.backbone.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae.stride[2] //
            self.backbone.patch_size[2] * self.backbone.patch_size[2])
        h = lat_h * self.vae.stride[1]
        w = lat_w * self.vae.stride[2]

        max_seq_len = ((F - 1) // self.vae.stride[0] + 1) * lat_h * lat_w // (
            self.backbone.patch_size[1] * self.backbone.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        noise = torch.randn(
            self.vae.z_dim,
            F // self.vae.stride[0] + 1,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=self.generator,
            device=self.device
        )

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=self.vae.stride[0], dim=1), msk[:, 1:]
        ], dim=1)
        msk = msk.view(1, msk.shape[1] // self.vae.stride[0], self.vae.stride[0], lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        context = self.encode_prompts([positive_prompt])

        cond_lat = self.vae_encode([
            torch.concat([
                torch.nn.functional.interpolate(img[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1),
                torch.zeros(3, F - 1, h, w)
            ], dim=1).to(self.device)
        ])[0]
        first_frame_latents = [cond_lat[:, 0:1, :, :].contiguous().clone()]
        y = torch.concat([msk, cond_lat])

        inputs = dict(
            # dit parameters
            x          = [noise],
            context    = context,
            seq_len    = max_seq_len,
            y          = [y],
            # match training: apply the clean latent to the first video-latent frame
            first_frame_latents = first_frame_latents,
            # misc parameters
            batch_size = torch.tensor(1, dtype=torch.long, device=self.device)
        )
        neg_inputs = self.prepare_neg_inputs(inputs, neg_texts=[negative_prompt])
        inputs = self.prepare_cfg_embed(inputs, neg_inputs, torch.tensor([inference_cfg.guidance_scale], device=self.device))

        return dict(inputs=inputs, neg_inputs=neg_inputs)


    def masks_like(self, tensor, zero=False, generator=None, p=0.2):
        assert isinstance(tensor, list)
        out1 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

        out2 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

        if zero:
            if generator is not None:
                for u, v in zip(out1, out2):
                    random_num = torch.rand(
                        1, generator=generator, device=generator.device).item()
                    if random_num < p:
                        u[:, 0] = torch.normal(
                            mean=-3.5,
                            std=0.5,
                            size=(1,),
                            device=u.device,
                            generator=generator).expand_as(u[:, 0]).exp()
                        v[:, 0] = torch.zeros_like(v[:, 0])
                    else:
                        u[:, 0] = u[:, 0]
                        v[:, 0] = v[:, 0]
            else:
                for u, v in zip(out1, out2):
                    u[:, 0] = torch.zeros_like(u[:, 0])
                    v[:, 0] = torch.zeros_like(v[:, 0])

        return out1, out2


    @torch.no_grad()
    def prepare_training_inputs(self, batch: dict) -> dict:
        texts = batch["text"]
        videos = batch["video"].to(comm.get_device(), non_blocking=True)  # Tensor: [B, T, H, W, C], ranging [0, 255]
        bsz = len(texts)
        self.seed = self.config.training.seed + self.iter * comm.get_world_size() + comm.get_rank()
        self.random = Random(self.seed)
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)

        # interpolate and pick frames
        if hasattr(self, "interpolator"):
            videos = videos.to(dtype=self.interpolator.weight_dtype,)
            videos = [self.interpolator(videos[i]).float() for i in range(bsz)]  # [T, C, H, W], ranging [0, 255]
        else:
            videos = [video.permute(0, 3, 1, 2) for video in videos.unbind(0)]  # [T, C, H, W], ranging [0, 255]

        # vae encode
        ys = []             # List[Tensor]
        seqlens = []        # List[int]
        latents = []        # List[Tensor]

        for i in range(bsz):
            video = videos[i]
            t = min(1 + ((video.shape[0] - 1) // self.vae.stride[0]) * self.vae.stride[0], self.config.data.max_frames)
            video = video[:t]
            h, w = video.shape[2:]

            assert h % self.vae.stride[1] == 0, f"h={h} must be divisible by stride={self.vae.stride[1]}"
            assert w % self.vae.stride[2] == 0, f"w={w} must be divisible by stride={self.vae.stride[2]}"
            lat_h = h // self.vae.stride[1]
            lat_w = w // self.vae.stride[2]

            # interpolate size
            video = video.to(dtype=torch.uint8)  # to [0, 255]
            video = (video / 255.) * 2. - 1.  # to [-1, 1]
            video = video.permute(1, 0, 2, 3)  # C, T, H, W

            # condition
            msk = torch.ones(1, t, lat_h, lat_w, device=self.device)
            msk[:, 1:] = 0
            msk = torch.concat([
                torch.repeat_interleave(msk[:, 0:1], repeats=self.vae.stride[0], dim=1), msk[:, 1:]
            ], dim=1)
            msk = msk.view(1, msk.shape[1] // self.vae.stride[0], self.vae.stride[0], lat_h, lat_w)
            msk = msk.transpose(1, 2)[0]

            u = torch.cat([video[:,:1], torch.zeros(3, t - 1, h, w).to(self.device)], dim=1)
            u = u.to(dtype=self.vae.weight_dtype)
            y = self.vae.encode(u.unsqueeze(0)).float().squeeze(0)
            y = torch.cat([msk, y])
            ys.append(y)

            # seqlens
            seqlen = ((t - 1) // self.vae.stride[0] + 1) * lat_h * lat_w // (
                self.backbone.patch_size[1] * self.backbone.patch_size[2])
            seqlens.append(seqlen)

            # real video
            video = video.to(dtype=self.vae.weight_dtype)
            latent = self.vae.encode(video.unsqueeze(0)).float().squeeze(0)
            latents.append(latent)

        max_seq_len = max(seqlens)
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size
        max_seq_len = torch.tensor(max_seq_len, dtype=torch.long, device=self.device)

        seqlens = torch.tensor(seqlens, dtype=torch.long, device=self.device)
        batch_size = torch.tensor(bsz, dtype=torch.long, device=self.device)

        # text encode
        if self.config.training.get("dropout") and self.config.training.dropout > 0:
            texts = [text if self.random.random() > self.config.training.dropout else "" for text in texts]
        context = self.encode_prompts(texts)

        return dict(
            # dit parameters
            videos=videos,
            latents=latents,
            context=context,
            seq_len=max_seq_len,
            y=ys,
            # misc parameters
            batch_size=batch_size,
            seqlens=seqlens
        )

    @torch.no_grad()
    def prepare_neg_inputs(self, inputs: dict, neg_texts: List[str]) -> dict:
        neg_inputs = common.deepcopy_with_tensor_clone(inputs)
        neg_context = self.encode_prompts(neg_texts)
        neg_inputs["context"] = neg_context
        return neg_inputs

    def prepare_cfg_embed(self, inputs: dict, neg_inputs: dict, scale: torch.Tensor) -> Tuple[dict, dict]:
        batch_size = inputs["batch_size"]
        neg_context = neg_inputs["context"]
        inputs["cfg_scale"] = scale
        inputs["neg_embeds"] = torch.stack([torch.mean(neg_context[i], dim=0) for i in range(batch_size)])  # [B,4096]
        return inputs

    def sample_timesteps(
        self,
        inputs: dict,
        timesteps: Timesteps,
        return_timesteps: bool = False
    ) -> Union[dict, Tuple[dict, torch.Tensor]]:
        bsz = inputs["batch_size"]
        seqlens = inputs["seqlens"]
        with local_seed(self.seed):
            t = timesteps.sample((bsz,), seqlens=seqlens, device=self.device)
        self.seed = yield_seed(self.seed)
        inputs["t"] = t

        if return_timesteps:
            return inputs, t
        return inputs

    def index_timesteps(self, inputs: dict, timesteps: SamplingTimesteps) -> torch.Tensor:
        index = timesteps.index(inputs["t"])
        return index

    def sample_noises(
        self,
        inputs: dict,
        return_noises: bool = False
    ) -> Union[dict, Tuple[dict, List[torch.Tensor]]]:
        latents = inputs["latents"]
        noises = []
        for latent in latents:
            noise = torch.empty_like(latent).normal_(generator=self.generator)
            noises.append(noise)
        inputs["noises"] = noises

        if return_noises:
            return inputs, noises
        return inputs

    def expand_timestep_to_tokens(
        self,
        timestep: torch.Tensor,
        latent_ref: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build per-token timesteps for DiT (same layout as add_noise).
        First latent-time block uses 0; padding uses scalar timestep.
        """
        _, mask2 = self.masks_like([latent_ref], zero=True)
        temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
        if isinstance(seq_len, torch.Tensor):
            sl = int(seq_len.reshape(-1)[0].item())
        else:
            sl = int(seq_len)
        if temp_ts.size(0) < sl:
            ts0 = timestep.reshape(-1)[0]
            temp_ts = torch.cat([
                temp_ts.new_ones(temp_ts.size(0)) * ts0,
                temp_ts.new_ones(sl - temp_ts.size(0)) * ts0
            ])
        return temp_ts.unsqueeze(0)

    def pin_latents_first_video_frame(
        self,
        x_list: List[torch.Tensor],
        first_frame_latents: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """DiffSynth-style: force first latent time slice to conditioning latent."""
        if not first_frame_latents:
            return x_list
        out = []
        for i, x in enumerate(x_list):
            x = x.clone()
            ff = first_frame_latents[i]
            x[:, 0:1, :, :] = ff.to(device=x.device, dtype=x.dtype)
            out.append(x)
        return out

    def add_noise(self, inputs: dict, neg_inputs: dict = None) -> Union[dict, Tuple[dict, dict]]:
        latents = inputs["latents"]
        noises = inputs["noises"]
        timesteps = inputs["t"]
        noisy_latents = self.schedule.forward(latents, noises, timesteps)
        # I2V (DiffSynth / Wan): after flow-mixing, pin first latent time slice to clean x0.
        pinned = []
        for i, nl in enumerate(noisy_latents):
            nl = nl.clone()
            nl[:, 0:1, :, :] = latents[i][:, 0:1, :, :].to(dtype=nl.dtype, device=nl.device)
            pinned.append(nl)
        noisy_latents = pinned

        inputs["x"] = noisy_latents

        inputs["t"] = self.expand_timestep_to_tokens(timesteps, latents[0], inputs["seqlens"])

        if neg_inputs is not None:
            neg_inputs["latents"] = latents
            neg_inputs["noises"] = noises
            neg_inputs["t"] = inputs["t"]
            neg_inputs["x"] = noisy_latents
            return inputs, neg_inputs
        return inputs

    def diffusion_training_loss_latents(
        self, pred_i: torch.Tensor, target_i: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        I2V (DiffSynth-style): diffusion MSE only on latent time > 0 (first slice is condition).
        Expects [C, T_lat, H, W] tensors from backbone output / targets.
        """
        if pred_i.dim() != 4 or pred_i.shape[1] <= 1:
            return pred_i, target_i
        return pred_i[:, 1:, :, :], target_i[:, 1:, :, :]

    def pred_single(self, model: nn.Module, inputs: dict) -> List[torch.Tensor]:
        return model(
            x          = [x.to(dtype=model.weight_dtype) for x in inputs["x"]],
            t          = inputs["t"],
            context    = [context.to(dtype=model.weight_dtype) for context in inputs["context"]],
            seq_len    = inputs["seq_len"],
            y          = [y.to(dtype=model.weight_dtype) for y in inputs["y"]],
            cfg_scale  = inputs.get("cfg_scale", None),  # use cfg embed
            neg_embeds = inputs.get("neg_embeds", None)  # use cfg embed
        )

    def pred_cfg(self, model: nn.Module, inputs: dict, neg_inputs: dict, scale: torch.Tensor) -> List[torch.Tensor]:
        pred = self.pred_single(model, dict(
            x        = inputs["x"] + neg_inputs["x"],
            t        = torch.cat([inputs["t"], neg_inputs["t"]]),
            context  = inputs["context"] + neg_inputs["context"],
            seq_len  = inputs["seq_len"],
            y        = inputs["y"] + neg_inputs["y"]
        ))

        batch_size = inputs["batch_size"]
        pos = pred[:batch_size]
        neg = pred[batch_size:]
        scale = scale.expand((batch_size,))
        pred = [neg[i] + (pos[i] - neg[i]) * scale[i] for i in range(batch_size)]
        return pred

    def convert_pred(self, inputs: dict, pred: List[torch.Tensor], s: torch.Tensor = None) -> List[torch.Tensor]:
        pred_x_0, pred_x_T = self.schedule.convert_from_pred(
            pred=pred,
            x_t=inputs["x"],
            t=inputs["t"]
        )
        return self.schedule.convert_to_pred(
            x_0=pred_x_0,
            x_T=pred_x_T,
            t=s or inputs["t"],
            pred_type=self.config.training.loss_type
        )

    def convert_target(self, inputs: dict, s: torch.Tensor = None) -> List[torch.Tensor]:
        return self.schedule.convert_to_pred(
            x_0=inputs["latents"],
            x_T=inputs["noises"],
            t=s or inputs["t"],
            pred_type=self.config.training.loss_type
        )

    def set_timesteps(
        self,
        t: torch.Tensor,
        inputs: dict,
        neg_inputs: dict = None
    ) -> Union[dict, Tuple[dict, dict]]:
        inputs["t"] = t
        if neg_inputs is not None:
            neg_inputs["t"] = t
            return inputs, neg_inputs
        return inputs

    def set_noisy_latents(
        self,
        x: List[torch.Tensor],
        inputs: dict,
        neg_inputs: dict = None
    ) -> Union[dict, Tuple[dict, dict]]:
        inputs["x"] = x
        if neg_inputs is not None:
            neg_inputs["x"] = x
            return inputs, neg_inputs
        return inputs

    def set_scale(self, scale: torch.Tensor, inputs: dict) -> dict:
        inputs["cfg_scale"] = scale
        return inputs

    def set_latents(
        self,
        latents: List[torch.Tensor],
        inputs: dict,
        neg_inputs: dict = None
    ) -> Union[dict, Tuple[dict, dict]]:
        inputs["latents"] = latents
        if neg_inputs is not None:
            neg_inputs["latents"] = latents
            return inputs, neg_inputs
        return inputs

    def set_noises(
        self,
        noises: List[torch.Tensor],
        inputs: dict,
        neg_inputs: dict = None
    ) -> Union[dict, Tuple[dict, dict]]:
        inputs["noises"] = noises
        if neg_inputs is not None:
            neg_inputs["noises"] = noises
            return inputs, neg_inputs
        return inputs

    def step_to(
        self,
        pred: List[torch.Tensor],
        inputs: dict,
        s: torch.Tensor
    ) -> List[torch.Tensor]:
        return self.sampler.step_to(
            pred=pred,
            x_t=inputs["x"],
            t=inputs["t"],
            s=s,
            generator=self.generator
        )

    def get_endpoint(
        self,
        pred: List[torch.Tensor],
        inputs: dict
    ) -> List[torch.Tensor]:
        return self.sampler.get_endpoint(
            pred=pred,
            x_t=inputs["x"],
            t=inputs["t"]
        )
    
    def convert_x0(self, inputs: dict, pred: List[torch.Tensor]) -> List[torch.Tensor]:
        pred_x_0, pred_x_T = self.schedule.convert_from_pred( 
            pred=pred,
            x_t=inputs["x"],
            t=inputs["t"]
        )
        return pred_x_0


    def track_loss(self, video, input_feature, visualize=False, use_depth_weighting=True):
        """Low-level trajectory alignment loss L^{phy}_{low} (Embodied
        Forcing, Sec. 3.2; eq. 5-8).

        Parameters
        ----------
        video : torch.Tensor
            Ground-truth video ``[T, C, H_pix, W_pix]`` in [0, 255].
        input_feature : torch.Tensor
            Refined DiT feature already in spatial-temporal grid layout,
            shape ``[1, D_low, F_p, H_p, W_p]``. The caller is expected
            to have run ``low_level_head(dit_hidden, dit_grid=...,
            valid_len=...)`` so that gradients flow back to the
            backbone through the same ``H^l`` that the high-level
            loss uses. ``D_low`` is the projection dim chosen by
            ``LowLevelPhysicsHead``; ``(F_p, H_p, W_p)`` is the DiT
            patch grid for this clip (latent-time times Wan
            ``patch_size = (1, 2, 2)``).
        """
        # latent as GT video
        video = video.to(self.device, video.dtype)
        video = video.unsqueeze(0) # torch.Size([1, T, C, H, W])

        # 1. Obtain video tracks (CoTracker3, dense grid).
        video_tracks, video_visibility = self.cotracker(
            video,
            grid_size=25,
            grid_query_frame=0,
            backward_tracking=None,
        )


        def get_normalized_depth_map(depth_model, first_frame: np.ndarray, device: torch.device, target_size=None):
            """Return the normalized first-frame depth map."""
            with torch.no_grad():
                depth = depth_model.infer_image(first_frame)  # (H, W) raw depth

            if target_size is not None:
                target_H, target_W = target_size
                depth = cv2.resize(depth, (target_W, target_H), interpolation=cv2.INTER_LINEAR)

            depth_min = depth.min()
            depth_max = depth.max()
            eps = 1e-6
            depth_norm = (depth - depth_min) / max(depth_max - depth_min, eps)
            depth_norm = np.clip(depth_norm, 0.0, 1.0)

            return depth_norm

        depth_norm = None
        if use_depth_weighting and self.depth_anything is not None:
            first_frame_rgb = video[0, 0]                      # (C,H,W)
            first_frame_rgb_np = first_frame_rgb.permute(1, 2, 0).contiguous().cpu().numpy()  # (H,W,3)
            first_frame_bgr = cv2.cvtColor(first_frame_rgb_np, cv2.COLOR_RGB2BGR)  # DepthAnythingV2 expects BGR
            H, W = video.shape[3], video.shape[4]
            depth_norm = get_normalized_depth_map(depth_model=self.depth_anything, first_frame=first_frame_bgr, device=self.device, target_size=(H, W))


        # 2. K-most active tracks
        topk_tracks, topk_visibility = self.select_topk_active_tracks(video_tracks, video_visibility, K=None, depth_norm=depth_norm)

        if visualize and comm.get_rank() == 0:
            save_folder = "./saved_videos/track_loss"
            os.makedirs(save_folder, exist_ok=True)

            existing_files = os.listdir(save_folder)
            video_id = len(existing_files) // 3 + 1

            # Save the main video
            video_filename = os.path.join(save_folder, f"video_{video_id}.mp4")
            self.save_video_as_mp4(video, video_filename, fps=16)

            # Save the original tracks visualization
            original_filename = f"Original_{video_id}"
            # Local import: only for the optional debug visualization; the
            # CoTracker teacher itself loads via torch.hub (weights/cotracker).
            from project.subtools.cotracker.utils.visualizer import Visualizer
            vis = Visualizer(save_dir=save_folder, pad_value=120, linewidth=3, mode="rainbow", tracks_leave_trace=-1)
            vis.visualize(
                video,
                video_tracks,
                video_visibility,
                query_frame=0,
                filename = original_filename,
            )

            # Save the top-k tracks visualization
            topk_filename = f"Topk_{video_id}"
            vis.visualize(
                video,
                topk_tracks,
                topk_visibility,
                query_frame=0,
                filename=topk_filename,
            )

        # 3.Sample K tokens
        def normalize_coords(tracks, H, W):
            x = tracks[..., 0] / (W - 1) * 2 - 1
            y = tracks[..., 1] / (H - 1) * 2 - 1
            return torch.stack([x, y], dim=-1)

        def temporal_compress(x, frame_group_size):
            B, T, K, C = x.shape

            first_frame = x[:, :1]  # [B, 1, K, 2]

            remaining_frames = x[:, 1:]  # [B, T-1, K, 2]
            grouped_frames = remaining_frames.reshape(B, -1, frame_group_size, K, C)  #[B, num_groups, frame_group_size, K, 2]
            averaged_frames = grouped_frames.mean(dim=2)

            result = torch.cat([first_frame, averaged_frames], dim=1)  # [B, 1+(T-1)/frame_group_size, K, 2]
            return result

        # ``input_feature`` is the refined DiT feature in
        # ``[1, D_low, F_p, H_p, W_p]`` layout (see docstring above).
        assert input_feature.dim() == 5, (
            f"track_loss expects [1, D_low, F_p, H_p, W_p], got {tuple(input_feature.shape)}"
        )
        latent = input_feature

        # When CoTracker yields no usable tracks, return a zero loss that still
        # connects to ``latent`` so FSDP backward stays consistent across ranks.
        if topk_tracks.shape[2] == 0:
            return latent.sum() * 0.0

        frame_group_size = 4  # fixed for Wan-VAE
        topk_tracks = temporal_compress(topk_tracks, frame_group_size)

        # Pixel resolution of the GT video (used to normalise the
        # tracker's pixel-space coordinates to the [0, 1] range that the
        # spatial soft-argmax below also lives in).
        H_pix, W_pix = video.shape[3], video.shape[4]

        norm_tracks = normalize_coords(topk_tracks, H=H_pix, W=W_pix)  # [B, T, K, 2]
        K = topk_tracks.shape[2]

        B, C, T, H, W = latent.shape

        # grid coords at frame 0
        norm_grid_0  = norm_tracks[:, 0, :, :].unsqueeze(2)      # [B, K, 1, 2]

        # frame-0 feature map
        feat_frame0 = latent[:, :, 0, :, :]                      # [B, C, H, W]
        query_feats = F.grid_sample(
            feat_frame0,                         # [B, C, H, W]
            norm_grid_0,                         # [B, K, 1, 2]
            mode='bilinear',
            align_corners=True
        ).squeeze(-1).permute(0, 2, 1)           # [B, K, C]


        # 4. similarity between query and every pixel of the remaining frames
        other_feats = latent[:, :, 1:, :, :]                      # [B, C, T-1, H, W]
        other_feats = other_feats.permute(0, 2, 1, 3, 4)          # [B, T-1, C, H, W]
        other_feats = other_feats.reshape(B, T-1, C, H*W)         # [B, T-1, C, HW]

        # dot product: (B,K,C) x (B,T-1,C,HW) -> (B,K,T-1,HW)
        sim = torch.einsum('bkc,btch->bkth', query_feats, other_feats)  # [B,K,T-1,HW]
        sim = sim.view(B, K, T-1, H, W)                                # [B,K,T-1,H,W]

        # 5. softmax over the spatial dim -> expected coordinate
        sim_flat = sim.view(B, K, T-1, -1)  # [B,K,T-1,HW]
        logits = sim_flat.float()
        logits = logits - logits.amax(dim=-1, keepdim=True)
        weights = F.softmax(logits, dim=-1).to(dtype=sim_flat.dtype)

        yy = torch.arange(H, device=sim.device, dtype=weights.dtype).view(H, 1).expand(H, W)
        xx = torch.arange(W, device=sim.device, dtype=weights.dtype).view(1, W).expand(H, W)
        yy_flat = yy.reshape(1, 1, 1, -1)
        xx_flat = xx.reshape(1, 1, 1, -1)

        exp_x = (weights * xx_flat).sum(dim=-1)
        exp_y = (weights * yy_flat).sum(dim=-1)
        # Predicted positions are in feature-grid units in [0, W) /
        # [0, H); divide to put them in [0, 1] like the GT below.
        pred_xy_norm = torch.stack([exp_x / W, exp_y / H], dim=-1)

        # 6. gather GT coords and compute the loss
        # GT tracker coords are in pixel units; normalise by the pixel
        # resolution to match ``pred_xy_norm`` regardless of how much
        # the DiT patch grid has compressed the spatial dim.
        gt_xy = topk_tracks[:, 1:, :, :]                      # [B,T-1,K,2]
        gt_xy = gt_xy.permute(0,2,1,3)                        # [B,K,T-1,2]
        scale = torch.tensor([1.0 / W_pix, 1.0 / H_pix], device=gt_xy.device, dtype=gt_xy.dtype)
        gt_xy_norm = gt_xy * scale

        # RMSD (sqrt of the coordinate MSE); eps keeps the gradient finite as MSE -> 0.
        mse = F.mse_loss(pred_xy_norm, gt_xy_norm)
        loss = torch.sqrt(mse + 1e-6)
        return loss

    def select_topk_active_tracks(self, tracks, visibilitys, K=None, depth_norm=None):
        """
        tracks: [B, T, N, 2]
        visibilitys: [B, T, N]
        K: int or None, if None, select tracks with motion magnitude above average
        return: [B, T, K, 2], [B, T, K]
        """

        def sample_depth_weights(depth_norm: np.ndarray, track_positions: torch.Tensor):
            """Sample the per-track depth weight from the normalized depth map."""
            H, W = depth_norm.shape
            N = track_positions.shape[0]
            
            x_coords = track_positions[:, 0].clamp(0, W - 1).long().cpu().numpy()
            y_coords = track_positions[:, 1].clamp(0, H - 1).long().cpu().numpy()
            
            depth_weights = depth_norm[y_coords, x_coords]  # (N,)
            
            return torch.from_numpy(depth_weights).to(track_positions.device)
        
        B, T, N, _ = tracks.shape
        topk_tracks_list = []
        topk_visibility_list = []
        for b in range(B):
            track = tracks[b]  # [T, N, 2]
            visibility = visibilitys[b]  # [T, N]
            diffs = track[1:] - track[:-1]  # [T-1, N, 2]
            dists = torch.norm(diffs, dim=2)  # [T-1, N]
            motion_magnitude = dists.sum(dim=0)  # [N]

            if depth_norm is not None:
                depth_weights = sample_depth_weights(depth_norm, track[0])  # [N]
                motion_magnitude = motion_magnitude * depth_weights
            
            if K is None:
                # Adaptive threshold: mean motion magnitude (paper Eq. 3).
                threshold = motion_magnitude.mean()
                
                above_threshold_indices = torch.where(motion_magnitude >= threshold)[0]
                K = len(above_threshold_indices)
            
            _, topk_indices = torch.topk(motion_magnitude, K)
            
            topk_track = track[:, topk_indices, :]  # [T, K, 2]
            topk_visibility = visibility[:, topk_indices]  # [T, K]
            topk_tracks_list.append(topk_track)
            topk_visibility_list.append(topk_visibility)
        return torch.stack(topk_tracks_list, dim=0), torch.stack(topk_visibility_list, dim=0)


    def _vae_decode_raw(self, latents: torch.Tensor) -> torch.Tensor:
        """Call ``WanVAE.decode`` directly, *bypassing* the frozen no_grad wrapper.

        The engine's :func:`build_models` wraps ``vae.decode`` with
        ``torch.no_grad()`` because ``grad_enabled=False`` in the
        config. That is fine for the standard usage (decoding GT video
        for visualisation / metric loss), but it prevents the gradient
        from flowing back to a latent that *should* be optimised. The
        VAE parameters stay frozen either way (``requires_grad=False``
        on every weight), so calling the raw method does NOT make the
        VAE itself trainable; it merely keeps the activations on the
        autograd graph so upstream modules receive
        a gradient.

        Note: bfloat16 ``autocast`` is still applied because the engine
        wrapper sets ``autocast.enabled=True`` only at attribute level
        — when we bypass the wrapper we need to handle dtype manually.
        """
        vae = self.vae
        # ``FSDP``-wrapped or plain ``nn.Module`` — both expose the raw
        # method on the class. Falling back to ``__class__`` is more
        # robust than relying on the instance attribute (which is the
        # wrapped version).
        raw_decode = type(vae).decode
        weight_dtype = getattr(vae, "weight_dtype", torch.float32)
        with torch.autocast("cuda", dtype=weight_dtype, cache_enabled=True):
            out = raw_decode(vae, latents.to(dtype=weight_dtype))
        return out

    def decode_latent_to_video01(
        self,
        latents: torch.Tensor,
        *,
        preserve_grad: bool = False,
    ) -> torch.Tensor:
        """Decode a latent tensor to a pixel video in ``[0, 1]``.

        Parameters
        ----------
        latents : torch.Tensor
            Shape ``[C, T_lat, H_lat, W_lat]`` (single-sample) or
            ``[B, C, T_lat, H_lat, W_lat]`` (batched).
        preserve_grad : bool
            * ``False`` (default): use the frozen ``self.vae.decode``
              wrapper. Suitable for the GT branch.
            * ``True``: bypass the wrapper so gradients flow back to
              ``latents``. Use this on the prediction branch.

        Returns
        -------
        torch.Tensor
            Pixel video, shape ``[B, T, 3, H, W]``, values in ``[0, 1]``.
        """
        if latents.dim() == 4:
            latents = latents.unsqueeze(0)
        assert latents.dim() == 5, f"expected 4D or 5D latents, got {tuple(latents.shape)}"

        if preserve_grad:
            video = self._vae_decode_raw(latents)
        else:
            # GT branch: VAE.decode under no_grad. We still use the raw
            # method (forced no_grad) to handle batched input uniformly.
            with torch.no_grad():
                video = self._vae_decode_raw(latents).detach()

        # WanVAE outputs values roughly in [-1, 1]; convert to [0, 1] for V-JEPA2.
        video = video.clamp(-1.0, 1.0).add(1.0).mul(0.5)
        # WanVAE output layout is [B, C, T, H, W]; V-JEPA2 expects [B, T, C, H, W].
        video = video.permute(0, 2, 1, 3, 4).contiguous()
        return video
    
    def save_video_as_mp4(self, video, filename, fps=16):
        """Save a video tensor [T, C, H, W] to ``filename`` as an MP4 (default 16 fps)."""
        # to CPU numpy, [T, C, H, W] -> [T, H, W, C]
        video = video.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()

        if video.shape[3] != 3:
            raise ValueError(f"Expected 3 channels (RGB), but got {video.shape[3]} channels.")

        video = video.clip(0, 255).astype('uint8')

        with imageio.get_writer(filename, fps=fps) as writer:
            for frame in video:
                frame_bgr = frame[..., ::-1]  # RGB -> BGR
                writer.append_data(frame_bgr)