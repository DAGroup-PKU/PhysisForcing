import math
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from project.diffusion.timesteps import SamplingTimesteps, Timesteps
from project.models import META_MODEL_REGISTRY
from project.utils import common, local_seed, yield_seed



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
            # 与训练一致：视频 latent 首帧应用 clean latent（step 后钉回）
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






    
            
    
