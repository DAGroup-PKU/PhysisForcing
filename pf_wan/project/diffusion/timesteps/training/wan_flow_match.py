from typing import Sequence, Union

import torch

from project.diffusion.timesteps import TIMESTEP_REGISTRY
from project.diffusion.timesteps.training import TrainingTimesteps


@TIMESTEP_REGISTRY.register()
class WanFlowMatchTrainingTimesteps(TrainingTimesteps):
    """
    Matches DiffSynth FlowMatchScheduler(\"Wan\") with ``set_timesteps(1000, training=True)``:
    1000 sigma bins and ``shift`` (default 5), ``timesteps = sigmas * 1000``.
    Each training step samples a random index into this table (uniform), same as
    ``FlowMatchSFTLoss`` with full ``[min, max)`` index range.
    """

    def __init__(
        self,
        T: Union[int, float] = 1000,
        shift: float = 5.0,
        num_steps: int = 1000,
        denoising_strength: float = 1.0,
        **kwargs,
    ):
        safe_kwargs = {
            k: kwargs[k]
            for k in (
                "dynamic_shift_in_training",
                "base_shift",
                "max_shift",
                "base_seqlen",
                "max_seqlen",
            )
            if k in kwargs
        }
        super().__init__(T=T, shift=None, **safe_kwargs)
        self.sigma_shift = float(shift)
        self.num_steps = int(num_steps)
        self.den_strength = float(denoising_strength)
        self._flow_timesteps = self._build_flow_timesteps()

    @staticmethod
    def _wan_sigmas_timesteps(
        num_inference_steps: int,
        denoising_strength: float,
        shift: float,
        num_train_timesteps: int,
    ):
        sigma_min = 0.0
        sigma_max = 1.0
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * float(num_train_timesteps)
        return sigmas, timesteps

    def _build_flow_timesteps(self) -> torch.Tensor:
        n_train = int(self.T) if isinstance(self.T, (int, float)) else 1000
        _, timesteps = self._wan_sigmas_timesteps(
            self.num_steps,
            self.den_strength,
            self.sigma_shift,
            n_train,
        )
        return timesteps.contiguous()

    def sample(
        self,
        size: Sequence[int],
        seqlens: Sequence[int],
        device: torch.device,
    ):
        table = self._flow_timesteps.to(device=device, dtype=torch.float32)
        idx = torch.randint(
            0, self.num_steps, size, device=device, dtype=torch.long
        )
        return table[idx]
