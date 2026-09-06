from typing import Sequence

import torch

from project.diffusion.timesteps import TIMESTEP_REGISTRY
from project.diffusion.timesteps.training import TrainingTimesteps


@TIMESTEP_REGISTRY.register()
class UniformTrainingTimesteps(TrainingTimesteps):
    def sample(
        self,
        size: Sequence[int],
        seqlens: Sequence[int],
        device: torch.device
    ):
        t = torch.rand(size)
        if self.shift is not None:
            t = self.shift * t / (1 + (self.shift - 1) * t)
        if self.dynamic_shift_in_training:
            m = (self.max_shift - self.base_shift) / (self.max_seqlen - self.base_seqlen)
            b = self.base_shift - m * self.base_seqlen
            seqlens = torch.tensor(seqlens)
            mu = m * seqlens + b
            t = mu * t / (1 + (mu - 1) * t)
        t = t.mul(self.T).to(device)
        return t if self.is_continuous() else t.round().long().clamp(0, self.T - 1)
