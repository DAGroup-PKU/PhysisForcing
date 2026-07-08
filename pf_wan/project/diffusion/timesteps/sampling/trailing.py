from typing import Optional

import numpy as np
import torch

from project.diffusion.timesteps import TIMESTEP_REGISTRY
from project.diffusion.timesteps.sampling import SamplingTimesteps


@TIMESTEP_REGISTRY.register()
class TrailingSamplingTimesteps(SamplingTimesteps):
    def set_timesteps(
        self,
        num_sampling_steps: int,
        device: torch.device,
        *,
        shift: Optional[float] = None
    ):
        if shift is None:
            shift = self.shift

        t = np.arange(1.0, 0, -1.0 / num_sampling_steps)
        if shift is not None:
            t = shift * t / (1 + (shift - 1) * t)

        if isinstance(self.T, float):
            timesteps = torch.from_numpy(t)
        else:
            timesteps = t * self.T - 1
            timesteps = timesteps.round().astype(np.int64)
            timesteps = torch.from_numpy(timesteps)

        self.num_sampling_steps = num_sampling_steps
        self.timesteps = timesteps.to(device)
