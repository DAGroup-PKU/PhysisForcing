import torch

from project.diffusion.schedules import SCHEDULE_REGISTRY, Schedule


@SCHEDULE_REGISTRY.register()
class CosineSchedule(Schedule):
    """
    Cosine diffusion schedule.
    Originally proposed as in improved DDPM (https://arxiv.org/abs/2102.09672)
    Continuous version was proposed in simple diffusion (https://arxiv.org/abs/2301.11093)
    Ours follows the continuous version without shift and clamping.

        x_t = cos(t * pi * 0.5) * x_0 + sin(t * pi * 0.5) * x_T

    Can be either used as continuous or discrete.
    """

    def A(self, t: torch.Tensor) -> torch.Tensor:
        return torch.cos(t / self.T * torch.pi * 0.5)

    def B(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sin(t / self.T * torch.pi * 0.5)
