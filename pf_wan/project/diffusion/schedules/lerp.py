import torch

from project.diffusion.schedules import SCHEDULE_REGISTRY, Schedule


@SCHEDULE_REGISTRY.register()
class LinearInterpolationSchedule(Schedule):
    """
    Linear interpolation schedule (lerp) is proposed by flow matching and rectified flow.
    It leads to straighter probability flow theoretically. It is also used by Stable Diffusion 3.
    <https://arxiv.org/abs/2209.03003>
    <https://arxiv.org/abs/2210.02747>

        x_t = (1 - t) * x_0 + t * x_T

    Can be either continuous or discrete.
    """

    def A(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - (t / self.T)

    def B(self, t: torch.Tensor) -> torch.Tensor:
        return t / self.T
