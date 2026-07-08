from typing import List, Union

import torch

from project.diffusion.schedules import Schedule
from project.utils.common import expand_dims


class Sampler:
    def __init__(self, schedule: Schedule, **kwargs):
        self.schedule = schedule

    def step_to(
        self,
        pred: Union[List[torch.Tensor], torch.Tensor],
        x_t: Union[List[torch.Tensor], torch.Tensor],
        t: torch.Tensor,
        s: torch.Tensor,
        generator: torch.Generator = None
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Steps from x_t at timestep t to x_s at timestep s. Returns x_s.
        """
        raise NotImplementedError

    def get_endpoint(
        self,
        pred: Union[List[torch.Tensor], torch.Tensor],
        x_t: Union[List[torch.Tensor], torch.Tensor],
        t: torch.Tensor
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Returns the endpoint of the denoising process.
        """
        pred_x_0, pred_x_T = self.schedule.convert_from_pred(pred, x_t, t)
        return pred_x_0
