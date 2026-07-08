from typing import List, Union

import torch

from project.diffusion.samplers import SAMPLER_REGISTRY, Sampler
from project.utils.common import expand_dims


@SAMPLER_REGISTRY.register()
class EulerSampler(Sampler):
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
        if isinstance(pred, list):
            return [self.step_to(pred[i], x_t[i], t[i:i+1], s[i:i+1]) for i in range(len(pred))]

        # Step from x_t to x_s.
        pred_x_0, pred_x_T = self.schedule.convert_from_pred(pred, x_t, t)
        pred_x_s = self.schedule.forward(pred_x_0, pred_x_T, s)
        return pred_x_s
