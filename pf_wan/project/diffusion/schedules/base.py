from enum import Enum
from typing import List, Tuple, Union

import torch

from project.utils.common import expand_dims


class PredictionType(str, Enum):
    """
    x_0:
        Predict data sample.
    x_T:
        Predict noise sample.
        Proposed by DDPM (https://arxiv.org/abs/2006.11239)
        Proved problematic by zsnr paper (https://arxiv.org/abs/2305.08891)
    v_cos:
        Predict velocity dx/dt based on the cosine schedule (A_t * x_T - B_t * x_0).
        Proposed by progressive distillation (https://arxiv.org/abs/2202.00512)
    v_lerp:
        Predict velocity dx/dt based on the lerp schedule (x_T - x_0).
        Proposed by rectified flow (https://arxiv.org/abs/2209.03003)
    x_t:
        Convert x_0 and x_T to x_t.
        Useful for distillation loss on noisy latents.
    """

    x_0 = "x_0"
    x_T = "x_T"
    v_cos = "v_cos"
    v_lerp = "v_lerp"
    x_t = "x_t"


class Schedule:
    """
    Diffusion schedules are uniquely defined by T, A, B:

        x_t = A(t) * x_0 + B(t) * x_T, where t in [0, T]

    Schedules can be continuous or discrete.
    """

    def __init__(
        self,
        T: Union[int, float],
        pred_type: PredictionType,
        **kwargs
    ):
        self.T = T
        self.pred_type = pred_type

    def A(self, t: torch.Tensor) -> torch.Tensor:
        """
        Interpolation coefficient A.
        Returns tensor with the same shape as t.
        """
        raise NotImplementedError

    def B(self, t: torch.Tensor) -> torch.Tensor:
        """
        Interpolation coefficient B.
        Returns tensor with the same shape as t.
        """
        raise NotImplementedError

    def forward(
        self,
        x_0: Union[List[torch.Tensor], torch.Tensor],
        x_T: Union[List[torch.Tensor], torch.Tensor],
        t: torch.Tensor
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Diffusion forward function.
        """
        if isinstance(x_0, list):
            return [self.forward(x_0[i], x_T[i], t[i:i+1]) for i in range(len(x_0))]
        else:
            t = self._build_t_like(t, x_0)
            return self.A(t) * x_0 + self.B(t) * x_T

    def _build_t_like(self, t: torch.Tensor, x_like: torch.Tensor) -> torch.Tensor:
        """
        Build a tensor shaped like x_like from input t according to the current rule:
        - If x_like is 4D (B, C, H, W), create zeros like x_like and set channels [1:] to the
          last element from t (keeping dtype and device), effectively broadcasting a scalar over
          spatial dims and frames except channel 0.
        - Otherwise, fall back to expanding dims to match x_like.ndim.
        """
        flat = t.reshape(-1)
        t_first = flat[0].to(dtype=x_like.dtype, device=x_like.device)
        t_last = flat[-1].to(dtype=x_like.dtype, device=x_like.device)
        t_full = torch.full_like(x_like, fill_value=t_last, dtype=x_like.dtype, device=x_like.device)
        t_full[:, 0, :, :] = t_first
        return t_full

    def convert_from_pred(
        self,
        pred: Union[List[torch.Tensor], torch.Tensor],
        x_t: Union[List[torch.Tensor], torch.Tensor],
        t: torch.Tensor,
        pred_type: PredictionType = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert from prediction. Return predicted x_0 and x_T.
        """
        if isinstance(x_t, list):
            pred_x0s, pred_xTs = [], []
            for i in range(len(x_t)):
                pred_x_0, pred_x_T = self.convert_from_pred(pred[i], x_t[i], t[i:i+1])
                pred_x0s.append(pred_x_0)
                pred_xTs.append(pred_x_T)
            return pred_x0s, pred_xTs

        if pred_type is None:
            pred_type = self.pred_type

        t = self._build_t_like(t, x_t)
        A_t = self.A(t)
        B_t = self.B(t)

        if pred_type == PredictionType.x_T:
            pred_x_T = pred
            pred_x_0 = (x_t - B_t * pred_x_T) / A_t
        elif pred_type == PredictionType.x_0:
            pred_x_0 = pred
            pred_x_T = (x_t - A_t * pred_x_0) / B_t
        elif pred_type == PredictionType.v_cos:
            pred_x_0 = A_t * x_t - B_t * pred
            pred_x_T = A_t * pred + B_t * x_t
        elif pred_type == PredictionType.v_lerp:
            pred_x_0 = (x_t - B_t * pred) / (A_t + B_t)
            pred_x_T = (x_t + A_t * pred) / (A_t + B_t)
        else:
            raise NotImplementedError

        return pred_x_0, pred_x_T

    def convert_to_pred(
        self,
        x_0: Union[List[torch.Tensor], torch.Tensor],
        x_T: Union[List[torch.Tensor], torch.Tensor],
        t: torch.Tensor,
        pred_type: PredictionType = None
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Convert to prediction target given x_0 and x_T.
        """
        if isinstance(x_0, list):
            return [self.convert_to_pred(x_0[i], x_T[i], t[i:i+1]) for i in range(len(x_0))]

        if pred_type is None:
            pred_type = self.pred_type

        if pred_type == PredictionType.x_T:
            return x_T
        if pred_type == PredictionType.x_0:
            return x_0
        if pred_type == PredictionType.v_cos:
            t = self._build_t_like(t, x_0)
            return self.A(t) * x_T - self.B(t) * x_0
        if pred_type == PredictionType.v_lerp:
            return x_T - x_0
        if pred_type == PredictionType.x_t:
            return self.forward(x_0, x_T, t)
        raise NotImplementedError
