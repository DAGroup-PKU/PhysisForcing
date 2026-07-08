from typing import Optional, Sequence, Union

import torch

from project.diffusion.timesteps import Timesteps
from project.utils import comm


class SamplingTimesteps(Timesteps):
    def __init__(
        self,
        T: Union[int, float],
        shift: Optional[float] = None,
        num_sampling_steps: Optional[int] = None,
        **kwargs
    ):
        super().__init__(T=T, shift=shift)
        self.num_sampling_steps = num_sampling_steps

        self.timesteps = None
        if self.num_sampling_steps is not None:
            self.set_timesteps(self.num_sampling_steps, comm.get_device(), shift=shift)

    def sample(
        self,
        size: Sequence[int],
        seqlens: Sequence[int],
        device: torch.device
    ):
        """
        This function will be called inside local_seed()
        so that there is no need to specify generator
        """
        assert self.timesteps is not None, "Timesteps must be set before calling sample"
        random_index = torch.randint(0, self.num_sampling_steps, size, device=device)
        timesteps = self.timesteps[random_index]
        return timesteps

    def set_timesteps(
        self,
        num_sampling_steps: int,
        device: torch.device,
        *,
        shift: Optional[float] = None
    ):
        raise NotImplementedError

    def get_next_timesteps(self, t: torch.Tensor) -> torch.Tensor:
        """
        Return next timesteps by t.
        Will return bound if index(t)+1 is reaching the end.
        """
        assert self.timesteps is not None, "Timesteps must be set before calling get_next_timesteps"
        curr_idx = self.index(t)
        next_idx = curr_idx + 1
        bound = 0.0 if self.is_continuous() else 0  # last step

        s = self.timesteps[next_idx.clamp_max(self.num_sampling_steps - 1)]
        s = s.where(next_idx < self.num_sampling_steps, bound)
        return s

    def get_timesteps_by_index(self, index: torch.Tensor) -> torch.Tensor:
        """
        Return timesteps by index.
        Will return bound if index is reaching the end.
        """
        assert self.timesteps is not None, "Timesteps must be set before calling get_timesteps_by_index"
        bound = 0.0 if self.is_continuous() else 0  # last step
        t = self.timesteps[index.clamp_max(self.num_sampling_steps - 1)]
        t = t.where(index < self.num_sampling_steps, bound)
        return t

    def index(self, t: torch.Tensor) -> torch.Tensor:
        """
        Find index by t.
        Return index of the same shape as t.
        Index is -1 if t not found in timesteps.
        """
        i, j = t.reshape(-1, 1).eq(self.timesteps).nonzero(as_tuple=True)
        idx = torch.full_like(t, fill_value=-1, dtype=torch.int)
        idx.view(-1)[i] = j.int()
        return idx

    def lerp(self, min_t: torch.Tensor, max_t: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """
        Return a random timestep between min_t (exclusive) and max_t (inclusive).
        Will handle continuous and discrete timesteps.
        """
        bsz = len(min_t)
        diff_t = max_t - min_t
        timepoints = 1. - torch.rand((bsz,), device=comm.get_device(), generator=generator)  # (0,1]
        timesteps = min_t + timepoints * diff_t
        if not self.is_continuous():
            timesteps = timesteps.round().long()
        return timesteps
