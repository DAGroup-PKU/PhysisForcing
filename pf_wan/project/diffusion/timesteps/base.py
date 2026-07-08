from typing import Optional, Sequence, Union

import torch


class Timesteps:
    def __init__(
        self,
        T: Union[int, float],
        shift: Optional[float] = None,
        **kwargs
    ):
        self.T = T
        self.shift = shift

    def is_continuous(self) -> bool:
        return isinstance(self.T, float)

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
        raise NotImplementedError
