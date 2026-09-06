from typing import Optional, Union

from project.diffusion.timesteps import Timesteps


class TrainingTimesteps(Timesteps):
    def __init__(
        self,
        T: Union[int, float],
        shift: Optional[float] = None,
        dynamic_shift_in_training: bool = False,
        base_shift: float = 1.0,
        max_shift: float = 1.0,
        base_seqlen: int = 32760,  # 832x480 81frames
        max_seqlen: int = 32760,
        **kwargs
    ):
        super().__init__(T=T, shift=shift)
        self.dynamic_shift_in_training = dynamic_shift_in_training
        self.base_shift = base_shift
        self.max_shift = max_shift
        self.base_seqlen = base_seqlen
        self.max_seqlen = max_seqlen

        if self.dynamic_shift_in_training:
            assert self.shift is None, "shift should be None when dynamic_shift_in_training is True"
