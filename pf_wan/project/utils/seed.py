import contextlib
import random
from typing import Optional

import numpy as np
import torch

__all__ = [
    "local_seed",
    "yield_seed",
]

@contextlib.contextmanager
def local_seed(seed: Optional[int]):
    """
    Create a local context with seed is set, but exit back to the original random state.
    If seed is None, do nothing.
    """
    if seed is not None:
        random_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(random_state)
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)
    else:
        yield


def yield_seed(seed, a=1103515245, c=12345, m=2**31):
    """
    Yield a random number from a given seed.
    """
    return (a * seed + c) % m
