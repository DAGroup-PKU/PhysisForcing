from project.utils import Registry

TIMESTEP_REGISTRY = Registry("timestep")

from .base import Timesteps
from .training import *
from .sampling import *
