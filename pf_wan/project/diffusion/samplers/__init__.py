from project.utils import Registry

SAMPLER_REGISTRY = Registry("sampler")

from .base import Sampler
from .euler import EulerSampler
from .consistency import ConsistencySampler
