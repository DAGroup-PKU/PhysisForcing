from project.utils import Registry

ENGINE_REGISTRY = Registry("ENGINE")

from .default import DefaultEngine
from .generate_i2v import GenerateI2V
