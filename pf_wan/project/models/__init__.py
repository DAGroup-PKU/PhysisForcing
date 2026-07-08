from project.utils import Registry

MODEL_REGISTRY = Registry("MODEL")

from .module import *
from .backbone import *
from .autoencoder import *
from .text_encoder import *

META_MODEL_REGISTRY = Registry("META_MODEL")

from .meta_model import *