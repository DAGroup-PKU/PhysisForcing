from project.utils import Registry

SCHEDULE_REGISTRY = Registry("schedule")

from .base import Schedule
from .cos import CosineSchedule
from .lerp import LinearInterpolationSchedule
from .vp import DiscreteVariancePreservingSchedule
