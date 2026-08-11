"""Single-pixel imaging semantic segmentation framework"""

from . import datasets
from . import models
from . import losses
from . import metrics
from . import reconstruction
from . import utils
from .trainer import SegmentationTrainer

__version__ = '1.0.0'

__all__ = [
    'datasets',
    'models',
    'losses',
    'metrics',
    'reconstruction',
    'utils',
    'SegmentationTrainer',
]