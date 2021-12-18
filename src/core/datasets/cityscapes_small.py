import numpy as np

from .tfds import load, prepare, load_and_prepare


def classes(info):
  return np.asarray([
    'road','sidewalk','parking','rail track','building','wall','fence','guard rail','bridge',
    'tunnel','pole','polegroup','traffic light','traffic sign','vegetation','terrain','sky',
    'person','rider','car','truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle'
  ])


__all__ = [
  'load',
  'prepare',
  'load_and_prepare',
  'classes'
]
