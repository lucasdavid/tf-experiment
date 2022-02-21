import os
import pickle

import numpy as np


def restore(file):
  context = _as_numpy_iterator(file)
  step, data = map(np.concatenate, zip(*context))
  
  return step, data


def save(file, step, data):
  os.makedirs(os.path.dirname(file), exist_ok=True)

  with open(file, 'wb') as f:
    pickle.dump((step, data), f)


def _as_numpy_iterator(file):
  with open(file, "rb") as f:
    while True:
      try:
        yield pickle.load(f)
      except EOFError:
        break


__all__ = [
  'restore',
  'save',
]