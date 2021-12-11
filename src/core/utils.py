import os
import sys
from math import ceil
from datetime import datetime

import tensorflow as tf


def normalize(x, reduce_min=True, reduce_max=True):
  if reduce_min: x -= tf.reduce_min(x, axis=(-3, -2), keepdims=True)
  if reduce_max: x = tf.math.divide_no_nan(x, tf.reduce_max(x, axis=(-3, -2), keepdims=True))

  return x


def visualize(
    image,
    title=None,
    rows=2,
    cols=None,
    figsize=(16, 7.2),
    cmap=None
):
  import seaborn as sns
  import matplotlib.pyplot as plt

  sns.set_style("whitegrid", {'axes.grid' : False})

  if image is not None:
    if isinstance(image, (list, tuple)) or len(image.shape) > 3:  # many images
      plt.figure(figsize=figsize)
      cols = cols or ceil(len(image) / rows)
      for ix in range(len(image)):
        plt.subplot(rows, cols, ix+1)
        visualize(image[ix],
                 cmap=cmap,
                 title=title[ix] if title is not None and len(title) > ix else None)
      plt.tight_layout()
      return

    if isinstance(image, tf.Tensor): image = image.numpy()
    if image.shape[-1] == 1: image = image[..., 0]
    plt.imshow(image, cmap=cmap)
  
  if title is not None: plt.title(title)
  plt.axis('off')


def unfreeze_top_layers(
    model: tf.keras.Model,
    layers: float,
    freeze_bn: bool
):
  if not layers:
    model.trainable = False
    return

  model.trainable = True

  frozen_layer_ix = int((1-layers) * len(model.layers))
  for ix, l in enumerate(model.layers):
    l.trainable = (ix > frozen_layer_ix and
                   (not isinstance(l, tf.keras.layers.BatchNormalization) or
                    not freeze_bn))
  
  print(f'Unfreezing {layers:.0%} layers. Bottom-most is the {frozen_layer_ix}-nth layer ({model.layers[frozen_layer_ix].name}).')


def log_begin(fun_name, *args, **kwargs):
  now = datetime.now()

  print(f'[{fun_name} started at {now}]')

  max_param_size = max(list(map(len, kwargs.keys())) or [0])

  print('  args:')
  print('    ' + ', '.join(map(str, args)))

  for k, v in kwargs.items():
    print(f'  {k:<{max_param_size}} = {v}')
  print()

  return now

def log_end(fun_name: str, started: datetime = None):
  now = datetime.now()
  elapsed = now - started if started else 'unknown'
  print(f'[{fun_name} ended at {now} ({elapsed} elapsed)]')


def logged(fn):
  def _log_wrapper(*args, **kwargs):
    started = log_begin(fn.__name__, *args, **kwargs)
    r = fn(*args, **kwargs)
    log_end(fn.__name__, started)

    return r

  return _log_wrapper



def get_preprocess_fn(preprocess_fn):
  if not isinstance(preprocess_fn, str):
    return preprocess_fn
  
  *mod, fn_name = preprocess_fn.split('.')
  mod = '.'.join(mod)

  if mod:
    return getattr(sys.modules[mod], fn_name)

  return globals[fn_name]
