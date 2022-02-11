# Copyright 2021 Lucas Oliveira David
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import tensorflow as tf

# region Data

def normalize(
    x: tf.Tensor,
    reduce_min: bool = True,
    reduce_max: bool = True,
    axis: Tuple[int] = (-3, -2)
):
  if reduce_min: x -= tf.reduce_min(x, axis=axis, keepdims=True)
  if reduce_max: x = tf.math.divide_no_nan(x, tf.reduce_max(x, axis=axis, keepdims=True))

  return x

def masked(
    x: tf.Tensor,
    maps: tf.Tensor
):
  return x * tf.image.resize(maps, x.shape[1:3])

# endregion

# region Model and Weights

def unfreeze_top_layers(
    model: tf.keras.Model,
    layers: Union[str, int, float],
    freeze_bn: bool,
):
  if not layers:
    model.trainable = False
    return

  model.trainable = True

  if isinstance(layers, str):
    idx = model.layers.index(model.get_layer(layers))
  elif isinstance(layers, float):
    idx = int((1 - layers) * len(model.layers))
  else:
    idx = layers
  
  if idx == 0 and not freeze_bn:
    model.trainable = True
  else:
    for ix, l in enumerate(model.layers):
      l.trainable = (ix > idx
                    and (not isinstance(l, tf.keras.layers.BatchNormalization)
                          or not freeze_bn))

  print(f'Unfreezing {1-idx/len(model.layers):.0%} of the model\'s layers '
        f'(layers={layers} freeze_bn={freeze_bn}). Bottom-most is the '
        f'{idx}-nth layer ({model.layers[idx].name}).')


def try_to_load_weights(model, weights, raise_on_failure: bool = False):
  try:
    model.load_weights(weights)
  except (OSError, FileNotFoundError):
    if raise_on_failure:
      raise

    print(f'Cannot restore weights from "{weights}".')

# endregion

# region Logging

def log_begin(
    fun_name: str,
    *args,
    with_margins: bool = True,
    with_arguments: bool = True,
    **kwargs
):
  now = datetime.now()

  if with_margins: print('_' * 65)
  print(fun_name)
  print(f'  started at: {now}')

  if with_arguments and args and kwargs:
    max_param_size = max(list(map(len, kwargs.keys())) or [0])

    print('  args:')
    print('    ' + '\n    '.join(map(str, args)))

    for k, v in kwargs.items():
      print(f'  {k:<{max_param_size}} = {v}')
    print()

  return now


def log_end(
    fun_name: str,
    started: datetime = None,
    with_margins: bool = True):
  if started:
    now = datetime.now()
    elapsed = now - started
    print(f'{fun_name} ended at {now} ({elapsed} elapsed)')
  if with_margins: print('_' * 65)


def logged(name=None, with_margins=True, with_arguments=True):
  def decorator(fn):
    def _log_wrapper(*args, **kwargs):
      started = log_begin(
        name or fn.__name__,
        *args,
        with_margins=with_margins,
        with_arguments=with_arguments,
        **kwargs
      )
      r = fn(*args, **kwargs)
      log_end(fn.__name__, started)

      return r
    return _log_wrapper
  return decorator


def get_preprocess_fn(preprocess_fn):
  if isinstance(preprocess_fn, str):
    *mod, fn_name = preprocess_fn.split('.')
    mod = '.'.join(mod)
    if mod:
      return getattr(sys.modules[mod], fn_name)
    return globals[fn_name]
  
  return preprocess_fn

# endregion

# region Generics

def dig(
    o: Optional[Dict],
    prop: Union[str, List[str]],
    default: Any = None, 
    required: bool = False):
  if not o:
    if required:
      raise TypeError(f'Cannot dig key {prop} from None.')

    return default

  if isinstance(prop, str):
    prop = prop.split('.')
  
  for p in prop:
    if p in o or required:
      o = o[p]
    else:
      return default

  return o


def to_list(x):
  return x if isinstance(x, list) else [x]


def unpack(x):
  return x[0] if isinstance(x, (list, tuple)) and len(x) == 1 else x

# endregion
