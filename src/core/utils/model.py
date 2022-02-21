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
from typing import Union

import tensorflow as tf


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
      l.trainable = (
          ix > idx
          and (not isinstance(l, tf.keras.layers.BatchNormalization) or not freeze_bn)
      )

  print(
      f'Unfreezing {1-idx/len(model.layers):.0%} of the model\'s layers '
      f'(layers={layers} freeze_bn={freeze_bn}). Bottom-most is the '
      f'{idx}-nth layer ({model.layers[idx].name}).'
  )


def try_to_load_weights(model, weights, raise_on_failure: bool = False):
  try:
    model.load_weights(weights)
  except (OSError, FileNotFoundError):
    if raise_on_failure:
      raise

    print(f'Cannot restore weights from "{weights}".')


def get_preprocess_fn(preprocess_fn):
  if isinstance(preprocess_fn, str):
    *mod, fn_name = preprocess_fn.split('.')
    mod = '.'.join(mod)
    if mod:
      return getattr(sys.modules[mod], fn_name)
    return globals[fn_name]
  
  return preprocess_fn
