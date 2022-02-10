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

import tensorflow as tf
import wandb.keras


def get(identifier, override=None):
  if identifier is None or isinstance(identifier, tf.keras.callbacks.Callback):
    return identifier

  if isinstance(identifier, tuple):
    cls, config = identifier
  elif isinstance(identifier, str):
    cls, config = identifier, {}
  elif isinstance(identifier, dict):
    cls, config = identifier['class_name'], identifier['config']
  elif isinstance(identifier, list):
    return [get(i, override) for i in identifier]
  else:
    raise ValueError(
        f'Cannot build a keras callback from the identifier {identifier}.'
    )

  config = dict(config)

  if 'override' in identifier:
    for k, v in identifier['override'].items():
      config[k] = override[v]

  cls = (
      getattr(tf.keras.callbacks, cls, None)
      or getattr(tf.keras.callbacks.experimental, cls, None)
      or getattr(wandb.keras, cls, None)
  )

  return cls(**(config or {}))
