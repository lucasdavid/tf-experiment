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

import os
import shutil
from typing import List

import tensorflow as tf

from .callbacks import get as cb_deserialize
from .utils import unfreeze_top_layers


def train_or_restore(
    nn,
    backbone,
    train,
    valid,
    run_params,
    perform,
    loss,
    optimizer,
    metrics,
    config,
    callbacks,
    finetune,
    paths,
    distributed,
):
  print('-' * 32)
  print('Training Classification Head')

  if not perform:
    print(
        'Training will be skipped (perform=false). Attempting to load '
        f'previously trained model from "{paths["export"]}"'
    )

    nn = tf.keras.models.load(paths['export'])
    return nn, None

  with distributed.scope():
    loss = tf.losses.get(loss)
    optimizer = tf.optimizers.get(dict(optimizer))
    metrics = [tf.metrics.get(m) for m in metrics]

    nn.compile(loss=loss, optimizer=optimizer, metrics=metrics)

  histories = []

  try:
    nn.fit(
        train,
        validation_data=valid,
        callbacks=cb_deserialize(callbacks, run_params),
        **config
    )
  except KeyboardInterrupt:
    print('\n  interrupted')
  else:
    print('\n  done')

  histories += nn.history.history

  print('-' * 32)
  print('Fine-Tuning Entire Network')

  with distributed.scope():
    nn.load_weights(paths['best'])

  unfreeze_top_layers(backbone, **finetune['unfreeze'])

  with distributed.scope():
    optimizer = tf.optimizers.get(dict(finetune['optimizer']))
    nn.compile(loss=loss, optimizer=optimizer, metrics=metrics)

  try:
    nn.fit(
        train,
        validation_data=valid,
        callbacks=cb_deserialize(finetune['callbacks'], run_params),
        **finetune['config']
    )
  except KeyboardInterrupt:
    print('\n  interrupted')
  else:
    print('\n  done')

  histories += nn.history.history

  with distributed.scope():
    nn.load_weights(paths['best'])
  nn.save(paths['export'], save_format='tf')
  
  try:
    if os.path.isfile(paths['best']):
      os.remove(paths['best'])
    else:
      shutil.rmtree(paths['best'])
  except FileNotFoundError:
    ...

  return nn, histories


__all__ = [
    'train_or_restore',
]
