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

import os
import shutil
import sys

import tensorflow as tf

from .callbacks import get as cb_deserialize
from .utils import dig, log_begin, logged, try_to_load_weights, unfreeze_top_layers


def train_head_and_finetune(
    nn,
    backbone,
    train,
    valid,
    run_params,
    perform,
    loss,
    scale_loss,
    optimizer,
    metrics,
    config,
    callbacks,
    finetune,
    paths,
    distributed,
):
  log_begin('Training Classification Head')

  _, loss, metrics = compile_distributed(nn, loss, scale_loss, optimizer, metrics, distributed)

  histories = []

  if perform:
    try:
      nn.fit(
          train,
          validation_data=valid,
          callbacks=cb_deserialize(callbacks, run_params),
          **config
      )
    except KeyboardInterrupt:
      print('\ninterrupted')
    else:
      print('\ndone')

      if os.path.exists(paths['ckpt']):
        print(f'Cleaning dangling backup folder {paths["ckpt"]}')
        shutil.rmtree(paths['ckpt'], ignore_errors=True)

    histories += nn.history.history
  else:
    print('Training will be skipped (perform=false). Attempting to load '
          f'previously trained model from "{paths["best"]}"')

  if dig(finetune, 'perform'):
    log_begin('Fine-Tuning Entire Network')

    with distributed.scope():
      try_to_load_weights(nn, paths['best'])

    unfreeze_top_layers(backbone, **finetune['unfreeze'])
    compile_distributed(nn, loss, scale_loss, finetune['optimizer'], metrics, distributed)

    try:
      nn.fit(
          train,
          validation_data=valid,
          callbacks=cb_deserialize(finetune['callbacks'], run_params),
          **finetune['config']
      )
    except FileNotFoundError as error:
      print(error, file=sys.stderr)

    except KeyboardInterrupt:
      print('\ninterrupted')
    else:
      print('\ndone')

    histories += nn.history.history
  else:
    print('fine-tuning will be skipped (finetune.perform=false)')

  with distributed.scope():
    try_to_load_weights(nn, paths['best'])

  nn.save(paths['export'], save_format='tf')

  try:
    if os.path.isfile(paths['best']):
      os.remove(paths['best'])
    else:
      shutil.rmtree(paths['best'], ignore_errors=True)
  except FileNotFoundError:
    ...

  return nn, histories


@logged('Distributed Model Compilation', heading=3)
def compile_distributed(
    nn,
    loss,
    scale_loss,
    optimizer,
    metrics,
    distributed,
):
  with distributed.scope():
    loss = tf.losses.get(loss)
    
    optimizer = tf.optimizers.get(dict(optimizer))
    if scale_loss:
      optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    metrics = [tf.metrics.get(m) for m in metrics]
    nn.compile(loss=loss, optimizer=optimizer, metrics=metrics)
  
  return optimizer, loss, metrics


__all__ = [
  'train_head_and_finetune',
  'compile_distributed',
]
