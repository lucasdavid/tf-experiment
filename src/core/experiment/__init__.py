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

import atexit
import os
import sys
from pprint import pprint
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import wandb as wandb_lib

from ..utils import dig, logged
from ..vis import visualize
from .sacred_utils import get_run_params


class Experiment:
  def __init__(
      self,
      mixed_precision_policy: tf.keras.mixed_precision.Policy,
      distributed_strategy: tf.distribute.Strategy,
      gpus_with_memory_growth: bool,
      sacred_run,
      wandb_run,
      config,
      run_params,
      paths,
  ):
    self.mixed_precision_policy = mixed_precision_policy
    self.distributed_strategy = distributed_strategy
    self.gpus_with_memory_growth = gpus_with_memory_growth
    self.sacred_run = sacred_run
    self.wandb_run = wandb_run
    self.config = config
    self.run_params = run_params
    self.paths = paths

  def finish(self):
    if self.wandb_run:
      self.wandb_run.finish(quiet=True)
    
    return self

  def log_examples(self, train=None, test=None, valid=None):
    for name, ds in zip('train valid test'.split(), (train, valid, test)):

      if ds is not None:
        key = f'{name}_samples'

        (x, *_), = ds.take(1).as_numpy_iterator()
        x = (127.5*(x+1.)).clip(0, 255).astype('uint8')

        # save in wandb
        wandb_lib.log({f'samples/{name}': [wandb_lib.Image(img) for img in x]})

        # save in disk
        path = dig(self.paths, key)
        if path:
          try: visualize(x, rows=4, figsize=(20, 4), to_file=path)
          except Exception as ex: print(f'failed to save image samples: {len(ex)}', sys.stderr)
        del x

  def log_evaluations(
      self,
      evaluations: pd.DataFrame,
  ):
    path = dig(self.paths, 'evaluation_report')
    if path:
      print(f'Saving evaluation report at {path}')
      evaluations.to_csv(path, index=False)

    if self.wandb_run:
      self.wandb_run.log({
        'evaluations': wandb_lib.Table(dataframe=evaluations)
      })
  
    return self

  def log_weights(
    self,
    names: List[str],
    weights: tf.Tensor,
  ):
    weights = weights.numpy()

    path = dig(self.paths, 'weights_report')
    if path:
      print(f'Saving evaluation report at {path}')
      np.savetxt(path, weights)

    if self.wandb_run:
      self.wandb_run.log({f'weights/{name}': wandb_lib.Histogram(weights[:, idx])
                          for idx, name in enumerate(names)})

    return self

def set_gpus_to_memory_growth():
  print('  setting gpus to memory growth mode')

  gpus = list(tf.config.list_physical_devices('GPU'))

  print(f'    devices: {len(gpus)}')

  for d in gpus:
    print(d)
    print(f'    setting device {d} to memory-growth mode...', end=' ')

    try:
      tf.config.experimental.set_memory_growth(d, True)
    except Exception as e:
      print('failure')
      print(e)
    else:
      print('done')


def appropriate_distributed_strategy():
  if tf.config.list_physical_devices('GPU'):
    return tf.distribute.MirroredStrategy()
  else:
    return tf.distribute.get_strategy()


@logged('Experiment Setup', with_arguments=False)
def setup(
    sacred_run,
    experiment_config: Dict[str, Any],
    paths: Dict[str, str],
    tf_seed: Optional[int] = None,
    precision_policy: str = None,
    gpus_with_memory_growth: bool = False,
    wandb: Dict[str, Any] = None,
) -> Experiment:
  # Override paths with run-specific properties.
  run_params = get_run_params(sacred_run)
  paths = {k: v.format(**run_params) for k, v in paths.items()}

  print('  run params:')
  pprint(run_params, indent=4)
  print('  paths:')
  pprint(paths, indent=4)

  for p, v in paths.items():
    run_params[f'paths.{p}'] = v

  # Setup Wandb connection.
  if wandb:
    wandb = dict(**wandb)
    if 'wandb_dir' in paths:
      os.makedirs(paths['wandb_dir'], exist_ok=True)
      wandb['dir'] = paths['wandb_dir']

    wandb_run = wandb_lib.init(config=experiment_config, **wandb)
  else:
    wandb_run = None

  # Setup tensorflow args.
  if tf_seed:
    print('  seeding tf with', tf_seed)
    tf.random.set_seed(tf_seed)

  if precision_policy:
    print('  setting precision policy to', precision_policy)

    mixed_precision_policy = tf.keras.mixed_precision.Policy(precision_policy)
    tf.keras.mixed_precision.set_global_policy(mixed_precision_policy)
  else:
    mixed_precision_policy = None

  if gpus_with_memory_growth:
    set_gpus_to_memory_growth()

  distributed_strategy = appropriate_distributed_strategy()

  if isinstance(distributed_strategy, tf.distribute.MirroredStrategy):
    # Fix missing threads not being closed.
    # See https://github.com/tensorflow/tensorflow/issues/50487
    atexit.register(distributed_strategy._extended._collective_ops._pool.close) # type: ignore

  return Experiment(
    mixed_precision_policy,
    distributed_strategy,
    gpus_with_memory_growth,
    sacred_run,
    wandb_run,
    experiment_config,
    run_params,
    paths
  )
