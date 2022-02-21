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
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import wandb as wandb_lib

from ..utils import attempt, dig, logged, printargs
from ..vis import visualize
from . import sacred_utils


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

  @logged('Experiment Teardown', with_arguments=False)
  def finish(self):
    if self.wandb_run:
      attempt(lambda: self.wandb_run.finish(quiet=True),
              description='closing Wandb.ai log',
              raise_errors=True)

    return self

  def log_examples(self, samples):
    wb_images = {}

    for name, ds in samples.items():
      (x, labels, *other_labels), = ds.take(1).as_numpy_iterator()
      x = (127.5*(x+1.)).clip(0, 255).astype('uint8')

      if self.wandb_run:
        wb_images.update({name: [wandb_lib.Image(img) for img in x]})

      path = dig(self.paths, 'samples_dir')
      if path:
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, name.replace('/', '-') + '.jpg')
        attempt(lambda: visualize(x, rows=4, figsize=(20, 4), to_file=path),
                f'saving {name} samples at `{path}`')
      del x
    
    if self.wandb_run:
      wandb_lib.log(wb_images)

    return self

  def log_evaluations(
      self,
      evaluations: pd.DataFrame,
      path: Optional[str] = None
  ):
    path = path or dig(self.paths, 'evaluation_report')
    if path:
      attempt(lambda: evaluations.to_csv(path),
              f'saving evaluation report at `{path}`')

    if self.wandb_run:
      attempt(lambda: self.wandb_run.log({'evaluations': wandb_lib.Table(dataframe=evaluations.reset_index())}),
              f'saving evaluation at wandb.ai')

      metrics = {
        f'evaluations/test/{k}/macro': v
        for k, v in dict(evaluations.loc['avg_macro'].dropna()).items()
      }

      metrics.update({
        f'evaluations/test/{k}/weighted': v
        for k, v in dict(evaluations.loc['avg_weighted'].dropna()).items()
      })

      self.wandb_run.summary.update(metrics)
  
    return self

  def log_weights(
    self,
    names: List[str],
    weights: Union[tf.Tensor, np.ndarray],
  ):
    if isinstance(weights, tf.Tensor):
      weights = weights.numpy()

    path = dig(self.paths, 'weights_report')
    if path:
      attempt(lambda: np.savetxt(path, weights),
              f'saving evaluation report at `{path}`')

    if self.wandb_run:
      self.wandb_run.log({f'weights/{name}': wandb_lib.Histogram(weights[:, idx])
                          for idx, name in enumerate(names)})

    return self

def set_gpus_to_memory_growth():
  gpus = list(tf.config.list_physical_devices('GPU'))
  for d in gpus:
    tf.config.experimental.set_memory_growth(d, True)

def set_mixed_precision_policy(policy):
  mixed_precision_policy = tf.keras.mixed_precision.Policy(policy)
  tf.keras.mixed_precision.set_global_policy(mixed_precision_policy)
  return mixed_precision_policy


def appropriate_distributed_strategy():
  if tf.config.list_physical_devices('GPU'):
    return tf.distribute.MirroredStrategy()
  else:
    return tf.distribute.get_strategy()


@logged('Experiment Setup', with_arguments=False)
def setup(
    config: Dict[str, Any],
    sacred_run: sacred_utils.Run,
    paths: Dict[str, str],
    tf_seed: Optional[int] = None,
    precision_policy: str = None,
    gpus_with_memory_growth: bool = False,
    wandb: Dict[str, Any] = None,
) -> Experiment:

  np.set_printoptions(linewidth=120)
  pd.set_option('display.max_rows', None)
  pd.set_option('display.max_columns', None)
  pd.set_option('display.width', 200)

  # Override paths with run-specific properties.
  run_params = sacred_utils.get_run_params(sacred_run)
  paths = {k: v.format(**run_params) for k, v in paths.items()}

  print('run params:')
  printargs(kwargs=run_params, indent=0)
  print('paths:')
  printargs(kwargs=paths, indent=0)

  for p, v in paths.items():
    run_params[f'paths.{p}'] = v

  # Setup Wandb connection.
  if wandb:
    wandb = dict(**wandb)
    if 'wandb_dir' in paths:
      os.makedirs(paths['wandb_dir'], exist_ok=True)
      wandb['dir'] = paths['wandb_dir']
    
    wandb_run = attempt(lambda: wandb_lib.init(config=config, **wandb),
                        'initializing wandb.ai run')
  else:
    wandb_run = None
  
  # Setup tensorflow args.
  if tf_seed:
    attempt(
      lambda: tf.random.set_seed(tf_seed),
      f'seeding tf with {tf_seed}')

  if precision_policy:
    mixed_precision_policy = attempt(
      lambda: set_mixed_precision_policy(precision_policy),
      f'setting precision policy to {precision_policy}')
  else:
    mixed_precision_policy = None

  if gpus_with_memory_growth:
    attempt(set_gpus_to_memory_growth)

  distributed_strategy = attempt(appropriate_distributed_strategy, raise_errors=True)

  if isinstance(distributed_strategy, tf.distribute.MirroredStrategy):
    # Fix missing threads not being closed.
    # See https://github.com/tensorflow/tensorflow/issues/50487
    attempt(lambda: atexit.register(distributed_strategy._extended._collective_ops._pool.close),  # type: ignore
            'hooking worker pool close')
  
  return Experiment(
    mixed_precision_policy,
    distributed_strategy,
    gpus_with_memory_growth,
    sacred_run,
    wandb_run,
    config,
    run_params,
    paths
  )
