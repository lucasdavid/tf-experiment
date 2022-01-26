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
"""
Salient Segmentation Experiment -- Evaluation

"""

import pandas as pd
import tensorflow as tf
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

import core

ex = Experiment(save_git_info=False)
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.main
def run(setup, dataset, evaluation, _log, _run):
  _log.info(__doc__)

  # region Setup
  if not len(evaluation['splits']):
    raise ValueError('Set at least one split to be evaluated.')

  if 'precision_policy' in setup and setup['precision_policy']:
    precision_policy = tf.keras.mixed_precision.Policy(setup['precision_policy'])
    tf.keras.mixed_precision.set_global_policy(precision_policy)

  if setup['gpus_with_memory_growth']:
    core.boot.gpus_with_memory_growth()

  strategy = core.boot.appropriate_distributed_strategy()

  run_params = core.utils.get_run_params(_run)
  paths = {k: v.format(**run_params) for k, v in setup['paths'].items()}
  for p, v in paths.items():
    run_params[f'paths.{p}'] = v
  # endregion

  # region Dataset
  train, valid, test, info = core.datasets.tfds.load_and_prepare(dataset)

  to_file = f'{run_params["report_dir"]}/images.jpg'
  (x, y), = train.take(1).as_numpy_iterator()
  core.utils.visualize((127.5*(x+1.)).astype('uint8'), rows=4, figsize=(20, 4), to_file=to_file)

  del x, y
  # endregion

  # region Model Restoration
  with strategy.scope():
    model = tf.keras.models.load_model(paths['export'])
    core.models.summary(model, _log.info)
  # endregion

  # region Evaluation
  names = ('train', 'validation', 'test')
  splits = (train.take(1), valid.take(1), test.take(1))
  evaluations = []

  for name, split in zip(names, splits):
    if name in evaluation['splits'] and evaluation['splits'][name]:
      evaluations.append(
        core.testing.evaluate(
          model,
          split,
          task=evaluation['task'],
          classes=core.datasets.tfds.classes(info),
        ).assign(split=name)
      )

  pd.concat(evaluations).to_csv(evaluation['report_path'].format(**paths), index=False)
  # endregion


if __name__ == '__main__':
  ex.run_commandline()
