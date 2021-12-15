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
Baseline Experiment.

Example outline:

  TFDS dataset → images → ResNet50 → softmax → predictions

  The {dataset.name} dataset is loaded from TFDS, and a 
  {dataset.prepare.task} is extracted from it, resulting in a
  `tf.data.Dataset` containing the supervision pair (images, labels).

  A {model} is instantiated, containing {model.backbone} as submodel.
  The model's head is trained using the hyperparameters described
  in {training.config}.

  Subsequentially, the top-most layers ({training.finetune.unfreeze}) are
  unfrozen and the entire stack is once again trained (considering the
  definitions in {training.finetuning}).

  The model is evaluated.
"""

from typing import Dict

import tensorflow as tf
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

import core

DS: tf.distribute.Strategy = None
R: tf.random.Generator = None
PATHS: Dict[str, str] = None

ex = Experiment(save_git_info=False)
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.main
def run(setup, dataset, model, training, evaluation, _log, _run):
  global DS, R, PATHS

  _log.info(__doc__)

  # region Setup
  run_params = core.utils.get_run_params(_run)

  core.boot.gpus_with_memory_growth()
  DS = core.boot.appropriate_distributed_strategy()
  R = tf.random.Generator.from_seed(setup['tf_seed'], alg='philox')
  PATHS = {k: v.format(**run_params) for k, v in setup['paths'].items()}

  for p, v in PATHS.items():
    run_params[f'paths.{p}'] = v
  # endregion

  # region Dataset
  train, valid, test, info = core.datasets.tfds.load_and_prepare(dataset, randgen=R)
  classes = core.datasets.tfds.classes(info)
  # endregion

  # region Model
  with DS.scope():
    inputs = tf.keras.Input(model['input_shape'], name='images')
    backbone = core.models.backbone.get(inputs, **model['backbone'])
    model = core.models.classification.head(inputs, backbone, **model['head'])
    core.models.summary(model, _log.info)
  # endregion

  # region Training and Evaluation
  model, histories = core.training.train_or_restore(
      model,
      backbone,
      train,
      valid,
      run_params=run_params,
      distributed=DS,
      paths=PATHS,
      **training
  )

  target_and_output = core.inference.target_and_output(model, test)

  core.testing.report(
      target_and_output,
      classes=classes,
      run_params=run_params,
      task=evaluation['task'],
      report_path=evaluation['report_path'].format(**PATHS)
  )
  # endregion


if __name__ == '__main__':
  ex.run_commandline()
