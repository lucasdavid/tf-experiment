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

  TFDS dataset → images → CNN → activation-fn → predictions

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


from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

import core

ex = Experiment(save_git_info=False)
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.main
def run(setup, dataset, model, training, evaluation, _log, _run):
  _log.info(__doc__)

  experiment_config = {
    'setup': setup,
    'dataset': dataset,
    'model': model,
    'training': training,
    'evaluation': evaluation
  }
  ex = core.experiment.setup(_run, experiment_config, **setup)

  train, valid, test, info = core.datasets.tfds.load_and_prepare(dataset)

  core.datasets.save_image_samples(train.take(1), ex.paths['train_samples'])
  core.datasets.save_image_samples(valid.take(1), ex.paths['valid_samples'])
  core.datasets.save_image_samples(test.take(1), ex.paths['test_samples'])

  with ex.distributed_strategy.scope():
    model, backbone = core.models.classification.build_model(**model)
    core.models.summary(model, _log.info)

  model, histories = core.training.train_or_restore(
    model,
    backbone,
    train,
    valid,
    run_params=ex.run_params,
    distributed=ex.distributed_strategy,
    paths=ex.paths,
    **training
  )

  evaluations = core.testing.evaluate(
    model,
    test,
    task=evaluation['task'],
    classes=core.datasets.tfds.classes(info),
    report_path=evaluation['report_path'].format(**ex.paths)
  )

  ex.wandb_run.finish()


if __name__ == '__main__':
  ex.run_commandline()
