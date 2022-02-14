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
"""
Train and Fine tune Baseline Experiment.

Experiment Outline:

  TFDS dataset → images → CNN → predictions → evaluation → report

Experiment Details:

  1. Experiment Setup [core.experiment.setup]
    GPUs and mixed precision mechanisms are setup;
    logging engines (such as Wandb) are connected.
  
  2. Load TFDS dataset [core.datasets.tfds.load_and_prepare]
    A tfds dataset is load according to its name, and (maybe) shuffled (?);
    the dataset is filtered according to {data.prepare.validation} (?);
    the task {data.prepare.task} is extracted from the tfrecords entries;
    samples are (maybe augmented) using [none|simple|randaug] aug strategy;
  
    2.1 if [data.prepare.augmentation.over] == samples
      the samples in the dataset are augmented;
      the samples are batched;
    2.2 else
      the samples are batched;
      the samples in the dataset are augmented;
    
    the number of batches in the dataset is limited by {data.prepare.take} (?);
    the batches in the dataset are preprocessed using {data.prepare.preprocess_fn} (?);
    the batches are cast to tf.float32.
  
  3. Analysis [core.datasets.save_image_samples]
    samples are saved in disk, for ad-hoc inspection.

  4. Model building [core.models.classification.build_model]
    a {model} is instantiated, containing {model.backbone} as submodel and {model.head}
    as task-specific head.

  5. Model training [core.training.train_head_and_finetune]
    the model's head is trained using the hyperparameters described in {training.config} (?);
    the top-most layers ({training.finetune.unfreeze}) are unfrozen, ignoring (or not) BatchNorm layers;
    the unfrozen layers are again trained according to {training.finetuning}.

  6. Model evaluation [core.testing.evaluate]
    the model is evaluated with respect to the task {evaluation.task};
    the evaluation report is saved at {paths.valid_report}

  7. Teardown [ex.finish]
    wandb logs are synched with the server's copy
    threads from the `tf.distribute.MirrorStrategy` are collected
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

  ex.log_examples(train.take(1), valid.take(1), test.take(1))

  with ex.distributed_strategy.scope():
    model, backbone = core.models.classification.build_model(**model)
    core.models.summary(model)

  model, histories = core.training.train_head_and_finetune(
    model,
    backbone,
    train,
    valid,
    run_params=ex.run_params,
    distributed=ex.distributed_strategy,
    paths=ex.paths,
    **training
  )

  classes = core.datasets.tfds.classes(info)
  evaluations = core.testing.evaluate(
    model,
    test,
    classes,
    **evaluation
  )

  layer = model.get_layer('head/logits')
  weights = getattr(layer, 'regularized_kernel', layer.kernel)

  (ex.log_evaluations(evaluations)
     .log_weights(classes, weights)
     .finish())


if __name__ == '__main__':
  ex.run_commandline()
