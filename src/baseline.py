"""Baseline Experiment.

Strategy Description:

  - MNIST dataset
  - ResNet50

Copyright 2021 Lucas Oliveira David

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

from core import datasets, models, testing, training
from core.boot import *
from core.utils import *

DS: tf.distribute.Strategy = None
R: tf.random.Generator = None

ex = Experiment(save_git_info=False)
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.main
def run(setup, dataset, model, evaluation, _log):
  global DS, R

  _log.info(__doc__)

  gpus_with_memory_growth()
  DS = appropriate_distributed_strategy()
  R = tf.random.Generator.from_seed(setup['tf_seed'], alg='philox')

  train, valid, test, info = datasets.tfds.load_and_prepare(dataset, randgen=R)
  classes = datasets.tfds.classes(info)

  with DS.scope():
    inputs = tf.keras.Input(model['input_shape'], name='images')
    backbone = models.backbone.get(inputs, **model['backbone'])
    model = models.classification.head(inputs, backbone, **model['head'])

  models.summary(model, _log.info)

  model, history = train_or_restore(model, backbone, train.take(2), valid.take(2))
  
  # Evaluation
  labels, probabilities = testing.labels_and_probs(model, test.take(2))
  evaluations = testing.metrics_per_label(labels, probabilities, classes=classes)

  mcm = evaluations.pop('multilabel_confusion_matrix')
  report = pd.DataFrame(evaluations)

  report.to_csv(evaluation['report_path'], index=False)

  print(report.round(4))


@ex.capture(prefix='training')
def train_or_restore(
    nn,
    backbone,
    train,
    valid,
    perform,
    loss,
    metrics,
    optimizer,
    epochs,
    callbacks,
    verbose,
    paths,
    finetune,
    _log,
):
  global DS

  _log.info('-' * 32)
  _log.info('Training Classification Head')

  if not perform:
    _log.info('Training will be skipped (perform=false). Attempting to load '
              f'previously trained model from "{paths["export"]}"')

    nn = tf.keras.models.load(paths['export'])
    return nn, None

  with DS.scope():
    loss = tf.losses.get(loss)
    optimizer = tf.optimizers.get(dict(optimizer))
    metrics = [tf.metrics.get(m) for m in metrics]

    nn.compile(loss=loss, optimizer=optimizer, metrics=metrics)

  history = training.run(
    nn,
    train,
    valid,
    epochs,
    verbose=verbose,
    callbacks=[training.get_callback(c) for c in callbacks],
  )

  _log.info('-' * 32)
  _log.info('Fine-Tuning Entire Network')

  with DS.scope():
    nn.load_weights(paths['best'])

  unfreeze_top_layers(backbone, finetune['training_layers'], finetune['freeze_bn'])

  with DS.scope():
    optimizer = tf.optimizers.get(dict(finetune['optimizer']))
    nn.compile(loss=loss, optimizer=optimizer, metrics=metrics)

  history = training.run(
    nn,
    train,
    valid,
    verbose=verbose,
    epochs=finetune['epochs'],
    initial_epoch=finetune['initial_epoch'],
    callbacks=[training.get_callback(c) for c in callbacks],
  )

  with DS.scope():
    nn.load_weights(paths['best'])
  
  nn.save(paths['export'], save_format='tf')

  return nn, history


if __name__ == '__main__':
  ex.run_commandline()
