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

import tensorflow as tf
from keras.utils.layer_utils import count_params
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

from core import models, testing, training, datasets
from core.boot import *
from core.utils import *

DS: tf.distribute.Strategy = None
R: tf.random.Generator = None

ex = Experiment(save_git_info=False)
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.main
def run(_log, model):
  _log.info(__doc__)

  setup()
  
  train, valid, test = retrieve_dataset()
  model, backbone = create_or_retrieve_model()
  model, history = train_or_restore(model, backbone, train, valid)
  
  evaluate(model, test)

  _log.info(history)


@ex.capture(prefix='setup')
def setup(tf_seed, _log):
  global DS, R
  _log.info('-' * 32)

  gpus_with_memory_growth()

  DS = appropriate_distributed_strategy()
  R = tf.random.Generator.from_seed(tf_seed, alg='philox')


@ex.capture(prefix='dataset')
def retrieve_dataset(
    data_dir,
    name,
    sizes,
    batch_size,
    buffer_size,
    parallel_calls,
    drop_remainder,
    task,
    classes,
    splits,
    augmentation,
    preprocess_fn,
    _log,
):
  global R

  parts, info = datasets.tfds.load(name, data_dir, splits)

  _log.info('-' * 32)
  _log.info(f'{info.name}: {info.description}')

  kwargs = dict(
    batch_size=batch_size,
    sizes=sizes,
    task=task,
    classes=classes,
    aug_policy=augmentation['policy'],
    aug_config=augmentation['config'],
    buffer_size=buffer_size,
    parallel_calls=parallel_calls,
    preprocess_fn=get_preprocess_fn(preprocess_fn),
    drop_remainder=drop_remainder,
    randgen=R
  )

  train, valid, test = (datasets.tfds.prepare(part, **kwargs)
                        for part in parts)

  print(f'train: {train}')
  print(f'valid: {valid}')
  print(f'test:  {test}')

  return train, valid, test

@ex.capture(prefix='model')
def create_or_retrieve_model(input_shape, backbone, units, activation, dropout_rate, name, _log):
  _log.info('-' * 32)

  with DS.scope():
    inputs = tf.keras.Input(input_shape, name='images')
    backbone_nn = models.backbone.get(inputs, **backbone)
    model = models.classification.head(inputs, backbone_nn, units, activation, dropout_rate, name)

  _log.info(f'Model {model.name}')
  _log.info(' →  '.join(f'{l.name} ({type(l).__name__})' for l in model.layers))

  trainable_params = count_params(model.trainable_weights)
  non_trainable_params = count_params(model.non_trainable_weights)
  
  _log.info(f'Total params:     {trainable_params + non_trainable_params}')
  _log.info(f'Trainable params: {trainable_params}')

  return model, backbone_nn


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

  if not perform:
    _log.info('Training will be skipped (perform=false). Attempting to load '
              f'previously trained model from "{paths["export"]}"')

    nn = tf.keras.models.load(paths['export'])
    return nn, None

  loss = tf.losses.get(loss)
  optimizer = tf.optimizers.get(dict(optimizer))
  metrics = [tf.metrics.get(m) for m in metrics]
  callbacks = [training.get_callback(c) for c in callbacks] if callbacks else []

  with DS.scope():
    nn.compile(loss=loss, optimizer=optimizer, metrics=metrics)

  _log.info(f'Head training for {epochs} epochs.')
  history = training.run(nn, train, valid, epochs, callbacks=callbacks, verbose=verbose)

  _log.info(f'loading best weights from {paths["best"]}')
  nn.load_weights(paths['best'])

  unfreeze_top_layers(backbone, finetune['training_layers'], finetune['freeze_bn'])

  with DS.scope():
    optimizer = tf.optimizers.get(dict(finetune['optimizer']))
    nn.compile(loss=loss, optimizer=optimizer, metrics=metrics)

  _log.info(f'Fine-tuning for {finetune["epochs"]} epochs.')
  
  history = training.run(
    nn,
    train,
    valid,
    epochs=finetune['epochs'],
    initial_epoch=finetune['initial_epoch'],
    callbacks=callbacks,
    verbose=verbose
  )

  _log.info(f'loading best weights from {paths["best"]}')
  nn.load_weights(paths['best'])

  _log.info(f'exporting model to {paths["export"]}')
  nn.save(paths['export'], save_format='tf')

  return nn, history


def evaluate(model, test):
  labels, probabilities = testing.labels_and_probs(model, test)


if __name__ == '__main__':
  ex.run_commandline()
