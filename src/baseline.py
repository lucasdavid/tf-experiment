"""Baseline Experiment.

Strategy Description:

  - MNIST dataset
  - ResNet50
"""

import tensorflow as tf
from keras.utils import layer_utils
from sacred import Experiment

from core import models
from core.boot import *
from core.datasets import tfds
from core.training import *
from core.utils import *

DS: tf.distribute.Strategy = None
R: tf.random.Generator = None

ex = Experiment(
  save_git_info=False
)

@ex.main
def run(_log):
  _log.info(__doc__)

  setup()
  train, valid, test = retrieve_dataset()
  model, history = compile_and_train_or_restore(train, valid)

  _log.info(history)


@ex.capture(prefix='setup')
def setup(tf_seed, _log):
  global DS, R

  _log.info('-' * 32)
  _log.info('Setup')

  gpus_with_memory_growth()

  DS = appropriate_distributed_strategy()
  R = tf.random.Generator.from_seed(tf_seed, alg='philox')


@ex.capture(prefix='dataset')
def retrieve_dataset(
    data_dir,
    name,
    splits,
    batch_size,
    augment,
    aug,
    buffer_size,
    parallel_calls,
    preprocess_fn,
    drop_remainder,
    _log,
):
  global R

  parts, info = tfds.load(name, data_dir, splits)
  
  _log.info('-' * 32)
  _log.info(f'Dataset {info.name}')
  _log.info(info.description)

  preprocess_fn = get_preprocess_fn(preprocess_fn)

  return (
    tfds.prepare(part, batch_size, augment, aug, buffer_size, parallel_calls, preprocess_fn, drop_remainder, R)
    for part in parts
  )


@ex.capture(prefix='model')
def create_model(
    input_shape,
    backbone,
    units,
    activation,
    dropout_rate,
    name,
    _log
):
  _log.info('-' * 32)

  inputs = tf.keras.Input(input_shape, name='images')

  model = models.classification.head(
    inputs,
    backbone=models.backbone.get(inputs, **backbone),
    classes=units,
    activation=activation,
    dropout_rate=dropout_rate,
    name=name
  )

  _log.info(f'Model {model.name}')
  _log.info(' →  '.join(f'{l.name} ({type(l).__name__})' for l in model.layers))
  
  trainable_params = layer_utils.count_params(model.trainable_weights)
  non_trainable_params = layer_utils.count_params(model.non_trainable_weights)
  _log.info(f'Total params:     {trainable_params + non_trainable_params}')
  _log.info(f'Trainable params: {trainable_params}')

  return model


@ex.capture(prefix='training')
def compile_and_train_or_restore(
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
    _log
):
  global DS

  _log.info('-' * 32)

  if not perform:
    _log.info('Training will be skipped (perform=false). Attempting to load '
              f'previously trained model from "{paths["export"]}"')

    nn = tf.keras.models.load(paths['export'])
    return nn, None
  
  with DS.scope():
    nn = create_model()

    nn.compile(
      optimizer=tf.optimizers.get(dict(optimizer)),
      loss=tf.losses.get(loss),
      metrics=metrics,
    )

  if callbacks:
    callbacks = [tf.keras.callbacks.get(c) for c in callbacks]

  history = train_fn(nn, train, valid, epochs, callbacks=callbacks, verbose=verbose)

  _log.info(f'exporting model to {paths["export"]}')
  nn.save(paths['export'], save_format='tf')

  return nn, history


if __name__ == '__main__':
  ex.run_commandline()
