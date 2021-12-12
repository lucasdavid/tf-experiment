from typing import List, Optional

import tensorflow as tf

from .callbacks import get as get_callback


def run(
    nn: tf.keras.Model,
    train: tf.data.Dataset,
    valid: tf.data.Dataset,
    epochs: int,
    train_steps: Optional[int] = None,
    valid_steps: Optional[int] = None,
    initial_epoch: int = 0,
    callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    verbose: int = 1,
):
  try:
    nn.fit(
      train,
      validation_data=valid,
      epochs=epochs,
      initial_epoch=initial_epoch,
      callbacks=callbacks,
      steps_per_epoch=train_steps,
      validation_steps=valid_steps,
      verbose=verbose,
    )
  except KeyboardInterrupt:
    print('\n  interrupted')
  else:
    print('\n  done')

  return nn.history.history


__all__ = [
  'run',
  'get_callback',
]
