from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

from ..utils import unfreeze_top_layers


def head(
    input_tensor: tf.Tensor,
    backbone: tf.keras.Model,
    classes: int,
    activation: Optional[str] = None,
    dropout_rate: Optional[float] = None,
    name: str = None,
):
  y = backbone(input_tensor)
  y = Dropout(rate=dropout_rate, name='top_dropout')(y)
  y = Dense(classes, activation=activation, name='predictions')(y)

  return tf.keras.Model(
    inputs=input_tensor,
    outputs=y,
    name=name
  )


def model(
    input_tensor: tf.Tensor,
    backbone: tf.keras.Model,
    classes: int,
    dropout_rate: Optional[float] = None,
    fine_tune_layers: float = 0,
    freeze_batch_norm: bool = True,
    weights: Optional[str] = None,
    trainable = False,
    name = None
):
  nn = head(
    backbone,
    input_tensor,
    classes,
    dropout_rate=dropout_rate,
    name=name
  )

  unfreeze_top_layers(
    backbone,
    fine_tune_layers,
    freeze_batch_norm
  )

  if weights:
    nn.load_weights(weights)

  nn.trainable = trainable

  return nn
