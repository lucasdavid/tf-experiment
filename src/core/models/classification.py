from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout


def head(
    input_tensor: tf.Tensor,
    backbone: tf.keras.Model,
    units: int,
    activation: Optional[str] = None,
    dropout_rate: Optional[float] = None,
    name: str = None,
):
  y = backbone(input_tensor)
  y = Dropout(rate=dropout_rate, name='top_dropout')(y)
  y = Dense(units, activation=activation, name='predictions')(y)

  return tf.keras.Model(
    inputs=input_tensor,
    outputs=y,
    name=name
  )
