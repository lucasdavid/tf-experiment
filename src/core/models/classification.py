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

from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense, Dropout

from .. import regularizers


class DenseKU(Dense):
  """Dense layer with kernel usage regularization.
  """
  def call(self, inputs):
    kernel = self.kernel
    ag = kernel
    ag = ag - tf.reduce_max(ag, axis=-1, keepdims=True)
    ag = tf.nn.softmax(ag)

    outputs = tf.matmul(a=inputs, b=ag * kernel)

    if self.use_bias:
      outputs = tf.nn.bias_add(outputs, self.bias)

    if self.activation is not None:
      outputs = self.activation(outputs)

    return outputs


def head(
    input_tensor: tf.Tensor,
    backbone: tf.keras.Model,
    units: int,
    activation: Optional[str] = 'linear',
    dropout_rate: Optional[float] = None,
    layer_class: str = None,
    kernel_initializer: str = None,
    kernel_regularizer: str = None,
    name: str = None,
):
  y = backbone(input_tensor)
  y = Dropout(rate=dropout_rate, name='head/drop')(y)
  y = get(layer_class)(
    units,
    name='head/logits',
    kernel_initializer=kernel_initializer,
    kernel_regularizer=regularizers.get(kernel_regularizer)
  )(y)
  y = Activation(activation, dtype='float32', name='head/predictions')(y)

  return tf.keras.Model(
    inputs=input_tensor,
    outputs=y,
    name=name
  )


def get(identifier):
  if not identifier:
    return tf.keras.layers.Dense
  
  identifier = str(identifier).lower()
  if identifier == 'dense':
    return tf.keras.layers.Dense
  if identifier == 'kernel_usage':
    return DenseKU
  
  raise ValueError(f'Cannot retrieve a classification layer for identifier {identifier}')
