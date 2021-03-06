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

from typing import Any, Dict, List, Optional

import tensorflow as tf
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout

from .. import regularizers
from . import backbones


class DenseKU(Dense):
  """Dense layer with kernel usage regularization.
  """
  def __init__(self, *args, alpha=1., **kwargs):
    super().__init__(*args, **kwargs)
    self.alpha = alpha
  
  def call(self, inputs):
    outputs = tf.matmul(a=inputs, b=self.regularized_kernel)

    if self.use_bias:
      outputs = tf.nn.bias_add(outputs, self.bias)

    if self.activation is not None:
      outputs = self.activation(outputs)

    return outputs
  
  @property
  def regularized_kernel(self):
    kernel = self.kernel
    ag = kernel
    ag = ag - tf.reduce_max(ag, axis=-1, keepdims=True)
    ag = tf.nn.softmax(ag)
    ag *= self.alpha

    return ag * kernel

  def get_config(self):
    config = super().get_config()
    config.update(
      alpha=self.alpha,
    )
    return config


def build_head(
    input_tensor: tf.Tensor,
    units: int,
    activation: Optional[str] = 'linear',
    batch_norm: bool = False,
    dropout_rate: Optional[float] = None,
    layer_class: str = None,
    kernel_initializer: str = 'glorot_uniform',
    kernel_regularizer: str = None,
    config: Optional[Dict[str, Any]] = None,
):
  y = input_tensor
  
  if batch_norm:
    y = BatchNormalization(name='head/bn')(y)
  
  if dropout_rate:
    y = Dropout(rate=dropout_rate, name='head/drop')(y)

  y = get(layer_class)(
    units,
    name='head/logits',
    kernel_initializer=kernel_initializer,
    kernel_regularizer=regularizers.get(kernel_regularizer),
    **(config or {}),
  )(y)
  y = Activation(activation, dtype='float32', name='head/predictions')(y)

  return y


def build_model(
    input_shape: List[int],
    backbone: Dict[str, Any],
    head: Dict[str, Any],
    name: str = None
):
  x = tf.keras.Input(input_shape, name='inputs')
  bb = backbones.get(x, **backbone)
  y = bb(x)
  y = build_head(y, **head)

  return tf.keras.Model(x, y, name=name), bb

def get(identifier):
  if not identifier:
    return tf.keras.layers.Dense
  
  identifier = str(identifier).lower()
  if identifier == 'dense':
    return tf.keras.layers.Dense
  if identifier == 'kernel_usage':
    return DenseKU
  
  raise ValueError(f'Cannot retrieve a classification layer for identifier {identifier}')
